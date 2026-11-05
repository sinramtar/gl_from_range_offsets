"""Modelled tides: CATS2008 prediction, the IBE air-pressure correction, the tidal range,
and the acquisition-time/geometry helpers the tide products are built on.

Requires pyTMD 3.x, whose grid interpolation accepts only 'nearest' and 'linear' (see
`INTERP_METHOD`). pyTMD resolves the model as `<cats>/CATS2008_v2023/CATS2008_v2023.nc`,
so the netCDF must sit in a subdirectory named for the model. Chunked model reads
(`CHUNKS`) and `extrapolate=True` both require dask.

Sea surface elevation is the astronomical tide plus the inverse-barometer elevation
`(p0 - p) / (rho * g)`, which is positive under low pressure: a 1 hPa pressure drop
raises the surface by roughly 1 cm. `get_corrected_tide` returns that sum on a grid.
"""

from datetime import datetime
from functools import lru_cache
import numpy as np
import geopandas as gpd
import requests
from shapely.geometry import shape

import pandas as pd
import pathlib

import pyTMD
from pyTMD.compute import tide_elevations


import rasterio as rio

from scipy.interpolate import RegularGridInterpolator

import urllib
from dateutil import parser


CATS_MODEL = 'CATS2008-v2023'

# grid interpolation for every tide lookup; nearest keeps each sample tied to one model
# cell rather than blending across the land/ocean mask boundary at the grounding line
INTERP_METHOD = 'nearest'

# margin kept around a cropped model region, in the model's CRS units (km for CATS2008),
# so interpolation at the crop edge has neighbouring cells
CROP_BUFFER_M = 50

# per-dimension chunks for the tide model read; keeps the constituent grids lazy so a
# prediction never materialises the full (10, 4051, 3325) cube
CHUNKS = {'x': 256, 'y': 256}

# Inverse barometer effect. A 1 hPa drop in air pressure raises the sea surface by
# roughly 1 cm; the ocean response is `-(p - p0) / (rho * g)`.
ACC_GRAVITY = 9.8              # m/s^2
DENSITY_SEAWATER = 1026        # kg/m^3
SEA_LEVEL_PRESSURE = 101325    # Pa, the reference pressure the anomaly is taken against

# NCEP/NCAR sea level pressure files count hours from here
NCEP_EPOCH = np.datetime64('1800-01-01')

# latitude margin kept around the requested points when the pressure cube is subset, so
# the interpolator still has neighbouring rows at the edge
LAT_PAD = 7.5

# cached pressure cubes and interpolators, per process; a handful of entries covers the
# usual case of one latitude extent spanning a few years
CACHE_SIZE = 4


def ibe_elevation(pressure: np.ndarray) -> np.ndarray:
    """Inverse-barometer elevation (m) from sea level pressure (Pa).

    Positive under low pressure: `(p0 - p) / (rho * g)`, about +1 cm per hPa below
    `SEA_LEVEL_PRESSURE`. Add it to an astronomical tide to get the sea surface.
    """
    return (SEA_LEVEL_PRESSURE - np.asarray(pressure, dtype = float)) / (
        DENSITY_SEAWATER * ACC_GRAVITY)


def get_tidal_amplitude(longs: np.ndarray, lats: np.ndarray, cats: pathlib.Path, model: str = CATS_MODEL,
                        type: str = 'drift', standard: str = 'GPS', delta_time: np.ndarray = None,
                        method: str = INTERP_METHOD, apply_flexure: bool = False,
                        extrapolate: bool = False, crop: bool = False,
                        buffer: float = CROP_BUFFER_M, chunks: dict = CHUNKS) -> np.ma.array:
    """Tide elevations at projected (EPSG:3031) coordinates, one epoch per point.

    Parameters
    ----------
    longs, lats : np.ndarray
        coordinates in EPSG:3031 (x and y in metres, despite the names)
    cats : pathlib.Path
        directory holding the tide model, laid out as `<cats>/CATS2008_v2023/CATS2008_v2023.nc`
    model : str
        entry in pyTMD's model database, by default CATS2008-v2023
    type : {'drift', 'grid', 'time series'}
        'drift' pairs each coordinate with its own epoch; 'grid' evaluates a grid at one
        epoch; 'time series' is one point over many epochs.
    standard : str
        time standard of `delta_time`; 'datetime' when passing datetime64
    delta_time : np.ndarray
        epoch(s) to evaluate at
    method : {'nearest', 'linear'}
        grid interpolation method
    apply_flexure : bool
        scale each constituent by the model's ice flexure field. Off by default, so a
        prediction is the open-water tide; a series taken with it on is not comparable
        with one taken with it off.
    extrapolate : bool
        fill points outside the model's ocean mask from the nearest valid cell. Off by
        default, so points off the model's ocean domain come back masked rather than
        silently filled from a neighbour. Requires dask.
    crop : bool
        subset the model to the requested coordinates before interpolating. Leave off:
        `ds.tmd.crop` calls `ds.chunk(-1).compute()`, which materialises the whole
        model and so defeats `chunks`.
    buffer : float
        margin (model CRS units) kept around the crop, so interpolation near the edge
        still has neighbouring cells to work with. Only used when `crop` is True.
    chunks : dict
        per-dimension chunking for the model read; keeps the constituent grids lazy.
        Requires dask. Pass None to read eagerly.

    Returns
    -------
    np.ma.array
        tide elevations (m), masked off the model's ocean domain
    """
    return tide_elevations(x = longs, y = lats, delta_time = delta_time, directory = cats,
                           model = model, crs = 3031, type = type, standard = standard,
                           method = method, extrapolate = extrapolate,
                           apply_flexure = apply_flexure, crop = crop, buffer = buffer,
                           chunks = chunks)


def _slp_files(dataset_dir: str, years: tuple) -> list:
    """Existing `slp.<year>.nc` files for `years`, oldest first."""
    dataset_dir = pathlib.Path(dataset_dir)
    return [p for p in (dataset_dir / f'slp.{year}.nc' for year in sorted(years))
            if p.exists()]


def _slp_years(when) -> tuple:
    """Years whose pressure files are needed to interpolate `when`.

    The neighbouring years are included because interpolation near the first or last
    record of a year reaches across the boundary.
    """
    years = set(pd.DatetimeIndex(when).year)
    years |= {min(years) - 1, max(years) + 1}
    return tuple(sorted(years))


@lru_cache(maxsize = CACHE_SIZE)
def _slp_cube(dataset_dir: str, years: tuple, lat_min: float, lat_max: float) -> tuple:
    """Load the NCEP pressure cube for `years`, cropped in latitude.

    Returns `((time_hours, lat, lon), cube)`, all fully materialised, with latitude
    ascending and longitude wrapped so 360 deg duplicates the first column. Cached per
    process, so repeated lookups over the same extent read the files once.
    """
    import xarray as xr

    paths = _slp_files(dataset_dir, years)
    if not paths:
        raise FileNotFoundError(
            f'{dataset_dir}: no slp.<year>.nc files covering years {years}')

    times, cubes, lat_axis, lon_axis, keep = [], [], None, None, None
    for path in paths:
        with xr.open_dataset(path) as ds:
            if lat_axis is None:
                # NCEP latitude runs north -> south; flip it so the axis ascends.
                lat = ds.lat.values[::-1]
                keep = np.where((lat >= lat_min - LAT_PAD)
                                & (lat <= lat_max + LAT_PAD))[0]
                lat_axis = lat[keep]
                lon_axis = np.append(ds.lon.values, 360.0)

            pressure = ds.slp.values[:, ::-1, :][:, keep, :]
            # Wrap the first longitude column round to 360 deg so a point between the
            # last grid meridian and 0 deg still interpolates.
            cubes.append(np.concatenate([pressure, pressure[:, :, :1]], axis = 2))
            times.append((ds.time.values - NCEP_EPOCH) / np.timedelta64(1, 'h'))

    return (np.concatenate(times), lat_axis, lon_axis), np.concatenate(cubes, axis = 0)


@lru_cache(maxsize = CACHE_SIZE)
def _slp_interpolator(dataset_dir: str, years: tuple, lat_min: float, lat_max: float):
    """Linear (time, lat, lon) interpolator over the NCEP pressure cube."""
    axes, cube = _slp_cube(dataset_dir, years, lat_min, lat_max)
    return RegularGridInterpolator(axes, cube, method = 'linear',
                                   bounds_error = False, fill_value = np.nan)


@lru_cache(maxsize = 8)
def _to_geographic(crs):
    """Transformer from `crs` to EPSG:4326, x/y ordered."""
    from pyproj import Transformer
    return Transformer.from_crs(crs, 4326, always_xy = True)


def sea_level_pressure(when, x, y, ap_model: pathlib.Path, crs = 3031,
                       lat_bounds: tuple = None) -> np.ndarray:
    """Sea level pressure (Pa) interpolated to points and epochs.

    `when`, `x` and `y` are broadcast against each other, so this serves both one point
    over many epochs and many points each with their own epoch.

    Parameters
    ----------
    when : array-like of datetime
        epochs to interpolate at
    x, y : array-like
        coordinates in `crs`
    ap_model : pathlib.Path
        directory of NCEP/NCAR files, named `slp.<year>.nc`
    crs : int or str
        CRS of `x` and `y`
    lat_bounds : tuple, optional
        (lat_min, lat_max) used to crop the cached cube. Pass the extent of the whole
        job so every call shares one cache entry; inferred from the points otherwise.

    Returns
    -------
    np.ndarray
        pressure (Pa), flattened to the broadcast shape
    """
    when = pd.DatetimeIndex(when)
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    lon, lat = _to_geographic(crs).transform(x.ravel(), y.ravel())
    lon = np.asarray(lon) % 360.0
    lat = np.asarray(lat)

    if lat_bounds is None:
        lat_min, lat_max = float(lat.min()), float(lat.max())
    else:
        lat_min, lat_max = float(lat_bounds[0]), float(lat_bounds[1])

    interpolator = _slp_interpolator(str(pathlib.Path(ap_model).resolve()),
                                     _slp_years(when), lat_min, lat_max)
    hours = (when.to_numpy(dtype = 'datetime64[ns]') - NCEP_EPOCH) / np.timedelta64(1, 'h')
    hours, lat, lon = np.broadcast_arrays(np.asarray(hours), lat, lon)
    return interpolator(np.column_stack((hours.ravel(), lat.ravel(), lon.ravel())))


def get_differential_tides_time_series(acq_times, tidesdir, shape: tuple = None):
    """Differential tide field per acquisition pair: primary tide minus secondary tide.

    A pair whose per-timestamp tide rasters are missing yields an all-NaN plane rather
    than raising, since the per-timestamp rasters exist only for the configured baseline
    (6-day) while some callers stack 6-day and 12-day pairs.

    Parameters
    ----------
    acq_times : array-like of (primary, secondary) datetimes
        one row per acquisition pair; the output keeps this order
    tidesdir : pathlib.Path
        directory of per-timestamp tide rasters, named `<YYYYMMDDHHMMSS>.tif`
    shape : tuple, optional
        (height, width) for the NaN planes. Inferred from the first pair that does
        resolve, so it is only needed when NO pair resolves.

    Returns
    -------
    np.ndarray (n_pairs, height, width)
        differential tides, NaN for any pair that could not be resolved
    """
    diff_tide_levels = []

    for times in acq_times:
        # sorted(), not list(): glob order is filesystem-dependent, so an unsorted
        # [0] picked an arbitrary raster whenever a date had more than one.
        tide_tifs = [sorted(tidesdir.glob('*' + time.strftime('%Y%m%d') + '*.tif'))
                     for time in times]
        if not (tide_tifs[0] and tide_tifs[1]):
            diff_tide_levels.append(None)   # filled once the shape is known
            continue

        with rio.open(tide_tifs[0][0]) as tif:
            primary_scene_tides = tif.read(1)
        with rio.open(tide_tifs[1][0]) as tif:
            secondary_scene_tides = tif.read(1)

        diff_tide_levels.append(primary_scene_tides - secondary_scene_tides)
        shape = shape or primary_scene_tides.shape

    if shape is None:
        raise FileNotFoundError(
            f'no tide raster in {tidesdir} matched any of the {len(diff_tide_levels)} '
            'acquisition pairs')

    missing = sum(level is None for level in diff_tide_levels)
    if missing:
        print(f'{missing}/{len(diff_tide_levels)} acquisition pairs had no tide raster '
              f'in {tidesdir}; those bands are NaN')

    return np.asarray([np.full(shape, np.nan) if level is None else level
                       for level in diff_tide_levels])

def get_window_coords(window: rio.windows.Window, bounds: tuple, res: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Get coordinates from rasterio Window object

    Parameters
    ----------
    window : rio.windows.Window
        window obj
    bounds : tuple
        window bounding box
    res : tuple
        raster resolution

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        longs and lats
    """    
    lats = np.arange(bounds[3], bounds[1], -1 * res[1])[:int(np.round(window.height))]
    longs = np.arange(bounds[0], bounds[2], res[0])[:int(np.round(window.width))]
    return longs, lats

def query_odata(roi, start_date, end_date):
    """
    Query Sentinel-1 SLC metadata from CDSE OData API and return a GeoDataFrame.

    Returns geometry, datetime, slice number, relative orbit number, and download URLs.
    """

    # Convert ROI to WKT (API requires WGS84)
    roi_polygon = roi.to_crs("EPSG:4326").union_all().wkt

    # OData filter string
    filter_str = (
        "Collection/Name eq 'SENTINEL-1' and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T23:59:00.000Z and "
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'SLC'"
        ") and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{roi_polygon}')"
    )

    # Build request
    url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$top=100&$filter={urllib.parse.quote(filter_str)}&$expand=Attributes"
    )

    all_items = []
    # Paging loop
    while url:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        all_items.extend(data.get("value", []))
        url = data.get("@odata.nextLink")

    if not all_items:
        return gpd.GeoDataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # ----------------------------------------------------
    # Extract start/end datetime
    # ----------------------------------------------------
    def parse_content_date(cd):
        if isinstance(cd, dict):
            return pd.Series([
                parser.isoparse(cd.get("Start")) if cd.get("Start") else None,
                parser.isoparse(cd.get("End")) if cd.get("End") else None,
            ])
        return pd.Series([None, None])

    df[["start_datetime", "end_datetime"]] = df["ContentDate"].apply(parse_content_date)
    df = df.drop(columns=["ContentDate"])

    # ----------------------------------------------------
    # Convert GeoFootprint (GeoJSON) → shapely geometry
    # ----------------------------------------------------
    if "GeoFootprint" in df.columns:
        df["geometry"] = df["GeoFootprint"].apply(lambda g: shape(g) if isinstance(g, dict) else None)
        df = df.drop(columns=["GeoFootprint"])
    else:
        df["geometry"] = None

    # ----------------------------------------------------
    # Extract nested OData Attributes (sliceNumber, relativeOrbit)
    # ----------------------------------------------------
    def get_attr(attrs, name):
        if isinstance(attrs, list):
            for item in attrs:
                if isinstance(item, dict) and item.get("Name") == name:
                    return item.get("Value")
        return None

    df["s1:frame_number"] = df["Attributes"].apply(lambda a: get_attr(a, "sliceNumber"))
    df["sat:relative_orbit"] = df["Attributes"].apply(lambda a: get_attr(a, "relativeOrbitNumber"))
    df['sat:orbit_state'] = df['Attributes'].apply(lambda a: get_attr(a, "orbitDirection"))
    df['sat:absolute_orbit'] = df['Attributes'].apply(lambda a: get_attr(a, "orbitNumber"))
    df['sar:polarizations'] = df['Attributes'].apply(lambda a: get_attr(a, "polarisationChannels"))

    df = df.drop(columns = ['Attributes'])
    # ----------------------------------------------------
    # Convert to GeoDataFrame
    # ----------------------------------------------------
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf

def get_accurate_acquisition_time(times: list[datetime], roi: gpd.GeoDataFrame) -> np.ndarray:
    """Returns accurate acqusition time (upto the hour) of S1 scenes for given ROI and time period

    Parameters
    ----------
    start_time : datetime
    stop_time : datetime
    rel_orbit : str
        relative orbit of scenes
    roi : Polygon
        ROI polygon in EPSG 4326 projection

    Returns
    -------
    np.ndarray
        array of accurate acquisition times
    """
    acquisitions_df = query_odata(roi, start_date = datetime.strftime(times[0], '%Y-%m-%d'), 
                                  end_date = datetime.strftime(times[-1], '%Y-%m-%d'))
    accurate_time = pd.to_datetime(acquisitions_df.start_datetime.values[0], format = 'mixed')
    accurate_times = np.array([datetime(year = date.year, month = date.month, day = date.day, 
                                        hour = accurate_time.hour, minute = accurate_time.minute, 
                                        second = accurate_time.second) for date in times])
    return accurate_times

def get_corrected_tide(epoch: datetime, lats: np.ndarray, longs: np.ndarray, cats: pathlib.Path,
                       ap_model: pathlib.Path, standard: str = 'datetime'):
    """Tide elevations over a grid at one epoch, with the IBE correction applied.

    Parameters
    ----------
    epoch : datetime
        epoch to evaluate at
    lats, longs : np.ndarray
        1-D grid axes in EPSG:3031 (y and x in metres, despite the names); the grid is
        formed here by meshgrid
    cats : pathlib.Path
        directory holding the tide model, as `<cats>/CATS2008_v2023/CATS2008_v2023.nc`
    ap_model : pathlib.Path
        directory of NCEP/NCAR sea level pressure files, named `slp.<year>.nc`
    standard : str
        time standard of `epoch`; 'datetime' since a datetime is what is passed

    Returns
    -------
    np.ndarray
        sea surface elevation (m), shaped (len(lats), len(longs))
    """
    lats = np.atleast_1d(lats)
    longs = np.atleast_1d(longs)
    # Latitude is flipped so the grid runs north -> south, i.e. raster row order.
    longs_ap, lats_ap = np.meshgrid(longs, np.flip(lats, axis = 0))
    pressure = sea_level_pressure(when = np.atleast_1d(np.datetime64(epoch)),
                                  x = longs_ap.reshape(-1), y = lats_ap.reshape(-1),
                                  ap_model = ap_model, crs = 3031)

    # type='grid' with the 2-D meshgrid: one epoch, every cell.
    tides = get_tidal_amplitude(lats = lats_ap, longs = longs_ap, cats = cats,
                                type = 'grid', standard = standard,
                                delta_time = np.atleast_1d(np.datetime64(epoch)))
    tides = np.asarray(tides).reshape(lats_ap.shape)
    pressure = np.asarray(pressure).reshape(tides.shape)
    return tides + ibe_elevation(pressure)

def infer_minor_amplitudes(zmajor: np.ndarray, constituents: list) -> np.ndarray:
    """Amplitudes of the 18 minor constituents, inferred from the 8 major ones.

    The standard admittance relations: each minor constituent is a fixed linear
    combination of the nearby major ones. Only minor constituents that are NOT already
    in `constituents` are inferred -- a constituent solved for by the model must not be
    added a second time, so those rows are zeroed.

    Used by `get_tidal_range_grid` to get a full tidal range rather than a
    major-constituents-only one; this is the amplitude-only form of
    `pyTMD.predict.infer_minor`.

    Parameters
    ----------
    zmajor : np.ndarray (npts, nc)
        complex amplitudes of the major constituents
    constituents : list of str
        names of the major constituents, matching `zmajor`'s columns

    Returns
    -------
    np.ndarray (npts, 18)
        minor constituent amplitudes, zeroed where the constituent is already major
    """
    # number of constituents
    npts, nc = np.shape(zmajor)
    cindex = ['q1','o1','p1','k1','n2','m2','s2','k2']
    # re-order zmajor to correspond to cindex
    z8 = np.ma.zeros((npts,8))
    ni = 0
    for i,c in enumerate(cindex):
        j = [j for j,val in enumerate(constituents) if val == c]
        if j:
            j1, = j
            z8[:,i] = zmajor[:,j1]
            ni += 1
    # list of minor constituents
    minor = ['2q1','sigma1','rho1','m1','m1','chi1','pi1','phi1','theta1',
        'j1','oo1','2n2','mu2','nu2','lambda2','l2','l2','t2']
    # only add minor constituents that are not on the list of major values
    minor_flag = [m not in constituents for m in minor]
    # estimate minor constituents
    zmin = np.zeros((npts,18))
    zmin[:,0] = 0.263*z8[:,0] - 0.0252*z8[:,1]# 2Q1
    zmin[:,1] = 0.297*z8[:,0] - 0.0264*z8[:,1]# sigma1
    zmin[:,2] = 0.164*z8[:,0] + 0.0048*z8[:,1]# rho1
    zmin[:,3] = 0.0140*z8[:,1] + 0.0101*z8[:,3]# M1
    zmin[:,4] = 0.0389*z8[:,1] + 0.0282*z8[:,3]# M1
    zmin[:,5] = 0.0064*z8[:,1] + 0.0060*z8[:,3]# chi1
    zmin[:,6] = 0.0030*z8[:,1] + 0.0171*z8[:,3]# pi1
    zmin[:,7] = -0.0015*z8[:,1] + 0.0152*z8[:,3]# phi1
    zmin[:,8] = -0.0065*z8[:,1] + 0.0155*z8[:,3]# theta1
    zmin[:,9] = -0.0389*z8[:,1] + 0.0836*z8[:,3]# J1
    zmin[:,10] = -0.0431*z8[:,1] + 0.0613*z8[:,3]# OO1
    zmin[:,11] = 0.264*z8[:,4] - 0.0253*z8[:,5]# 2N2
    zmin[:,12] = 0.298*z8[:,4] - 0.0264*z8[:,5]# mu2
    zmin[:,13] = 0.165*z8[:,4] + 0.00487*z8[:,5]# nu2
    zmin[:,14] = 0.0040*z8[:,5] + 0.0074*z8[:,6]# lambda2
    zmin[:,15] = 0.0131*z8[:,5] + 0.0326*z8[:,6]# L2
    zmin[:,16] = 0.0033*z8[:,5] + 0.0082*z8[:,6]# L2
    zmin[:,17] = 0.0585*z8[:,6]# t2
    # only add minor constituents that are not major
    return np.where(minor_flag, np.abs(zmin), 0.0)


def get_tidal_range_grid(bounds: tuple, cats: pathlib.Path, model: str = CATS_MODEL,
                         spacing: tuple = (2000, -2000), crs: str = 'EPSG:3031',
                         method: str = INTERP_METHOD,
                         extrapolate: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total tidal range on a regular grid: sum of major + inferred minor amplitudes.

    The range is a property of the tide model alone -- no epoch is involved -- so this
    is a map of how much tide there is to work with, not a prediction. That makes it
    the natural first look at a new region: where the flexure signal will be large
    enough to invert, and where it will not.

    Parameters
    ----------
    bounds : tuple
        (xmin, ymin, xmax, ymax), a geopandas-style `total_bounds`, in `crs`
    cats : pathlib.Path
        directory holding the tide model
    model : str
        entry in pyTMD's model database
    spacing : tuple
        (dx, dy) in `crs` units. dy is negative: the grid runs north -> south so the
        array is in image order and can be handed straight to `imshow`.
    crs : str
        CRS of `bounds` and `spacing`
    method : {'nearest', 'linear'}
        grid interpolation
    extrapolate : bool
        fill cells off the model's ocean domain from the nearest valid cell

    Returns
    -------
    tide_range : np.ndarray (ny, nx)
        total tidal range (m); NaN wherever the model masks the cell as land
    x, y : np.ndarray
        the grid coordinates, for the plot extent
    """
    xmin, ymin, xmax, ymax = bounds
    x = np.arange(xmin, xmax + spacing[0], spacing[0])
    y = np.arange(ymax, ymin + spacing[1], spacing[1])
    xgrid, ygrid = np.meshgrid(x, y)

    # pyTMD 3: constituents come from the model dataset via the `.tmd` accessor.
    # `coords_as` reprojects into the model's own CRS and shapes the coordinates for
    # the requested layout; `interp` then samples every constituent at those points.
    tide_model = pyTMD.io.model(directory = cats).from_database(model)
    ds = tide_model.open_dataset(group = 'z', chunks = CHUNKS, append_node = False)
    X, Y = ds.tmd.coords_as(xgrid, ygrid, type = 'grid', crs = crs)
    local = ds.tmd.interp(X, Y, method = method, extrapolate = extrapolate)

    # Each constituent is complex (amplitude and phase); the range cares only about
    # amplitude, so drop the phase with abs().
    constituents = list(local.tmd.constituents)
    amplitude = np.stack([np.abs(np.asarray(local[c])) for c in constituents], axis = -1)

    minor_amplitude = infer_minor_amplitudes(
        amplitude.reshape(-1, amplitude.shape[-1]), constituents)
    tide_range = (np.sum(amplitude, axis = -1)
                  + np.sum(minor_amplitude, axis = -1).reshape(amplitude.shape[:-1]))
    return tide_range, x, y
