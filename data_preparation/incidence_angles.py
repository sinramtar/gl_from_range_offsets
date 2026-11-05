"""Incidence-angle mosaic on the range-offsets grid.

Each Sentinel-1 IW SLC is orbit-corrected, debursted and terrain-corrected with ESA SNAP
(`esa_snappy`), which writes the incidence angle from the ellipsoid on a geographic grid.
That band is reprojected bilinearly onto the range-offsets grid in EPSG:3031. The
per-scene rasters are then combined by averaging overlapping valid values, and the
seam slivers at scene edges (the top 1 % of the angle gradient magnitude) are refilled
from their surroundings.

SNAP and its Python bindings are required only for `scene_incidence_angle`.
"""

import numpy as np
import rasterio as rio
from rasterio.fill import fillnodata
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

DEM_NAME = 'Copernicus 30m Global DEM'
TERRAIN_SPACING_M = 30.0
READ_BLOCK = 2048
SLIVER_PERCENTILE = 99
FILL_SEARCH_DISTANCE = 20


def _terrain_corrected(slc_path):
    from esa_snappy import GPF, HashMap, ProductIO

    product = ProductIO.readProduct(str(slc_path))
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    params = HashMap()
    params.put('ApplyOrbitFile', True)
    product = GPF.createProduct('Apply-Orbit-File', params, product)
    product = GPF.createProduct('TOPSAR-Deburst', HashMap(), product)
    params = HashMap()
    params.put('demName', DEM_NAME)
    params.put('pixelSpacingInMeter', TERRAIN_SPACING_M)
    params.put('mapProjection', 'WGS84(DD)')
    params.put('saveIncidenceAngleFromEllipsoid', True)
    return GPF.createProduct('Terrain-Correction', params, product)


def _read_incidence_band(product):
    """Incidence angle (degrees) and its geographic transform from a terrain-corrected
    SNAP product."""
    import jpy

    name = next((band for band in product.getBandNames()
                 if 'incidenceanglefromellipsoid' in band.lower()), None)
    if name is None:
        raise RuntimeError('terrain-corrected product has no incidenceAngleFromEllipsoid band')
    band = product.getBand(name)
    width, height = band.getRasterWidth(), band.getRasterHeight()
    angles = np.zeros((height, width), np.float32)
    for y0 in range(0, height, READ_BLOCK):
        rows = min(READ_BLOCK, height - y0)
        for x0 in range(0, width, READ_BLOCK):
            cols = min(READ_BLOCK, width - x0)
            block = np.zeros((rows, cols), np.float32)
            band.readPixels(x0, y0, cols, rows, block)
            angles[y0:y0 + rows, x0:x0 + cols] = block

    pixel = jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
    geocoding = product.getSceneGeoCoding()
    upper_left = geocoding.getGeoPos(pixel(0, 0), None)
    upper_right = geocoding.getGeoPos(pixel(width, 0), None)
    lower_left = geocoding.getGeoPos(pixel(0, height), None)
    transform = from_origin(upper_left.lon, upper_left.lat,
                            (upper_right.lon - upper_left.lon) / width,
                            (upper_left.lat - lower_left.lat) / height)
    angles[angles == 0] = np.nan
    return angles, transform


def scene_incidence_angle(slc_path, grid, output):
    """Write the incidence angle of one SLC (`.SAFE.zip`) on `grid`, a dict with the
    `transform`, `width` and `height` of the range-offsets grid in EPSG:3031."""
    angles, transform = _read_incidence_band(_terrain_corrected(slc_path))
    destination = np.full((grid['height'], grid['width']), np.nan, dtype=np.float32)
    reproject(source=angles, destination=destination, src_transform=transform,
              src_crs='EPSG:4326', src_nodata=np.nan, dst_crs='EPSG:3031',
              dst_transform=grid['transform'], dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    profile = {'driver': 'GTiff', 'height': grid['height'], 'width': grid['width'],
               'count': 1, 'dtype': 'float32', 'crs': 'EPSG:3031',
               'transform': grid['transform'], 'nodata': np.nan}
    with rio.open(output, 'w', **profile) as dest:
        dest.write(destination, 1)
        dest.update_tags(1, long_name='incidence_angle_from_ellipsoid', units='degrees')


def _mean_merge(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """`rasterio.merge` method: pairwise mean where both rasters are valid, otherwise the
    valid one."""
    both = ~merged_mask & ~new_mask
    only_new = merged_mask & ~new_mask
    merged_data[both] = (merged_data[both] + new_data[both]) / 2.0
    merged_data[only_new] = new_data[only_new]


def merge_incidence_angles(scene_rasters, output):
    """Mosaic the per-scene incidence angles and refill the seam slivers."""
    sources = [rio.open(path) for path in scene_rasters]
    try:
        mosaic, transform = merge(sources, dtype=np.float32, nodata=np.nan,
                                  method=_mean_merge)
        meta = sources[0].meta.copy()
    finally:
        for source in sources:
            source.close()

    angles = mosaic[0]
    gx = np.pad(np.abs(np.diff(angles, axis=1)), ((0, 0), (0, 1)), mode='edge')
    gy = np.pad(np.abs(np.diff(angles, axis=0)), ((0, 1), (0, 0)), mode='edge')
    gradient = np.hypot(gx, gy)
    sliver = gradient > np.nanpercentile(gradient, SLIVER_PERCENTILE)
    keep = np.isfinite(angles) & ~sliver
    filled = fillnodata(np.where(keep, angles, 0).astype(np.float32), mask=keep,
                        max_search_distance=FILL_SEARCH_DISTANCE)
    filled[~np.isfinite(angles) & ~sliver] = np.nan

    meta.update({'height': angles.shape[0], 'width': angles.shape[1], 'count': 1,
                 'transform': transform, 'dtype': 'float32', 'nodata': np.nan})
    with rio.open(output, 'w', **meta) as dest:
        dest.write(filled, 1)
