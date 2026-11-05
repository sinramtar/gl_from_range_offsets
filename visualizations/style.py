"""Shared figure style: constrained layout, a single font size, and the RADARSAT basemap.

Import `use_style()` before building a figure and create figures with
`layout="constrained"`, which accounts for colorbars and cartopy's geo-axes. The module
also carries the map furniture the figure scripts share: the two CRSs, the scale bar and
graticule presets, the Antarctica locator inset, and the panel labeller.
"""

import functools
import math
import pathlib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib_map_utils.core.scale_bar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.windows import from_bounds

# font size for titles, axis labels, legends and colorbar labels
FONTSIZE = 16

# smaller size for the graticule labels, which are on all four edges and collide at FONTSIZE
GRATICULE_FONTSIZE = 9

# the projection every map is drawn in, and the geographic CRS the graticule is labelled in
CRS_3031 = ccrs.SouthPolarStereo(true_scale_latitude=-71)
CRS_4326 = ccrs.PlateCarree()

# Quantarctica/ADD basemap categories, used only where the vector basemap replaces the mosaic
BASEMAP_COLORS = {"Ocean": "#a3bdd1", "Ice shelf": "#cfe1eb", "Land": "#f3f1f0"}

# navy / vermillion / dark green, shared by every figure that splits by acquisition year
YEAR_COLORS = {2019: "#332288", 2020: "#D55E00", 2021: "#117733"}

# hard-coded so the month axes cannot follow the machine's locale; see month_formatter
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# EPSG:3031 100 m mosaic, 52224 x 60416 px; always read as a decimated window
RADARSAT = pathlib.Path(__file__).resolve().parents[2] / "data" / "RADARSAT_Mosaic.jp2"


def use_style(fontsize=FONTSIZE):
    """One font size everywhere, and constrained layout by default."""
    plt.rcParams.update({
        "figure.constrained_layout.use": True,
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "legend.title_fontsize": fontsize,
        "figure.titlesize": fontsize,
    })


@functools.lru_cache(maxsize=8)
def read_basemap(extent, path=RADARSAT, max_px=2000):
    """Decimated read of the mosaic over `extent` = (xmin, xmax, ymin, ymax) in EPSG:3031,
    downsampled to at most `max_px` on the long side. Returns (array, extent actually read):
    the window is snapped to the raster, so the returned extent may be smaller near the edge
    and is what `imshow` must be given. Cached on its arguments, hence the tuple `extent`."""
    xmin, xmax, ymin, ymax = extent
    with rasterio.open(path) as src:
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        # ceil, not floor: floor leaves a 2900 px window undecimated at max_px=2000
        scale = max(1, math.ceil(max(window.width, window.height) / max_px))
        arr = src.read(
            1,
            window=window,
            out_shape=(max(1, int(window.height // scale)), max(1, int(window.width // scale))),
            resampling=rasterio.enums.Resampling.average,
        )
        left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
    return arr, (left, right, bottom, top)


def add_basemap(ax, extent, crs, path=RADARSAT, cmap="gray", alpha=1.0, max_px=2000,
                interpolation="bilinear"):
    """Draw the RADARSAT mosaic behind everything else on map axis `ax`. `extent` is the
    map extent in EPSG:3031 and `crs` the axis's cartopy CRS. `interpolation="none"` embeds
    the array in the PDF at its own resolution (pair it with a `max_px` matched to the panel
    size); any other value makes the vector backend resample it, driving up the file size."""
    try:
        arr, img_extent = read_basemap(tuple(extent), path=path, max_px=max_px)
    except (FileNotFoundError, rasterio.errors.RasterioIOError):
        return None
    if arr.size == 0:
        return None

    valid = arr[arr > 0]
    vmin, vmax = (np.percentile(valid, [2, 98]) if valid.size else (0, 255))
    return ax.imshow(arr, extent=img_extent, origin="upper", transform=crs,
                     cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                     interpolation=interpolation, zorder=0)


# ------------------------------------------------------------------ map furniture

def add_graticule(ax, fontsize=GRATICULE_FONTSIZE, color="dimgray"):
    """Dashed lat/lon graticule labelled on the bottom and left only."""
    return ax.gridlines(
        draw_labels={"bottom": "x", "left": "y"}, linestyle="--", linewidth=0.5,
        color=color, crs=CRS_4326, dms=True, x_inline=False, y_inline=False,
        xlabel_style={"size": fontsize, "color": color},
        ylabel_style={"size": fontsize, "color": color},
        xpadding=3, ypadding=3, rotate_labels=False,
    )


def add_scalebar(ax, max_km=50, location="lower left", major_div=2, fontsize=8,
                 color="black"):
    """Scale bar preset. `max_km` is the bar's full length, chosen against the panel's real
    width since the bar reports true projected distance. The axis must be in EPSG:3031 map
    coordinates, not pixel indices, for the printed distance to be correct."""
    bar = {"projection": 3031, "minor_type": "none", "height": 0.05,
           "major_div": major_div, "max": max_km}
    if color == "white":
        bar["facecolors"] = "white"
    scalebar = ScaleBar(
        location=location, style="boxes", bar=bar,
        labels={"style": "minor_first", "loc": "above", "pad": 0},
        units={"loc": "text", "pad": 0},
        text={"stroke_color": color, "textcolor": color, "fontsize": fontsize,
              "stroke_width": 0.3, "fontweight": "light", "fontfamily": "sans-serif"},
    )
    ax.add_artist(scalebar)
    return scalebar


def add_colorbar_inset(fig, ax, sm, label, loc="lower right", bbox=(0, 0.2, 0.97, 1),
                       width="20%", height="3%", orientation="horizontal",
                       ticks=None, ticklabels=None, fontsize=10):
    """A small colorbar floated inside a map panel rather than stealing its width. Defaults
    to the interferogram phase ramp (ticks at -pi and pi); pass `ticks`/`ticklabels` for
    anything else."""
    cax = inset_axes(ax, width=width, height=height, loc=loc,
                     bbox_transform=ax.transAxes, bbox_to_anchor=bbox)
    cbar = fig.colorbar(sm, cax=cax, pad=0.046, fraction=0.046, orientation=orientation)
    cbar.set_label(label=label, size=fontsize, labelpad=-0.1)
    if ticks is None:
        ticks, ticklabels = np.linspace(-np.pi, np.pi, 2), [r"$-\pi$", r"$\pi$"]
    cbar.set_ticks(ticks, labels=ticklabels, size=fontsize)
    cbar.ax.tick_params(length=0, pad=0.1)
    cbar.outline.set_linewidth(0.5)
    return cbar


def add_antarctica_inset(ax, roi_bbox, loc="upper right", width="30%", height="30%",
                         bbox=(0, 0, 1, 1), bbox_color="red",
                         set_extent=(-170, 170, -85, -65)):
    """Continent locator inset with the region of interest marked."""
    inset = inset_axes(ax, width=width, height=height, loc=loc, axes_class=GeoAxes,
                       axes_kwargs=dict(projection=CRS_3031),
                       bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    inset.set_extent(list(set_extent), crs=CRS_4326)
    inset.add_feature(cfeature.LAND)
    inset.add_feature(cfeature.OCEAN)
    for spine in inset.spines.values():
        spine.set_linewidth(0.1)
    roi_bbox.plot(ax=inset, color=bbox_color)
    return inset


def month_formatter():
    """Month-abbreviation tick formatter that is always English, regardless of the machine's
    LC_TIME. Use this rather than `DateFormatter("%b")`, which follows the locale that GDAL
    adopts on import."""
    return mticker.FuncFormatter(
        lambda x, _pos: MONTH_ABBR[mdates.num2date(x).month - 1])


def panel_label(ax, label, fontsize=None, boxed=False):
    """The bold "(a)" in a panel's top-left corner, in axes coordinates."""
    kwargs = {}
    if boxed:
        kwargs["bbox"] = dict(boxstyle="square", fc="white", alpha=0.5)
    return ax.text(0.02, 0.97, label, transform=ax.transAxes,
                   fontsize=fontsize or FONTSIZE, fontweight="bold",
                   va="top", ha="left", **kwargs)


def panel_labels(axes, fontsize=None, boxed=False, start=0):
    """Label `axes` in order as (a), (b), (c), ..."""
    return [panel_label(ax, f"({chr(97 + start + i)})", fontsize=fontsize, boxed=boxed)
            for i, ax in enumerate(axes)]


def savefig(fig, savepath, dpi=300):
    """Write both a .pdf (what LaTeX includes) and a .png (for quick viewing)."""
    savepath = pathlib.Path(savepath)
    for suffix in (".pdf", ".png"):
        p = savepath.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print("wrote", p)
