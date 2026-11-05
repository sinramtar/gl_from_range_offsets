"""Range offsets against modelled tides. Left column: the per-pixel correlation between the
detrended range offsets and the modelled differential tide, weighted by significance
(correlation x (1 - p)), one map per year. Right column: the two time series at the reference
pixel marked in (b), all three years on a common month axis. It reads precomputed
inputs from `data/figure_inputs/` and is run on its own, outside the Snakemake workflow.

Usage
-----
    python -m visualizations.correlation_maps --data-dir ../data
"""

import argparse
import pathlib

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from visualizations import style

YEARS = (2019, 2020, 2021)
FONTSIZE = 12
GRATICULE_FONTSIZE = 5

# All three years are drawn on one dummy year so the seasonal cycles overlie.
COMMON_YEAR = 2021


def load(data_dir):
    inputs = data_dir / "figure_inputs"
    with np.load(inputs / "correlation_maps.npz") as data:
        maps = {int(y): data[str(y)] for y in YEARS}
        extent = list(data["extent"])
    series = pd.read_parquet(inputs / "correlation_series.parquet")
    ref_point = gpd.read_parquet(inputs / "correlation_ref_point.parquet")
    ref_gl = gpd.read_file(data_dir / "tmoc_gl_2020.geojson").to_crs("EPSG:3031")
    return maps, extent, series, ref_point, ref_gl


def build(maps, extent, series, ref_point, ref_gl, roi):
    fig = plt.figure(figsize=(12, 9), dpi=300)
    # This figure places its colorbar by hand from `ax.get_position()`, which is only
    # meaningful once the axes are at their final positions. Constrained layout (on by
    # default via style.use_style) defers positioning to draw time and then moves
    # everything again, which dropped the colorbar on top of panel (c). Fixed
    # GridSpec spacing instead.
    fig.set_layout_engine("none")
    grid = plt.GridSpec(nrows=3, ncols=2, figure=fig, width_ratios=[1, 1.5],
                        height_ratios=[1, 1, 1], wspace=0.01, hspace=0.25)

    map_axes = []
    for row, year in enumerate(YEARS):
        ax = fig.add_subplot(grid[row, 0], projection=style.CRS_3031)
        ax.set_facecolor("lightgray")
        ax.set_title(year, fontsize=FONTSIZE)
        style.add_graticule(ax, fontsize=GRATICULE_FONTSIZE)
        img = ax.imshow(maps[year], cmap="Blues", vmin=0, vmax=1, extent=extent,
                        transform=style.CRS_3031, origin="upper")
        ax.margins(0)
        style.add_scalebar(ax, max_km=100, fontsize=5)
        map_axes.append(ax)

    style.add_antarctica_inset(
        map_axes[0],
        roi_bbox=gpd.GeoSeries([box(*roi.total_bounds)], crs="EPSG:3031"))

    # The reference GL and the sampled pixel are marked once, on the middle map, so
    # they do not triple-print; the caption points at (b) for both.
    ref_gl.clip(roi.total_bounds).plot(ax=map_axes[1], color="red", linewidth=1.2)
    ref_point.plot(ax=map_axes[1], color="yellow", markersize=8, marker="*")

    series_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]
    xlim = (pd.Timestamp(f"{COMMON_YEAR}-01-01"), pd.Timestamp(f"{COMMON_YEAR}-12-31"))

    for ax, year in zip(series_axes, YEARS):
        group = series[series.year == year]
        aligned = group["time"].apply(lambda t: t.replace(year=COMMON_YEAR))
        ax.plot(aligned, group["diff_tides"], marker="o", linestyle="-")
        ax.plot(aligned, group["range_offsets"], marker="o", linestyle="-")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.3)
        if year != YEARS[-1]:
            ax.tick_params(labelbottom=False)

    series_axes[1].set_ylabel("Variable [m]")
    series_axes[-1].set_xlabel("Month")
    series_axes[-1].legend(labels=["Diff. tides", "Range offsets"])
    series_axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    series_axes[-1].xaxis.set_major_formatter(style.month_formatter())

    style.panel_labels(map_axes + series_axes, fontsize=FONTSIZE)

    # The colorbar spans the map column only, below it, so it cannot be mistaken for
    # an axis of the time series panels.
    pos = map_axes[-1].get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Correlation", fontsize=10)
    return fig


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=repo / "data")
    parser.add_argument("--roi", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / "rois" / "LarsenC.geojson")
    parser.add_argument("--out", type=pathlib.Path,
                        default=repo / "figures" / "correlation_maps.pdf")
    args = parser.parse_args()

    style.use_style(fontsize=FONTSIZE)
    roi = gpd.read_file(args.roi).to_crs("EPSG:3031")
    fig = build(*load(args.data_dir), roi=roi)
    style.savefig(fig, args.out)


if __name__ == "__main__":
    main()
