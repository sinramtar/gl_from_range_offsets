"""S1 DInSAR against our GL points, two figures from one module sharing their loaders and
colours.

`--variant good`: the three interferograms of a double difference, each with the S1
delineation and our GL points from both constituent pairs, paired with the distance
distributions.

`--variant low`: the counter-example, a double difference whose two interferograms have
strongly mismatched coherence, where the fringes break down and the GL points scatter.

It reads precomputed inputs from `data/figure_inputs/` and is run on its own, outside the
Snakemake workflow.

Usage
-----
    python -m visualizations.s1_comp --data-dir ../data --variant good
    python -m visualizations.s1_comp --data-dir ../data --variant low
"""

import argparse
import pathlib

import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualizations import style

FONTSIZE = 12

# Same as `csx_comp.py`, so P1/P2 read identically across both comparison figures.
P1_COLOR = "green"
P2_COLOR = "purple"


# ------------------------------------------------------------------ good coherence

def load_good(data_dir):
    inputs = data_dir / "figure_inputs"
    with np.load(inputs / "s1_comp_phase.npz") as d:
        uuids = [str(u) for u in d["uuids"]]
        phases = {u: d[f"phase_{u}"] for u in uuids}
        extents = {u: list(d[f"extent_{u}"]) for u in uuids}
    gls = gpd.read_parquet(inputs / "s1_comp_gls.parquet")
    p1 = gpd.read_parquet(inputs / "s1_comp_p1_points.parquet")
    p2 = gpd.read_parquet(inputs / "s1_comp_p2_points.parquet")
    distances = pd.read_parquet(inputs / "s1_comp_distances.parquet")
    return uuids, phases, extents, gls, p1, p2, distances


def _pair_label(name, pts):
    return (f"{name} {pts.primary_time.dt.date.values[0]}"
            f"_{pts.secondary_time.dt.date.values[0]}")


def build_good(uuids, phases, extents, gls, p1, p2, distances):
    fig = plt.figure(figsize=(10, 10), dpi=100, layout="constrained")
    grid = plt.matplotlib.gridspec.GridSpec(ncols=2, nrows=len(uuids), figure=fig)
    axes = []

    for row, uuid in enumerate(uuids):
        extent = extents[uuid]
        ax1 = fig.add_subplot(grid[row, 0], projection=style.CRS_3031)
        ax1.set_extent(extent, crs=style.CRS_3031)
        style.add_graticule(ax1, fontsize=5)
        style.add_basemap(ax1, extent, style.CRS_3031, alpha=0.8)
        ax1.imshow(phases[uuid], cmap=cmc.romaO, alpha=0.8, extent=extent,
                   transform=style.CRS_3031, origin="upper")
        gls.loc[gls.UUID == uuid].plot(ax=ax1, color="black", linewidth=1.5)
        p1.clip(extent).plot(ax=ax1, color=P1_COLOR, markersize=8, zorder=3,
                             label=_pair_label("P1", p1))
        p2.clip(extent).plot(ax=ax1, color=P2_COLOR, markersize=8, zorder=3,
                             label=_pair_label("P2", p2))
        style.add_scalebar(ax1, max_km=30, location="lower right", major_div=1)

        ax2 = fig.add_subplot(grid[row, 1])
        for pair, color, label in [(1, P1_COLOR, "P1"), (2, P2_COLOR, "P2")]:
            sel = distances[(distances.UUID == uuid) & (distances.acq_pair == pair)]
            sns.histplot(sel, x="distance", label=label, color=color, ax=ax2, alpha=0.5)
        ax2.set_xlabel("")
        axes += [ax1, ax2]

    axes[-1].legend(fontsize=FONTSIZE)
    axes[-1].set_xlabel("Distance from S1 GL delineation (m)")
    style.panel_labels(axes, fontsize=FONTSIZE)
    return fig


# ------------------------------------------------------------------- low coherence

def load_low(data_dir):
    inputs = data_dir / "figure_inputs"
    with np.load(inputs / "s1_low_coh_phase.npz") as d:
        phase, extent = d["phase"], list(d["extent"])
    points = gpd.read_parquet(inputs / "s1_low_coh_points.parquet")
    return phase, extent, points


def build_low(phase, extent, points):
    fig = plt.figure(figsize=(5, 5), dpi=100, layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection=style.CRS_3031)
    ax.set_extent(extent, crs=style.CRS_3031)
    style.add_basemap(ax, extent, style.CRS_3031, alpha=0.8)
    img = ax.imshow(phase, cmap=cmc.romaO, alpha=0.8, extent=extent,
                    transform=style.CRS_3031, origin="upper")
    points.plot(ax=ax, color=P1_COLOR, markersize=4, label=_pair_label("", points).strip())
    style.add_scalebar(ax, max_km=30, major_div=1)
    ax.legend()
    style.add_colorbar_inset(fig, ax, img, label="radians", bbox=(0, 0.25, 0.97, 1))
    return fig


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=repo / "data")
    parser.add_argument("--variant", choices=("good", "low"), default="good")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    style.use_style(fontsize=FONTSIZE)
    if args.variant == "good":
        fig = build_good(*load_good(args.data_dir))
        default = repo / "figures" / "s1_comp_with_legend.pdf"
    else:
        fig = build_low(*load_low(args.data_dir))
        default = repo / "figures" / "s1_comp_low_coh.pdf"
    style.savefig(fig, args.out or default)


if __name__ == "__main__":
    main()
