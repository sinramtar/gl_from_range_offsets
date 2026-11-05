"""The highest-coherence CSK double difference against our GL points: (a) its wrapped
phase with the manual CSK delineation and our GL points from each of its two constituent
acquisition pairs (shown separately), (b) the distribution of point-to-delineation
distances for those two pairs. It reads precomputed inputs from
`data/figure_inputs/` and is run on its own, outside the Snakemake workflow.

Usage
-----
    python -m visualizations.csx_comp --data-dir ../data
"""

import argparse
import pathlib

import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from visualizations import style

FONTSIZE = 12

# Green/purple for the two constituent pairs, kept identical in `s1_comp.py` so P1 and
# P2 mean the same thing in both comparison figures.
P1_COLOR = "green"
P2_COLOR = "purple"


def load(data_dir):
    inputs = data_dir / "figure_inputs"
    with np.load(inputs / "csx_comp_phase.npz") as d:
        phase, extent = d["phase"], list(d["extent"])
    gl = gpd.read_parquet(inputs / "csx_comp_gl.parquet")
    p1 = gpd.read_parquet(inputs / "csx_comp_p1.parquet")
    p2 = gpd.read_parquet(inputs / "csx_comp_p2.parquet")
    return phase, extent, gl, p1, p2


def _pair_label(name, pair):
    return (f"{name} {pair.primary_time.dt.date.values[0]}"
            f"_{pair.secondary_time.dt.date.values[0]}")


def build(phase, extent, gl, p1, p2):
    fig = plt.figure(figsize=(12, 5), dpi=100, layout="constrained")

    ax1 = fig.add_subplot(1, 2, 1, projection=style.CRS_3031)
    ax1.set_extent(extent, crs=style.CRS_3031)
    style.add_graticule(ax1, fontsize=8)
    style.add_basemap(ax1, extent, style.CRS_3031, alpha=0.8)
    img = ax1.imshow(phase, cmap=cmc.romaO, alpha=0.8, extent=extent,
                     transform=style.CRS_3031, origin="upper")

    gl.plot(ax=ax1, color="black", linewidth=2.5, label="CSK GL")
    p1.plot(ax=ax1, color=P1_COLOR, markersize=12, zorder=3, label=_pair_label("P1", p1))
    p2.plot(ax=ax1, color=P2_COLOR, markersize=12, zorder=3, label=_pair_label("P2", p2))
    style.add_colorbar_inset(fig, ax1, img, label="radians",
                             bbox=(0, 0.2, 0.97, 1))
    style.add_scalebar(ax1, max_km=5, location="lower right", major_div=1,
                       fontsize=FONTSIZE)
    ax1.legend(fontsize=FONTSIZE)

    ax2 = fig.add_subplot(1, 2, 2)
    sns.histplot(p1, x="distance", label="P1", color=P1_COLOR, ax=ax2, alpha=0.5)
    sns.histplot(p2, x="distance", label="P2", color=P2_COLOR, ax=ax2, alpha=0.5)
    ax2.legend(fontsize=FONTSIZE)
    ax2.set_xlabel("Distance from CSK GL delineation (m)")

    style.panel_labels([ax1, ax2], fontsize=16)
    return fig


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=repo / "data")
    parser.add_argument("--out", type=pathlib.Path,
                        default=repo / "figures" / "best_csx.pdf")
    args = parser.parse_args()

    style.use_style(fontsize=FONTSIZE)
    fig = build(*load(args.data_dir))
    style.savefig(fig, args.out)


if __name__ == "__main__":
    main()
