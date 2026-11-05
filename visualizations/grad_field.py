"""How the flow-bend gradient field is built. Top row, shelf-wide: (a) the gradient of the
smoothed standard deviation of the range offsets time series (across the flexure zone),
(b) ice speed (along flow), (c) the combined field, taking the flexure gradient near the GL
and the ice-flow direction elsewhere. Bottom row: the same three over Crane, with the
reference GL points and resulting flow-bend lines on (f). It reads precomputed
inputs from `data/figure_inputs/` and is run on its own, outside the Snakemake workflow.

Usage
-----
    python -m visualizations.grad_field --data-dir ../data
"""

import argparse
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from visualizations import style

FONTSIZE = 12

# Ice speed spans four orders of magnitude between the shelf interior and the trunk;
# without a ceiling the map is one bright streak on a black field.
VELOCITY_VMAX = 3

PANELS = [("offsets_grad", "Offsets stddev gradient"),
          ("velocity", "Ice speed"),
          ("combined", "Normalised combined gradient")]


def load(data_dir, crane_roi_path):
    with np.load(data_dir / "figure_inputs" / "grad_field.npz") as d:
        cache = {k: d[k] for k in d.files}
    crane_roi = gpd.read_file(crane_roi_path).to_crs("EPSG:3031")
    flowbends = gpd.read_file(data_dir / "hybrid_flowlines.gpkg").to_crs("EPSG:3031")
    ref_gl = gpd.read_file(data_dir / "tmoc_gl_2020.geojson").to_crs("EPSG:3031")
    return cache, crane_roi, flowbends.clip(crane_roi.total_bounds), ref_gl


def build(cache, crane_roi, crane_flowbends, ref_gl):
    fig = plt.figure(figsize=(20, 15), dpi=100, layout="constrained")
    extent = list(cache["extent"])
    crane_extent = list(cache["crane_extent"])

    # The gradient field is only defined where the offsets are; its clipped minimum is
    # the nodata floor, so masking at it removes the background rather than a real low.
    valid = cache["offsets_grad"] > cache["offsets_grad"].min()

    axes = []
    for i, (key, title) in enumerate(PANELS):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_title(title)
        vmax = VELOCITY_VMAX if key == "velocity" else None
        img = ax.imshow(np.where(valid, cache[key], np.nan), cmap="plasma",
                        extent=extent, vmax=vmax)
        fig.colorbar(img, ax=ax, pad=0.046, fraction=0.046, label="m/d")
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)

    crane_roi.minimum_rotated_rectangle().boundary.plot(ax=axes[1], color="red",
                                                        linewidth=2.0)

    for i, (key, _) in enumerate(PANELS):
        ax = fig.add_subplot(2, 3, i + 4)
        vmax = VELOCITY_VMAX if key == "velocity" else np.nanmax(cache[key])
        ax.imshow(cache[f"crane_{key}"], cmap="plasma", extent=crane_extent,
                  vmin=np.nanmin(cache[key]), vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)

    ref_gl.clip(crane_roi.total_bounds).plot(ax=axes[-1], color="red",
                                             label="Ref. GL")
    crane_flowbends.plot(ax=axes[-1], color="white", label="Flow-bend lines", alpha=0.5)
    axes[-1].legend(facecolor="black", labelcolor="white")

    style.panel_labels(axes, fontsize=FONTSIZE)
    return fig


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    rois = pathlib.Path(__file__).resolve().parents[1] / "rois"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=repo / "data")
    parser.add_argument("--crane-roi", type=pathlib.Path, default=rois / "Crane.geojson")
    parser.add_argument("--out", type=pathlib.Path,
                        default=repo / "figures" / "gradients.pdf")
    args = parser.parse_args()

    style.use_style(fontsize=FONTSIZE)
    fig = build(*load(args.data_dir, args.crane_roi))
    style.savefig(fig, args.out)


if __name__ == "__main__":
    main()
