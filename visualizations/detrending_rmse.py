"""Residual between the detrended range-offset vertical deflection and the modelled
differential tide, under monthly vs yearly detrending of the ice-flow signal. Panels
(a, b) are the per-pixel RMSE under each choice; panel (c) is the residual per acquisition
pair through time. It reads precomputed inputs from
`data/figure_inputs/` and is run on its own, outside the Snakemake workflow.

Usage
-----
    python -m visualizations.detrending_rmse --data-dir ../data --year 2020
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from visualizations import style

FONTSIZE = 12

# cap the colour range at 5 m so a few saturating shear-margin pixels do not wash out
# the shelf-wide structure
VMAX = 5


def load(data_dir, year):
    path = data_dir / "figure_inputs" / f"detrending_rmse_{year}.npz"
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def build(cache, year):
    fig = plt.figure(figsize=(10, 7), dpi=200, layout="constrained")
    grid = plt.GridSpec(nrows=2, ncols=2, figure=fig)
    extent = list(cache["extent"])

    map_axes = []
    for col, label in enumerate(["Monthly", "Yearly"]):
        ax = fig.add_subplot(grid[0, col])
        ax.set_title(f"{label} avg. velocity spatial RMSE {year}")
        img = ax.imshow(cache[f"spatial_rmse_{label.lower()}"], vmax=VMAX, extent=extent)
        fig.colorbar(img, ax=ax, label="(m)", fraction=0.046, pad=0.046)
        ax.set_xticks([])
        ax.set_yticks([])
        map_axes.append(ax)

    ax3 = fig.add_subplot(grid[1, :])
    ax3.set_title(f"Temporal RMSE {year}")
    times = cache["times"].astype("datetime64[s]").astype("O")
    for label in ("monthly", "yearly"):
        ax3.plot(times, cache[f"temporal_rmse_{label}"],
                 label=f"{label.capitalize()} detrended", marker="o")
    ax3.set_ylabel("RMSE [m]")
    ax3.set_xlabel("Time")
    ax3.legend()

    style.panel_labels(map_axes + [ax3], fontsize=FONTSIZE)
    print(f"overall RMSE -- monthly {float(cache['overall_rmse_monthly']):.3f} m, "
          f"yearly {float(cache['overall_rmse_yearly']):.3f} m")
    return fig


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=repo / "data")
    parser.add_argument("--year", type=int, default=2020, choices=(2019, 2020, 2021))
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="default: <repo>/figures/detrending_rmse_<year>.pdf")
    args = parser.parse_args()

    style.use_style(fontsize=FONTSIZE)
    fig = build(load(args.data_dir, args.year), args.year)
    style.savefig(fig, args.out or repo / "figures" / f"detrending_rmse_{args.year}.pdf")


if __name__ == "__main__":
    main()
