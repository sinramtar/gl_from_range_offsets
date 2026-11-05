"""Every acquisition pair's grounding-line posterior on one flow-bend line, plus their
cumulative posterior, drawn as the equally weighted mixture (1/K) * sum of the single-pair
posteriors.

Input is a directory of `<flowline_id>/<pair>.zstd` BIP results. Pairs are retained with
the configured `Posterior_params` filters, and the KDE grid spans the configured
grounding-line prior bounds.

Usage
-----
    python -m gl_from_range_offsets.visualizations.posteriors_sum --config config.yaml \
        --inversions-dir inversion_outputs/<roi>/<expt> --out gl_posteriors_sum_1563.pdf
"""

import argparse
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gl_from_range_offsets.postprocessing import filtering
from gl_from_range_offsets.postprocessing.posterior_io import load_result, pair_diagnostics
from gl_from_range_offsets.visualizations import style

FONTSIZE = 12
EXAMPLE_ID = 1563

MIXTURE_COLOR = "darkgreen"


def load_from_results(results_dir, flowline_id, config):
    """KDE of every retained pair's grounding-line posterior on one flow-bend.

    The stored `samples` are CEBU's `f_s_iid`, already resampled with `Wlast_normed`, so
    they are equally weighted posterior draws and the KDE uses no weights.
    """
    from scipy.stats import gaussian_kde

    ness_min, tide_min = filtering.thresholds(config)
    grid = np.linspace(*map(float, config["Prior"]["gl"]["bounds"]),
                       int(config["Plot_params"]["Nplot"]))
    curves, labels = [], []
    for path in sorted((pathlib.Path(results_dir) / str(flowline_id)).glob("*.zstd")):
        d = load_result(path)
        if "Null" in d:
            continue
        diagnostics = pair_diagnostics(d)
        if diagnostics["nESS"] < ness_min or abs(diagnostics["diff_tides"]) < tide_min:
            continue
        x = np.asarray(d["samples"], float)[:, 3]
        curves.append(gaussian_kde(x)(grid))
        labels.append(path.stem)
    if not curves:
        raise FileNotFoundError(f"no retained .zstd results for flow-bend {flowline_id}")
    return grid, np.array(curves), np.array(labels), float(config["Flowline_params"]["landward"])


def build(grid, curves, labels, landward, flowline_id):
    fig = plt.figure(figsize=(10, 5), dpi=100, layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("lightgray")

    # centre the axis on the reference GL, at `landward` metres along the profile
    x = grid - landward

    norm = plt.Normalize(0, len(curves))
    cmap = mpl.colormaps["inferno"]
    for index, curve in enumerate(curves):
        ax.plot(x, curve, color=cmap(norm(index)), alpha=0.7)

    ax.plot(x, curves.mean(axis=0), color=MIXTURE_COLOR, label="Cumulative posterior",
            zorder=100, linewidth=2.5)
    ax.set_xlabel("Distance along flow-bend line from reference GL (m)")
    ax.set_ylabel("KDE based PDF")
    ax.legend()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.046, fraction=0.046, label="Acquisition pair")
    step = max(1, len(labels) // 10)
    cb.set_ticks(np.arange(len(labels))[::step], labels=sorted(labels)[::step])
    print(f"flow-bend {flowline_id}: {len(curves)} pairs")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    parser.add_argument("--inversions-dir", type=pathlib.Path, required=True,
                        help="directory of `<flowline_id>/<pair>.zstd` BIP results")
    parser.add_argument("--flowline-id", type=int, default=EXAMPLE_ID)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="default: gl_posteriors_sum_<id>.pdf in the working directory")
    args = parser.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    out = args.out or pathlib.Path(f"gl_posteriors_sum_{args.flowline_id}.pdf")
    if out.suffix == "":
        out = out / f"gl_posteriors_sum_{args.flowline_id}.pdf"

    style.use_style(fontsize=FONTSIZE)
    data = load_from_results(args.inversions_dir, args.flowline_id, config)
    fig = build(*data, flowline_id=args.flowline_id)
    style.savefig(fig, out)


if __name__ == "__main__":
    main()
