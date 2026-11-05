#!/usr/bin/env python
"""Credible-interval width against the governing quantities.

Writes `ci_widths_regplots.{pdf,png}`: per (flow-bend, pair), the 68% and 95% CI widths
against ice thickness, ice thickness uncertainty, differential tide level and range
offsets error, shown per level as the binned median with its interquartile band
(equal-count x bins, smoothed by a rolling median over five bins). Rows are restricted to
`paper_points.parquet`; the tide panel's x-axis is the inferred `diff_tides`, with the
configured tide threshold marked.

Usage
-----
    python -m gl_from_range_offsets.visualizations.ci_widths_regplots --config config.yaml \
        --postprocessing-dir ... --figures-dir ...
"""
from __future__ import annotations

import argparse
import pathlib
import warnings

warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

from gl_from_range_offsets.postprocessing import filtering, products
from gl_from_range_offsets.visualizations.ci_spatial import C68, C95


def load_ci(path, level, paper_keys):
    """Per-pair CI rows at `level`, restricted to the paper-point keys."""
    df = pd.read_parquet(path)
    df = df.loc[df["level"].astype(int).eq(level)].rename(columns={"flowline_id": "Id"})
    df = df.dropna(subset=["diff_tides", "width"])
    df = filtering._normalise_times(df)
    keys = pd.MultiIndex.from_arrays(
        [df["Id"].astype(int), df["primary_time"], df["secondary_time"]])
    df = df[keys.isin(paper_keys)]
    print(f"  ci_{level}: {len(df)} rows in the paper point set")
    return df


PANELS = [
    ("ice_thickness", "Ice thickness (m)"),
    ("ice_thickness_uncertainty", "Ice thickness\nuncertainty (m)"),
    ("diff_tides", "Diff tide level (m)"),
    ("error", "Range offsets error (m)"),
]


def binned_quantiles(x, y, n_bins=40):
    """Median and 25-75% bounds of `y` per equal-count `x` bin (`n_bins`), lightly
    smoothed, returned as a frame with columns x, q25, med, q75."""
    m = np.isfinite(x) & np.isfinite(y)
    d = pd.DataFrame({"x": np.asarray(x)[m], "y": np.asarray(y)[m]})
    lo, hi = d["x"].quantile([0.005, 0.995])   # keep the curves off the sparse tails
    d = d[(d["x"] >= lo) & (d["x"] <= hi)]
    bins = pd.qcut(d["x"], n_bins, duplicates="drop")
    g = d.groupby(bins, observed=True)
    out = {"x": g["x"].median()}
    for name, q in (("q25", 0.25), ("med", 0.50), ("q75", 0.75)):
        out[name] = (g["y"].quantile(q)
                     .rolling(5, center=True, min_periods=1).median())
    return pd.DataFrame(out).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--postprocessing-dir", type=pathlib.Path, required=True)
    ap.add_argument("--figures-dir", type = pathlib.Path, required = True)
    args = ap.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    _, tide_min = filtering.thresholds(config)
    out = args.figures_dir / "ci_widths_regplots"

    paper_keys = products.load_paper_point_keys(
        args.postprocessing_dir / "paper_points.parquet"
    )
    print(f"paper point file: {len(paper_keys)} (flow-bend, pair) keys")
    pair_intervals = args.postprocessing_dir / "pair_intervals.parquet"
    levels = {68: load_ci(pair_intervals, 68, paper_keys),
              95: load_ci(pair_intervals, 95, paper_keys)}
    colors = {68: C68, 95: C95}

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True,
                             layout="constrained")
    for ax, (col, xlabel) in zip(axes, PANELS):
        # light grey face: the bright 95% pink needs it to read on a white page
        ax.set_facecolor("0.92")
        ax.grid(color="white", linewidth=0.8)
        ax.set_axisbelow(True)
        for level in (95, 68):  # 95 beneath, 68 on top, like the nested bands
            df = levels[level]
            b = binned_quantiles(df[col].values, df["width"].values)
            ax.fill_between(b["x"], b["q25"], b["q75"], color=colors[level],
                            alpha=0.35, linewidth=0, zorder=2)
            ax.plot(b["x"], b["med"], color=colors[level], lw=2.8, zorder=5,
                    path_effects=[pe.Stroke(linewidth=4.2, foreground="white"),
                                  pe.Normal()])
        if col == "diff_tides":
            for v in (-tide_min, tide_min):
                ax.axvline(v, color="0.2", linewidth=0.9, linestyle="--", zorder=4)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Credible interval width (m)")
    axes[0].set_ylim(0, 2000)
    # one handle per level: median line over its interquartile band
    handles = [(mpl.patches.Patch(facecolor=colors[lv], alpha=0.35, linewidth=0),
                mpl.lines.Line2D([], [], color=colors[lv], lw=2.8))
               for lv in (95, 68)]
    axes[0].legend(handles, ["95% (median, IQR)", "68% (median, IQR)"],
                   handler_map={tuple: HandlerTuple(ndivide=None)},
                   loc="upper right", framealpha=0.9, fontsize=13)

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
        print("wrote", f"{out}.{ext}")


if __name__ == "__main__":
    main()
