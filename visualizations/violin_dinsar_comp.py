"""Rebuild the DInSAR comparison violin figure (`figures/dinsar_comp.{pdf,png}`).

Runs off `data/table1_distances_points.csv`, the per-point output of
`table1_distances.py`, so the figure and Table 2 of the manuscript are guaranteed to
come from the same points, the same filter set and the same sign convention. The
printed signed medians are therefore the table's signed medians.

The sign convention is `basemap` (implemented by
`postprocessing.table1_distances.BasemapSigner`): the side of the nearest DInSAR segment on which
the point falls is labelled by the maximum-area basemap category within a 5 km
single-sided buffer of that segment, Land being landward and Ice shelf / Ocean
seaward. Positive = seaward of the DInSAR delineation.

    python -m visualizations.violin_dinsar_comp
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gl_from_range_offsets.visualizations import style

SOURCES = ["S1 manual", "S1 automatic", "CSK"]
XLIM = 1500

def load(path, kind="paper", variant="basemap"):
    path = pathlib.Path(path)
    d = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    d = d[(d["kind"] == kind) & (d["variant"] == variant)].copy()
    d["year"] = d.group.str.split().str[0]
    d["source"] = d.group.str.split(n=1).str[1]
    return d


def plot(d, out_dir):
    style.use_style()
    palette = dict(zip(SOURCES, sns.color_palette("Set2")))

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    sns.violinplot(
        data=d, x="d", y="year", hue="source", hue_order=SOURCES,
        order=sorted(d.year.unique()), palette=palette, cut=0, density_norm="width",
        inner=None, linewidth=1.2, ax=ax,
    )
    # Outline every violin: the fills are light and, where two hue levels touch, the
    # boundary between them is otherwise only a colour change.
    for body in ax.collections:
        body.set_edgecolor("0.15")
        body.set_linewidth(1.2)

    # The median of each violin, printed on it. `dodge` in the violin call places the
    # hue levels on sub-rows; reproduce that offset so the label sits on its own violin.
    years = sorted(d.year.unique())
    width = 0.8 / len(SOURCES)
    for i, year in enumerate(years):
        for j, source in enumerate(SOURCES):
            sub = d[(d.year == year) & (d.source == source)]
            if sub.empty:
                continue
            med = sub.d.median()
            y = i - 0.4 + width * (j + 0.5)
            ax.plot([med, med], [y - width / 2, y + width / 2], color="white", lw=2)
            ax.annotate(f"{med:.0f}", (med, y - width / 2), ha="center", va="bottom",
                        fontsize=style.FONTSIZE - 3)

    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.set_xlim(-XLIM, XLIM)
    ax.set_xlabel("Signed distance to DInSAR grounding lines (m)\n"
                  "(positive = seaward, negative = landward)")
    ax.set_ylabel("Year")
    ax.legend(title="DInSAR source", loc="lower right")

    out_dir = pathlib.Path(out_dir)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"dinsar_comp.{ext}", dpi=200)
    print("wrote", out_dir / "dinsar_comp.pdf")
    for (year, source), sub in d.groupby(["year", "source"]):
        print(f"  {year} {source:<13} n={len(sub):>6}  signed median {sub.d.median():8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--kind", default="paper")
    ap.add_argument("--variant", default="basemap")
    ap.add_argument("--postprocessing-dir", type=pathlib.Path, required=True)
    ap.add_argument("--figures-dir", type = pathlib.Path, required = True)
    args = ap.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    table = args.postprocessing_dir / "table1_distances_points.parquet"
    plot(load(table, args.kind, args.variant), args.figures_dir)
