"""Distribution of the range-offset error standard deviation for a set of correlation
thresholds (ridge plot, one KDE per threshold with its mean).

Per acquisition pair, the stable-pixel offsets are binned by the pair's own correlation
coefficient exactly as in `range_offsets.get_data_error`, so every pixel carries the
standard deviation of its bin. For a threshold t, the distribution is that of the
per-pixel standard deviations over all pairs at pixels with correlation >= t. Because the
standard deviation is constant within a (pair, bin), the distribution is represented
exactly by the bin values weighted by their pixel counts.

Usage
-----
    python -m gl_from_range_offsets.visualizations.offset_error_pdf \
        --detrended data/LarsenC/detrended_range_offsets_20{19,20,21}.tif \
        --correlations data/LarsenC/all_correlation_coeffs.tif \
        --velocity-mag data/LarsenC/antarctica_iv_200m_s1_all_v20210908_mag.tif \
        --table offset_error_bins.parquet --out offset_error_pdf.pdf
"""

import argparse
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.stats import gaussian_kde

from gl_from_range_offsets.data_preparation import range_offsets as ro
from gl_from_range_offsets.visualizations import style

THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.1), 1)
STABLE_SPEED = 0.01  # m/day
N_BINS = 10
GRID = np.linspace(0.0, 1.0, 500)
FONTSIZE = 11


def bin_table(detrended, correlations_tif, velocity_mag, thresholds=THRESHOLDS):
    """Rows (pair, bin, sigma, threshold, n): the error standard deviation of each
    correlation bin of each pair and the number of valid pixels in that bin whose
    correlation is at least `threshold`."""
    with rio.open(correlations_tif) as tif:
        bounds = tif.bounds
        correlation_band = {}
        for index, description in enumerate(tif.descriptions, start=1):
            correlation_band.setdefault(description, index)
    with rio.open(velocity_mag) as tif:
        speed = tif.read(1, window=rio.windows.from_bounds(*bounds, transform=tif.transform))

    rows = []
    with rio.open(correlations_tif) as corr:
        for path in detrended:
            with rio.open(path) as tif:
                window = rio.windows.from_bounds(*bounds, transform=tif.transform)
                for index, pair in enumerate(tif.descriptions, start=1):
                    if pair not in correlation_band:
                        continue
                    range_shift = tif.read(index, window=window)
                    correlation = corr.read(correlation_band[pair])
                    valid = ~np.isnan(range_shift) & np.isfinite(correlation)
                    stable = np.where(speed < STABLE_SPEED, range_shift, np.nan)
                    _, sigmas = ro.fill_error_std(errors=stable, correlation=correlation,
                                                  valids_mask=valid, n_bins=N_BINS)
                    bins = ro.correlation_bins(correlation, N_BINS)
                    for threshold in thresholds:
                        counts = np.bincount(bins[valid & (correlation >= threshold)],
                                             minlength=N_BINS)
                        for b in np.flatnonzero((counts > 0) & np.isfinite(sigmas)):
                            rows.append((pair, int(b), float(sigmas[b]),
                                         float(threshold), int(counts[b])))
                    print(f"  {pair}", flush=True)
    return pd.DataFrame(rows, columns=["pair", "bin", "sigma", "threshold", "n"])


def summary(table):
    """Pixel count and weighted mean standard deviation per threshold."""
    g = table.groupby("threshold")
    return pd.DataFrame({
        "pixels": g["n"].sum(),
        "pairs": g["pair"].nunique(),
        "mean_sigma": g.apply(lambda d: np.average(d["sigma"], weights=d["n"]),
                              include_groups=False),
        "min_sigma": g["sigma"].min(),
    })


def plot(table, out, reference_means=None):
    """Ridge plot, lowest threshold at the top. `reference_means` (threshold -> mean)
    draws a second, dashed mean marker per ridge for comparison."""
    style.use_style(fontsize=FONTSIZE)
    thresholds = sorted(table["threshold"].unique())
    stats = summary(table)
    cmap = mpl.colormaps["copper_r"]
    colors = {t: cmap(0.15 + 0.75 * i / max(1, len(thresholds) - 1))
              for i, t in enumerate(thresholds)}
    step, height = 1.0, 1.6

    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    for row, t in enumerate(reversed(thresholds)):
        d = table[table["threshold"] == t]
        base = row * step
        density = gaussian_kde(d["sigma"], weights=d["n"])(GRID)
        density = height * density / density.max()
        ax.fill_between(GRID, base, base + density, color=colors[t], alpha=0.85,
                        linewidth=0, zorder=2 * (len(thresholds) - row))
        ax.plot(GRID, base + density, color="white", linewidth=1.0,
                zorder=2 * (len(thresholds) - row) + 0.5)
        mean = stats.loc[t, "mean_sigma"]
        top = base + np.interp(mean, GRID, density)
        ax.plot([mean, mean], [base, top], color="0.1", linewidth=1.2,
                zorder=2 * (len(thresholds) - row) + 1)
        if reference_means is not None and t in reference_means:
            ref = reference_means[t]
            ax.plot([ref, ref], [base, base + np.interp(ref, GRID, density)], color="0.1",
                    linewidth=1.0, linestyle=(0, (2, 2)),
                    zorder=2 * (len(thresholds) - row) + 1)
        label = rf"$\rho \geq {t:.1f}$:  $E[\sigma] = {mean:.3f}$"
        if reference_means is not None and t in reference_means:
            label += rf" (was ${reference_means[t]:.3f}$)"
        ax.text(0.995, base + 0.12, label, ha="right", va="bottom", fontsize=FONTSIZE - 1,
                color="0.1", zorder=100)

    ax.set_xlim(GRID[0], GRID[-1])
    ax.set_ylim(-0.1, (len(thresholds) - 1) * step + height + 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Offset error standard deviation (m)")
    ax.set_ylabel("KDE based PDF")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    style.savefig(fig, out)
    plt.close(fig)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detrended", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--correlations", type=pathlib.Path, required=True)
    ap.add_argument("--velocity-mag", type=pathlib.Path, required=True)
    ap.add_argument("--table", type=pathlib.Path, required=True,
                    help="Parquet cache of the per-bin table; reused when it exists")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--reference-means", type=float, nargs="*", default=None,
                    help="means to compare against, one per threshold from 0.0 to 0.9")
    args = ap.parse_args()

    if args.table.exists():
        table = pd.read_parquet(args.table)
    else:
        table = bin_table(args.detrended, args.correlations, args.velocity_mag)
        args.table.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.table, index=False)
    reference = (dict(zip(THRESHOLDS, args.reference_means))
                 if args.reference_means else None)
    stats = plot(table, args.out, reference)
    print(stats.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
