"""Grounding-line credible-interval calibration against Sentinel-1 DInSAR delineations.

A calibration unit is a (DInSAR delineation d, flow-bend line f) pair. Its comparison
distribution is the equal-weight mixture of Gaussian approximations to the two constituent
pair posteriors on f, and its PIT value is that mixture's CDF at the DInSAR crossing,
measured along f relative to its reference grounding-line crossing. Coverage confidence
intervals come from a block bootstrap over whole delineations. Distances are in metres
(EPSG:3031).

Run with ``snakemake --snakefile postprocessing/Snakefile calibration``.
"""

import argparse
import pathlib
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import filtering
from .run_metadata import write_sidecar


def _read_ci(path, columns=None):
    """Read a Parquet interval product and normalize its flow-bend key."""
    parquet_columns = None if columns is None else [
        "flowline_id" if column == "Id" else column for column in columns
    ]
    ci = pd.read_parquet(path, columns=parquet_columns)
    ci = ci.rename(columns={"flowline_id": "Id"})
    return ci


def load_pair_posteriors(path):
    """Per (flow-bend, pair) posterior as a Gaussian from the per-pair 68% file: mu the
    midpoint of the stored interval, sigma its half-width."""
    ci = _read_ci(
        path, ["Id", "primary_time", "secondary_time", "level", "left", "right"]
    )
    ci = ci.loc[ci["level"].astype(int).eq(68)].copy()
    ci = ci.dropna(subset=["left", "right"])
    ci = filtering._normalise_times(ci)
    ci["Id"] = ci["Id"].astype(int)
    key = ["Id", "primary_time", "secondary_time"]
    if ci.duplicated(key).any():
        raise ValueError("68% pair intervals contain duplicate flow-bend/acquisition-pair keys")
    ci["mu"] = (ci["left"] + ci["right"]) / 2.0
    ci["sigma"] = (ci["right"] - ci["left"]) / 2.0
    return ci[key + ["mu", "sigma"]]


def load_reference_uncertainty(path):
    """Load and validate the per-calibration-unit DInSAR uncertainty product."""
    frame = pd.read_parquet(path)
    required = {"dinsar", "Id", "reference_sigma_m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reference-uncertainty product lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Id"] = frame["Id"].astype(int)
    if frame.duplicated(["dinsar", "Id"]).any():
        raise ValueError("reference-uncertainty product has duplicate DInSAR/flow-bend keys")
    finite = frame["reference_sigma_m"].dropna().to_numpy(float)
    if ((~np.isfinite(finite)) | (finite < 0)).any():
        raise ValueError("reference_sigma_m must be null or finite and nonnegative")
    return frame


def pit_mixture(mus, sigmas, d_true, sigma_ref=0.0):
    """PIT value for an equally weighted Gaussian mixture, `mus`/`sigmas` shaped
    (n_units, n_components) with NaN marking an absent component. The mixture CDF is the
    mean of the component CDFs, each widened by `sigma_ref` in quadrature."""
    mus, sigmas = np.asarray(mus, float), np.asarray(sigmas, float)
    d_true = np.asarray(d_true, float)[:, None]
    sigma_ref = np.asarray(sigma_ref, dtype=float)
    if sigma_ref.ndim == 0:
        sigma_ref = np.full(len(mus), float(sigma_ref))
    if sigma_ref.shape != (len(mus),):
        raise ValueError("sigma_ref must be scalar or one value per calibration unit")
    scale = np.sqrt(sigmas ** 2 + sigma_ref[:, None] ** 2)
    valid = np.isfinite(mus) & np.isfinite(scale) & (scale > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf = norm.cdf((d_true - mus) / scale)
    cdf = np.where(valid, cdf, 0.0)
    n = valid.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cdf.sum(axis=1) / n
    return np.where(n > 0, u, np.nan)


def attach_pair_posteriors(units, pair_post):
    """Attach both exact constituent-pair posteriors to each calibration unit.

    ``pair_post`` is the already-filtered canonical paper product. Units without two
    finite, positive-width pair posteriors are excluded.
    """
    out = units.copy()
    for k in (1, 2):
        right = pair_post.rename(columns={"primary_time": f"p{k}", "secondary_time": f"s{k}",
                                          "mu": f"mu_{k}", "sigma": f"sigma_{k}"})
        out = out.merge(right[["Id", f"p{k}", f"s{k}", f"mu_{k}", f"sigma_{k}"]],
                        on=["Id", f"p{k}", f"s{k}"], how="left", validate="many_to_one")

    mus = out[["mu_1", "mu_2"]].to_numpy(float)
    sigmas = out[["sigma_1", "sigma_2"]].to_numpy(float)
    keep = (np.isfinite(mus) & np.isfinite(sigmas) & (sigmas > 0)).all(axis=1)
    out = out.loc[keep].copy()
    mus, sigmas = mus[keep], sigmas[keep]
    mu = mus.mean(axis=1)
    within = np.mean(sigmas ** 2, axis=1)
    between = np.mean((mus - mu[:, None]) ** 2, axis=1)
    out["mu"], out["sigma"] = mu, np.sqrt(within + between)
    out["width"] = 2.0 * out["sigma"]
    return out


def identifiable_mask(units, max_span):
    """Units whose truth is identifiable: one crossing, sharply localised."""
    return units.n_cross.eq(1) & units.d_span.le(max_span)


def prior_support_mask(units, gl_prior):
    """Keep truths inside the grounding-line prior's support."""
    return units.d_true.between(*gl_prior)


def run_calibration(units, pair_post, *, sigma_ref, max_span, gl_prior):
    """PIT values for identifiable units with two exact pair posteriors. Units without a
    reference uncertainty are evaluated unadjusted and keep a NaN `reference_sigma_m`."""
    if "n_cross" in units.columns:
        keep = identifiable_mask(units, max_span)
        print(f"  [paper] dropping {(~keep).sum()} unidentifiable units "
              f"({100 * (~keep).mean():.1f}%): >1 crossing or "
              f"span >{max_span:.0f} m")
        units = units.loc[keep]

    keep = prior_support_mask(units, gl_prior)
    print(f"  [paper] dropping {(~keep).sum()} units outside the GL prior support "
          f"({100 * (~keep).mean():.1f}%): truth outside "
          f"[{gl_prior[0]:.0f}, {gl_prior[1]:.0f}] m of arc")
    units = units.loc[keep]

    before_pairs = len(units)
    m = attach_pair_posteriors(units, pair_post)
    print(f"  [paper] exact two-pair cohort: {len(m)} units "
          f"({before_pairs - len(m)} without two valid pair posteriors)")
    if isinstance(sigma_ref, str):
        reference_sigma = m[sigma_ref].to_numpy(float)
        applied_sigma = np.nan_to_num(reference_sigma, nan=0.0)
    else:
        applied_sigma = sigma_ref
    u = pit_mixture(m[["mu_1", "mu_2"]].to_numpy(float),
                    m[["sigma_1", "sigma_2"]].to_numpy(float),
                    m["d_true"], sigma_ref=applied_sigma)
    # np.asarray, not .values: pandas 3 string columns yield a StringArray from .values
    result = {"u": np.asarray(u), "block": np.asarray(m["dinsar"]),
              "dinsar": np.asarray(m["dinsar"]), "flowline": np.asarray(m["Id"]),
              "sep": np.asarray(m["sep"]), "width": np.asarray(m["width"]),
              "resid": np.asarray(m["mu"] - m["d_true"]), "kind": "paper"}
    if isinstance(sigma_ref, str):
        result["reference_sigma_m"] = np.asarray(reference_sigma)
    return result


def reliability_from_pit(u, block, levels=None, n_boot=2000, seed=0):
    """Empirical central coverage at each nominal `level`, with block-bootstrap 95% CIs
    resampled over whole `block`s. Returns (levels, coverage, ci_lo, ci_hi)."""
    if levels is None:
        levels = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])
    u = np.asarray(u, float)
    block = np.asarray(block)
    rng = np.random.default_rng(seed)

    def cov(uu):
        lo, hi = (1 - levels) / 2, (1 + levels) / 2
        return ((uu[:, None] >= lo) & (uu[:, None] <= hi)).mean(0)

    point = cov(u)
    blocks = np.unique(block)
    idx = {b: np.where(block == b)[0] for b in blocks}
    boot = np.empty((n_boot, len(levels)))
    for i in range(n_boot):
        pick = np.concatenate([idx[b] for b in rng.choice(blocks, blocks.size)])
        boot[i] = cov(u[pick])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return levels, point, lo, hi


def _savefig(fig, savepath, dpi=150):
    """Write `fig` as both .pdf and .png via `visualizations.style.savefig`."""
    from ..visualizations import style

    style.savefig(fig, savepath, dpi=dpi)


def pit_diagnostics(u):
    """KS test of `u` against Uniform(0, 1) with a plain-language verdict, returned as a
    dict (n, ks_p, mean, verdict)."""
    from scipy import stats
    u = np.asarray(u, float)
    u = u[~np.isnan(u)]
    ks = stats.kstest(u, "uniform")
    var, mean = u.var(), u.mean()
    if ks.pvalue > 0.05:
        v = "consistent with calibrated (flat PIT)"
    elif var > 1 / 12:
        v = "U-shaped -> OVERCONFIDENT (intervals too narrow)"
    else:
        v = "central hump -> UNDERCONFIDENT (intervals too wide)"
    if abs(mean - 0.5) > 0.1:
        v += f"; {'landward' if mean < 0.5 else 'seaward'}-biased (mean PIT={mean:.2f})"
    return {"n": len(u), "ks_p": float(ks.pvalue), "mean": float(mean), "verdict": v}


def report(result, levels=None, mark=(0.68, 0.95)):
    """Print the headline calibration numbers with block-bootstrap CIs."""
    lv, emp, lo, hi = reliability_from_pit(result["u"], result["block"], levels=levels)
    print(f"\n[{result['kind']}]  {len(result['u'])} (DInSAR, flow-bend) units, "
          f"{len(np.unique(result['block']))} blocks, "
          f"{(result['width'] < filtering.COLLAPSED_CI_M).sum()} collapsed")
    for L in mark:
        i = int(np.argmin(np.abs(lv - L)))
        print(f"  {L:>4.0%} nominal -> {emp[i]:.3f} empirical (95% CI {lo[i]:.3f}-{hi[i]:.3f})")
    d = pit_diagnostics(result["u"])
    print(f"  median |mu - truth| = {np.median(np.abs(result['resid'])):.0f} m, "
          f"median residual = {np.median(result['resid']):+.0f} m")
    print("  " + d["verdict"])
    return lv, emp, lo, hi


def report_reference_comparison(rows, mark=(0.68, 0.95)):
    """Print one compact comparison table for named reference-error scenarios."""
    print("\nReference-error scenario comparison")
    print("  filter       scenario              sigma_ref        68% coverage        95% coverage")
    for row in rows:
        cells = []
        for level in mark:
            index = int(np.argmin(np.abs(row["levels"] - level)))
            cells.append(
                f"{row['coverage'][index]:.3f} "
                f"({row['low'][index]:.3f}-{row['high'][index]:.3f})")
        sigma = row["sigma_ref"]
        sigma_label = f"{sigma:7.1f} m" if np.isfinite(sigma) else " per-unit "
        print(f"  {row['filter']:<12} {row['scenario']:<21} {sigma_label}  "
              f"{cells[0]:>18}  {cells[1]:>18}")


def plot_reliability_and_pit(results, levels=None, mark=(0.68, 0.95),
                             savepath=None):
    """Reliability curves and one PIT histogram per scenario, panels labelled (a), (b), ..."""
    import matplotlib.pyplot as plt
    from ..visualizations import style
    style.use_style()

    n = len(results)
    fig = plt.figure(figsize=(7.0 + 2.5 * n, 5.4), layout="constrained")
    gs = fig.add_gridspec(1, 1 + n, width_ratios=[1.35] + [1] * n)
    axc = fig.add_subplot(gs[0, 0])
    axc.plot([0, 1], [0, 1], "k--", lw=1.3, label="calibrated")
    axc.text(0.0, 1.02, "(a)", transform=axc.transAxes,
             ha="left", va="bottom", fontweight="bold")

    for j, (name, res) in enumerate(results.items()):
        c = f"C{j}"
        sigma_ref = res.get("sigma_ref")
        scenario_label = name if sigma_ref is None or not np.isfinite(sigma_ref) else (
            f"{name} ($\\sigma_{{ref}}$={sigma_ref:.1f} m)"
        )
        lv, pt, lo, hi = reliability_from_pit(res["u"], res["block"], levels=levels)
        axc.fill_between(lv, lo, hi, color=c, alpha=0.20)
        axc.plot(
            lv,
            pt,
            "o-",
            color=c,
            label=f"{scenario_label} (n={len(res['u'])})",
        )

        axp = fig.add_subplot(gs[0, 1 + j])
        axp.hist(np.asarray(res["u"], float), bins=20, range=(0, 1), density=True,
                 color=c, edgecolor="white")
        axp.axhline(1.0, color="k", ls="--")
        axp.set_xlabel("PIT  u = F(truth)")
        axp.set_yticks([])
        axp.text(0.0, 1.02, f"({chr(ord('b') + j)})", transform=axp.transAxes,
                 ha="left", va="bottom", fontweight="bold")

    for L in mark:
        axc.axvline(L, color="0.85", lw=0.8, zorder=0)
    axc.set_xlabel("nominal credible level")
    axc.set_ylabel("empirical coverage")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.set_aspect("equal")
    axc.legend(loc="upper left")
    axc.grid(alpha=0.2)
    if savepath:
        _savefig(fig, savepath)
    return fig


# One colour per group and a marker per mission, so the groups survive greyscale and CVD.
GROUP_STYLE = {
    "2019 S1 manual":    ("#2a78d6", "o"),
    "2019 S1 automatic": ("#1baf7a", "s"),
    "2020 S1 automatic": ("#eda100", "s"),
    "2021 S1 automatic": ("#4a3aa7", "s"),
}
_MISSION_ORDER = {"S1 manual": 0, "S1 automatic": 1}


def dinsar_group(dinsar_id):
    """(year, mission) label of a DInSAR scene from its stored id. Sentinel-1 ids start
    with `SEN` (automatic delineations carry `_HH_`/`_VV_`, manual ones do not); other ids
    are CSK. The year is the first acquisition date's."""
    s = str(dinsar_id)
    year = re.search(r"20\d{6}", s).group(0)[:4]
    if not s.startswith("SEN"):
        return f"{year} CSK"
    auto = "_HH_" in s or "_VV_" in s
    return f"{year} S1 {'automatic' if auto else 'manual'}"


def split_by_group(result):
    """{group label: sub-result} in Table-1 order (year, then mission)."""
    labels = np.array([dinsar_group(d) for d in result["dinsar"]])
    def key(lab):
        year, mission = lab.split(" ", 1)
        return (int(year), _MISSION_ORDER.get(mission, 9))
    out = {}
    for lab in sorted(np.unique(labels), key=key):
        m = labels == lab
        out[lab] = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in result.items()}
        out[lab]["kind"] = f"{result['kind']} | {lab}"
    return out


def plot_reliability_by_group(result, levels=None, mark=(0.68, 0.95),
                              savepath=None):
    """All year x mission reliability curves for `result` in one axes, one colour per
    group with block-bootstrap 95% error bars, and a PIT histogram per group alongside."""
    import matplotlib.pyplot as plt
    from ..visualizations import style
    style.use_style()

    groups = split_by_group(result)
    n = len(groups)
    ncol_h = max(1, (n + 1) // 2)
    fig = plt.figure(figsize=(7.0 + 2.5 * ncol_h, 7.0), layout="constrained")
    gs = fig.add_gridspec(2, 1 + ncol_h, width_ratios=[2.6] + [1] * ncol_h)
    axc = fig.add_subplot(gs[:, 0])
    axc.plot([0, 1], [0, 1], "--", color="#898781", lw=1.3, label="calibrated", zorder=1)

    for j, (name, res) in enumerate(groups.items()):
        c, mk = GROUP_STYLE.get(name, ("#9e9e9e", "D"))
        lv, pt, lo, hi = reliability_from_pit(res["u"], res["block"], levels=levels)
        axc.errorbar(lv, pt, yerr=[pt - lo, hi - pt], color=c, marker=mk, ms=7,
                     lw=2, elinewidth=1.1, capsize=3, alpha=0.95, zorder=3,
                     label=f"{name} (n={len(res['u'])})")

        axp = fig.add_subplot(gs[j // ncol_h, 1 + j % ncol_h])
        axp.hist(np.asarray(res["u"], float), bins=15, range=(0, 1), density=True,
                 color=c, edgecolor="white")
        axp.axhline(1.0, color="#898781", ls="--", lw=1)
        axp.set_yticks([])
        axp.set_xticks([0, 0.5, 1])
        axp.set_title(name, fontsize=14)
        if j // ncol_h == 1:
            axp.set_xlabel("PIT  u = F(truth)", fontsize=14)

    for L in mark:
        axc.axvline(L, color="#e1e0d9", lw=0.8, zorder=0)
    axc.set_xlabel("nominal credible level")
    axc.set_ylabel("empirical coverage")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.set_aspect("equal")
    axc.legend(loc="upper left", fontsize=14)
    axc.grid(alpha=0.2)

    if savepath:
        _savefig(fig, savepath)
    return fig


def report_by_stratum(result, stratum_of, levels=None, mark=(0.68, 0.95)):
    """stratum_of: callable(dinsar_id, flowline_id) -> label. Prints coverage per
    stratum so a region-dependent failure does not hide in the pool."""
    labels = np.array([stratum_of(d, f) for d, f in zip(result["dinsar"], result["flowline"])])
    for lab in np.unique(labels):
        m = labels == lab
        lv, emp, _, _ = reliability_from_pit(result["u"][m], result["block"][m], levels=levels)
        line = "; ".join(f"{L:.0%}->{emp[int(np.argmin(np.abs(lv - L)))]:.2f}" for L in mark)
        print(f"  [{lab}] n={m.sum():6d}  {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--postprocessing-dir", type = pathlib.Path, required = True)
    ap.add_argument("--calibration-dir", type = pathlib.Path, required = True)
    ap.add_argument("--figures-dir", type = pathlib.Path, required = True)
    ap.add_argument("--truth-cache", type=pathlib.Path, required=True)
    ap.add_argument("--reference-uncertainty", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    cal = config["Calibration"]
    gl_prior = tuple(float(value) for value in config["Prior"]["gl"]["bounds"])
    threshold = float(cal["coverage_threshold"])
    max_span = float(cal["identifiable_crossing_tolerance_m"])

    pit_out = args.calibration_dir / "pit.parquet"
    coverage_out = args.calibration_dir / "coverage.parquet"
    reliability_out = args.figures_dir / "reliability.pdf"

    units = pd.read_parquet(args.truth_cache)
    units = units.loc[units["dinsar"].astype(str).str.startswith("SEN")].copy()
    reference = load_reference_uncertainty(args.reference_uncertainty)
    n_units = len(units)
    units = units.merge(
        reference,
        on=["dinsar", "Id"],
        how="inner",
        validate="one_to_one",
    )
    print(f"dropping {n_units - len(units)} units of DInSAR delineations discarded "
          f"from the reference-uncertainty product")
    pair_intervals = args.postprocessing_dir / "pair_intervals.parquet"
    pair_post = load_pair_posteriors(pair_intervals)
    print(f"{len(units)} Sentinel-1 (DInSAR, flow-bend) units "
          f"within {threshold:.0f} m of a flow-bend line")

    scenarios = [("none", 0.0), ("source", "reference_sigma_m")]
    results = {}
    comparison_rows = []
    for scenario_name, sigma_ref in scenarios:
        res = run_calibration(
            units,
            pair_post,
            sigma_ref=sigma_ref,
            max_span=max_span,
            gl_prior=gl_prior,
        )
        res["kind"] = scenario_name
        display_sigma = float(sigma_ref) if not isinstance(sigma_ref, str) else np.nan
        res["sigma_ref"] = display_sigma
        levels, coverage, low, high = report(res)
        comparison_rows.append({
            "filter": "paper",
            "scenario": scenario_name,
            "sigma_ref": display_sigma,
            "levels": levels,
            "coverage": coverage,
            "low": low,
            "high": high,
        })
        results[scenario_name] = res

    report_reference_comparison(comparison_rows)

    args.calibration_dir.mkdir(parents=True, exist_ok=True)
    pit = pd.concat([
        pd.DataFrame({k: v for k, v in r.items() if k != "kind"}).assign(scenario=name)
        for name, r in results.items()
    ], ignore_index=True)
    pit.to_parquet(pit_out, index=False)
    coverage = pd.concat([
        pd.DataFrame({
            "scenario": row["scenario"],
            "sigma_ref": row["sigma_ref"],
            "level": row["levels"],
            "coverage": row["coverage"],
            "low": row["low"],
            "high": row["high"],
        })
        for row in comparison_rows
    ], ignore_index=True)
    coverage.to_parquet(coverage_out, index=False)
    print("wrote", pit_out)
    print("wrote", coverage_out)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    figure = plot_reliability_and_pit(
        results,
        savepath=reliability_out,
    )
    plt.close(figure)
    for scenario_name, sigma_ref in scenarios:
        result = results[scenario_name]
        scenario_out = reliability_out.with_name(
            f"{reliability_out.stem}_{scenario_name}{reliability_out.suffix}"
        )
        figure = plot_reliability_and_pit(
            {scenario_name: result},
            savepath=scenario_out,
        )
        plt.close(figure)

        group_stem = "reliability_by_group" if scenario_name == "none" else (
            f"reliability_{scenario_name}_by_group"
        )
        group_out = reliability_out.with_name(group_stem + reliability_out.suffix)
        print(f"\n=== by year x mission [{scenario_name}] ===")
        report_by_stratum(result, lambda d, f: dinsar_group(d))
        figure = plot_reliability_by_group(result, savepath=group_out)
        plt.close(figure)

    sidecar_config = {
        "coverage_threshold": threshold,
        "identifiable_crossing_tolerance_m": max_span,
        "gl_prior": gl_prior,
        "max_reference_half_width_m": float(cal["max_reference_half_width_m"]),
        "reference_error_scenarios": {
            "none": 0.0,
            "source": "per-unit values from reference_uncertainty.parquet; nulls remain unadjusted",
        },
        "cohort": "both-pairs",
        "bootstrap_block": "dinsar",
    }
    for output in (pit_out, coverage_out):
        write_sidecar(
            output,
            repo=pathlib.Path(__file__).resolve().parents[1],
            inputs=[args.truth_cache, args.reference_uncertainty, pair_intervals],
            configuration=sidecar_config,
            seed=int(config["CEBU_params"]["seed"]),
        )


if __name__ == "__main__":
    main()
