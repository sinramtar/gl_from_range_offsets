"""Summary statistics of the cumulative grounding-line posterior.

The variance of the equally weighted mixture of the K pair posteriors on a flow-bend line
decomposes as

    Var[p_bar] = mean_k sigma_k^2            (mean single-pair posterior variance)
               + mean_k (mu_k - mu_bar)^2    (between-pair variability)

Both terms are evaluated per flow-bend line and summarised by their medians. mu and sigma
are the midpoint and half-width of each pair's 68% interval in `pair_intervals.parquet`,
in the along-flow-bend arc coordinate. Interval rows are restricted to the keys of
`paper_points.parquet`. The module also reports cumulative interval widths, the inset
regions of the GL map, and the scene-to-scene spread of the DInSAR delineations.

Usage
-----
    snakemake --snakefile postprocessing/Snakefile paper
"""

import argparse
import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

from .run_metadata import write_sidecar

# The stretch at which the 95% interval is widest is located by a rolling mean over this
# many flow-bend lines, matching `gl_points_map.py` so both describe the same inset.
WIDEST_WINDOW = 20
INSET_HALF_WIDTH = 13_000.0


def load_points(path):
    """Paper GL points: one per (flow-bend, acquisition pair). Geometry unused."""
    df = gpd.read_parquet(path).drop(columns="geometry")
    df["Id"] = df["flowline_id"].astype(int)
    for col in ("primary_time", "secondary_time"):
        df[col] = pd.to_datetime(df[col])
    return df[["Id", "primary_time", "secondary_time"]]


def load_pair_posteriors(path):
    """Per-(flow-bend, pair) posterior as mu and sigma along the flow-bend, in metres."""
    ci = pd.read_parquet(path)
    ci = ci.loc[ci["level"].astype(int).eq(68)].copy()
    ci = ci.rename(columns={"flowline_id": "Id"})
    ci = ci[["Id", "primary_time", "secondary_time", "left", "right"]]
    ci = ci.dropna(subset=["left", "right"])
    ci["Id"] = ci["Id"].astype(int)
    for col in ("primary_time", "secondary_time"):
        ci[col] = pd.to_datetime(ci[col])
    key = ["Id", "primary_time", "secondary_time"]
    if ci.duplicated(key).any():
        raise ValueError("68% pair intervals contain duplicate flow-bend/acquisition-pair keys")
    ci["mu"] = (ci["left"] + ci["right"]) / 2.0
    ci["sigma"] = (ci["right"] - ci["left"]) / 2.0
    return ci


def load_cumulative_ci(path, level):
    """Cumulative (all-pairs) credible interval per flow-bend line."""
    ci = pd.read_parquet(path)
    ci = ci.loc[ci["level"].astype(int).eq(level)]
    ci = ci.rename(columns={"flowline_id": "Id"})
    ci["Id"] = ci["Id"].astype(int)
    return ci.dropna(subset=["width"])


def decompose(pairs):
    """Variance decomposition per flow-bend line. `pairs` holds the paper
    (flow-bend, pair) posteriors as mu, sigma. Returns one row per flow-bend with the
    within- and between-pair variance terms, the number of contributing pairs `K`, and
    the between-pair share of the variance."""
    g = pairs.groupby("Id")
    out = pd.DataFrame({
        "K": g.size(),
        "within_var": g["sigma"].apply(lambda s: np.mean(s ** 2)),
        "between_var": g["mu"].var(ddof=0),
    })
    out["within_sd"] = np.sqrt(out["within_var"])
    out["between_sd"] = np.sqrt(out["between_var"])
    out["total_sd"] = np.sqrt(out["within_var"] + out["between_var"])
    out["frac_between"] = out["between_var"] / (out["within_var"] + out["between_var"])
    return out.reset_index()


def widest_ci_bbox(ci95, flowlines):
    """Centre and bounding box of the stretch at which the 95% interval is widest."""
    rows = ci95.merge(flowlines[["Id", "geometry"]], on="Id").sort_values("Id")
    smoothed = (rows["width"]
                .rolling(WIDEST_WINDOW, center=True, min_periods=WIDEST_WINDOW // 4)
                .mean())
    row = rows.loc[smoothed.idxmax()]
    geom = row["geometry_y"]
    c = geom.interpolate(geom.length / 2)
    lon, lat = gpd.GeoSeries([c], crs=3031).to_crs(4326).iloc[0].coords[0]
    box = (c.x - INSET_HALF_WIDTH, c.y - INSET_HALF_WIDTH,
           c.x + INSET_HALF_WIDTH, c.y + INSET_HALF_WIDTH)
    return box, dict(Id=int(row["Id"]), lon=lon, lat=lat, local_mean_width_95=smoothed.max())


def ids_in_box(flowlines, box):
    """Flow-bend lines whose midpoint -- where the line crosses the reference GL -- is in the
    box. The lines are 40 km long, so selecting on the whole geometry would pull in lines
    that merely pass through the box far from the grounding zone."""
    mid = flowlines.geometry.apply(lambda g: g.interpolate(g.length / 2))
    inside = ((mid.x >= box[0]) & (mid.x <= box[2])
              & (mid.y >= box[1]) & (mid.y <= box[3]))
    return set(flowlines.loc[inside, "Id"])


def region_stats(name, dec, ci68, ci95, pairs, ids=None):
    """Summary statistics for region `name`, shelf-wide or over the flow-bend `ids`."""
    if ids is not None:
        dec, ci68, ci95 = (d[d["Id"].isin(ids)] for d in (dec, ci68, ci95))
        pairs = pairs[pairs["Id"].isin(ids)]
    w68, w95 = ci68["width"], ci95["width"]
    return {
        "region": name,
        "flow_bend_lines": int(dec["Id"].nunique()),
        "gl_points": int(len(pairs)),
        "acquisition_pairs": int(pairs.groupby(["primary_time", "secondary_time"]).ngroups),
        "pairs_per_line_median": float(dec["K"].median()),
        "width_68_median": float(w68.median()),
        "width_95_median": float(w95.median()),
        "width_95_max": float(w95.max()),
        "width_95_over_68": float(w95.median() / w68.median()),
        "within_sd_median": float(dec["within_sd"].median()),
        "between_sd_median": float(dec["between_sd"].median()),
        "frac_between_median": float(dec["frac_between"].median()),
    }


def width_vs_npairs(dec, ci95):
    """Pearson correlation between the cumulative 95% width and the number of
    contributing pairs `K`."""
    m = dec.merge(ci95, on="Id")
    return float(m[["width", "K"]].corr().iloc[0, 1])


def dinsar_scene_spread(truth, max_sep=500.0, years=(2019, 2021), mission="all"):
    """Scene-to-scene scatter of the DInSAR delineations, one crossing per
    (DInSAR, flow-bend), in the along-flow-bend arc coordinate. Rows are kept only where
    the delineation crosses the flow-bend (`sep` < `max_sep`, `n_cross` == 1). `mission`
    selects "s1", "csk" or "all", told apart by pair separation (CSK: 1 day, S1: 6 days).
    Returns scatter statistics over `years`."""
    t = truth[(truth["sep"] < max_sep) & (truth["n_cross"] == 1)].copy()
    if mission != "all":
        pair_days = (pd.to_datetime(t["s1"], format="%Y%m%d")
                     - pd.to_datetime(t["p1"], format="%Y%m%d")).dt.days
        t = t[pair_days == 1] if mission == "csk" else t[pair_days > 1]
    t["year"] = pd.to_datetime(t["p1"], format="%Y%m%d").dt.year
    t = t[t["year"].between(*years)]

    per_line = t.groupby("Id")["d_true"].agg(["std", "min", "max", "size"])
    per_line = per_line[per_line["size"] >= 2]
    per_line["range"] = per_line["max"] - per_line["min"]

    annual = t.groupby(["Id", "year"])["d_true"].median().unstack()
    both = annual[[years[0], years[1]]].dropna()

    return {
        "scenes": int(t["dinsar"].nunique()),
        "flow_bend_lines": int(len(per_line)),
        "scene_to_scene_sd_median": float(per_line["std"].median()),
        "scene_to_scene_range_median": float(per_line["range"].median()),
        "annual_median_spread": float(annual.median().max() - annual.median().min()),
        "paired_annual_shift_median": float((both[years[1]] - both[years[0]]).median()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--postprocessing-dir", type=pathlib.Path, required=True)
    ap.add_argument("--calibration-dir", type=pathlib.Path, required=True)
    ap.add_argument("--flow-bends", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    out_json = args.postprocessing_dir / "posterior_spread.json"

    points = load_points(args.postprocessing_dir / "paper_points.parquet")
    pair_post = load_pair_posteriors(args.postprocessing_dir / "pair_intervals.parquet")
    flowlines = gpd.read_file(args.flow_bends).to_crs(3031)
    flowlines = flowlines[~flowlines.geometry.isna()][["Id", "geometry"]]
    ci68 = load_cumulative_ci(args.postprocessing_dir / "cumulative_intervals.parquet", 68)
    ci95 = load_cumulative_ci(args.postprocessing_dir / "cumulative_intervals.parquet", 95)
    truth = pd.read_parquet(args.calibration_dir / "truth_cache.parquet")

    pairs = points.merge(pair_post, on=["Id", "primary_time", "secondary_time"], how="inner")
    dec = decompose(pairs)

    box_widest, widest = widest_ci_bbox(ci95, flowlines)
    box_crane = tuple(
        gpd.read_file(pathlib.Path(__file__).resolve().parents[1] / "rois" / "Crane.geojson")
        .to_crs(3031)
        .total_bounds
    )

    regions = [region_stats("shelf-wide", dec, ci68, ci95, pairs)]
    for name, box in (("Beaglehole/Friederichen", box_crane), ("widest 95% CI", box_widest)):
        regions.append(region_stats(name, dec, ci68, ci95, pairs, ids_in_box(flowlines, box)))

    out = {
        "paper_points": int(len(points)),
        "paper_pairs": int(points.groupby(["primary_time", "secondary_time"]).ngroups),
        "paper_flow_bend_lines": int(points["Id"].nunique()),
        "points_per_pair_median": float(
            points.groupby(["primary_time", "secondary_time"]).size().median()),
        "single_inversion_width_68_median": float(2 * pairs["sigma"].median()),
        "corr_width95_npairs": width_vs_npairs(dec, ci95),
        "widest_inset": widest,
        "regions": regions,
        "dinsar": dinsar_scene_spread(truth),
        "dinsar_s1": dinsar_scene_spread(truth, mission="s1"),
    }

    print(f"paper points               : {out['paper_points']}")
    print(f"acquisition pairs          : {out['paper_pairs']}")
    print(f"flow-bend lines with points: {out['paper_flow_bend_lines']}")
    print(f"points per pair            : median {out['points_per_pair_median']:.0f}")
    print(f"single-inversion 68% width : median {out['single_inversion_width_68_median']:.0f} m")
    print(f"corr(95% width, no. pairs) : {out['corr_width95_npairs']:+.2f}")
    print(f"widest stretch             : flow-bend {widest['Id']} at "
          f"{abs(widest['lat']):.2f}S, {abs(widest['lon']):.2f}W, "
          f"local mean 95% width {widest['local_mean_width_95'] / 1000:.1f} km")

    print(f"\n{'region':<24} {'lines':>6} {'points':>7} {'K':>4} "
          f"{'68% w':>7} {'95% w':>7} {'95 max':>7} {'95/68':>6} {'within':>7} {'between':>8} "
          f"{'f_betw':>7}")
    for r in regions:
        print(f"{r['region']:<24} {r['flow_bend_lines']:>6} {r['gl_points']:>7} "
              f"{r['pairs_per_line_median']:>4.0f} {r['width_68_median']:>7.0f} "
              f"{r['width_95_median']:>7.0f} {r['width_95_max']:>7.0f} "
              f"{r['width_95_over_68']:>6.2f} "
              f"{r['within_sd_median']:>7.0f} {r['between_sd_median']:>8.0f} "
              f"{r['frac_between_median']:>7.2f}")
    print("(widths, within and between in metres; within/between are the two variance "
          "terms as standard deviations)")

    for key, label in (("dinsar", "all missions"), ("dinsar_s1", "S1 only")):
        d = out[key]
        print(f"\nDInSAR delineations ({label}), {d['scenes']} scenes on "
              f"{d['flow_bend_lines']} flow-bend lines:")
        print(f"  scene-to-scene scatter   : median sd {d['scene_to_scene_sd_median']:.0f} m, "
              f"median full range {d['scene_to_scene_range_median'] / 1000:.1f} km")
        print(f"  annual medians span      : {d['annual_median_spread']:.0f} m "
              f"(paired 2021-2019 shift: {d['paired_annual_shift_median']:+.0f} m)")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    write_sidecar(
        out_json,
        repo=pathlib.Path(__file__).resolve().parents[1],
        inputs=[
            args.postprocessing_dir / "paper_points.parquet",
            args.postprocessing_dir / "pair_intervals.parquet",
            args.postprocessing_dir / "cumulative_intervals.parquet",
            args.calibration_dir / "truth_cache.parquet",
        ],
        configuration={"widest_window": WIDEST_WINDOW, "inset_half_width": INSET_HALF_WIDTH},
        seed=int(config["CEBU_params"]["seed"]),
    )
    print(f"\nwritten: {out_json}")


if __name__ == "__main__":
    main()
