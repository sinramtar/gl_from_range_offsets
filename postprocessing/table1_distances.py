"""Distance measures between the inverted GL points and the DInSAR delineations,
per year x mission stratum, for the paper point set.

Method:

  per DInSAR scene, per constituent pair -> take the GL points of that exact
  range-offset pair (Sentinel) or of the pair with the nearest primary date (CSK),
  clip them to the scene bounds, measure each point's Euclidean distance to the
  nearest delineation part, keep points inside the matched part's bbox, sign the
  distance, and drop |modified z| >= 3 outliers per (scene, pair).

Usage
-----
    snakemake --snakefile postprocessing/Snakefile paper
"""

import argparse
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.stats import median_abs_deviation

from . import calibrate, calibration_truth, filtering
from .run_metadata import write_sidecar


def load_points(path):
    """The canonical paper point set."""
    pts = gpd.read_parquet(path)
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty].copy()
    pts = filtering._normalise_times(pts)
    pts["flowline_id"] = pts["flowline_id"].astype(int)
    return pts


def match_csk_pairs(scenes, points, max_dt_days=None):
    """Replace each CSK scene's pair dates with the nearest inverted pair's, matched on
    the primary date over the whole point set (`max_dt_days` caps the gap, dropping
    scenes beyond it). Sentinel scenes pass through untouched. Returns the scenes frame
    with matched p1/p2 (s* set to None)."""
    avail = np.sort(pd.to_datetime(points["primary_time"].unique(), format="%Y%m%d").values)
    out, dts = [], []
    for rec in scenes.itertuples():
        row = {"dinsar": rec.dinsar, "geometry": rec.geometry,
               "p1": rec.pairs[0][0], "s1": rec.pairs[0][1],
               "p2": rec.pairs[1][0], "s2": rec.pairs[1][1], "dt": 0.0}
        if not str(rec.dinsar).startswith("SEN"):
            # CSK: nearest primary date; the secondary is then unconstrained
            for k in ("p1", "p2"):
                want = pd.to_datetime(row[k], format="%Y%m%d").to_datetime64()
                near = avail[np.argmin(np.abs(avail - want))]
                row["dt"] = max(row["dt"], abs((near - want) / np.timedelta64(1, "D")))
                row[k] = pd.Timestamp(near).strftime("%Y%m%d")
            row["s1"] = row["s2"] = None
            if max_dt_days is not None and row["dt"] > max_dt_days:
                continue
            dts.append(row["dt"])
        out.append(row)
    if dts:
        print(f"csk match: {len(dts)} scenes, median |dt| {np.median(dts):.0f} d, "
              f"max {np.max(dts):.0f} d")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=scenes.crs)


class BasemapSigner:
    """Sign a point by the categorised basemap.

    A point's sign is the category of the largest basemap polygon inside the 5-km
    single-sided buffer on the point's side of its matched delineation part: Land -> -1
    (landward), Ice shelf / Ocean -> +1 (seaward). A point in neither buffer is
    attributed to the right-hand side; an unmapped category yields NaN. Cached per
    (scene, part, side)."""
    CATEGORY = {"Ice shelf": 1.0, "Land": -1.0, "Ocean": 1.0}

    def __init__(self, basemap, buffer_m=5000.0):
        ex = basemap.explode(index_parts=False)
        self.geoms = np.asarray(ex.geometry.values, dtype=object)
        self.cats = ex["Category"].values
        self.tree = shapely.STRtree(self.geoms)
        self.buffer_m = buffer_m
        self._cache = {}

    def _side(self, part, key, left):
        k = (key, left)
        if k not in self._cache:
            buf = shapely.buffer(part, self.buffer_m if left else -self.buffer_m,
                                 single_sided=True)
            cand = self.tree.query(buf, predicate="intersects")
            if cand.size == 0:
                sign = np.nan
            else:
                areas = shapely.area(shapely.intersection(self.geoms[cand], buf))
                sign = self.CATEGORY.get(self.cats[cand[np.argmax(areas)]], np.nan)
            self._cache[k] = (buf, sign)
        return self._cache[k]

    def sign(self, points, parts, part_idx, scene_key):
        out = np.empty(len(points))
        for i in range(len(points)):
            buf_l, sign_l = self._side(parts[i], (scene_key, part_idx[i]), True)
            if shapely.intersects(points[i], buf_l):
                out[i] = sign_l
            else:
                out[i] = self._side(parts[i], (scene_key, part_idx[i]), False)[1]
        return out


def _zcut(signed, z_max):
    """Drop |modified z| >= `z_max` outliers from signed distances `signed`."""
    if len(signed) > 2:
        z = (0.6745 * (signed - np.nanmedian(signed))) / (median_abs_deviation(signed) or np.nan)
        signed = signed[np.abs(np.nan_to_num(z)) < z_max]
    return signed


def scene_distances(scene_geom, pts, fl_geoms, z_max=3.0, bm=None, scene_key=None):
    """Signed distances of the GL points of one (scene, pair) to the delineation, under
    the arc convention and, when `bm` is given, the basemap one too. `pts` must already be
    the points of the matched pair. Points must fall inside the bbox of their nearest
    delineation part, and per-(scene, pair) |modified z| >= `z_max` outliers are dropped,
    per sign convention. Returns {variant: signed distances}."""
    pts = pts.iloc[pts.sindex.query(shapely.box(*scene_geom.bounds), predicate="intersects")]
    if pts.empty:
        return {}

    parts = shapely.get_parts(scene_geom)
    tree = shapely.STRtree(parts)
    near = tree.query_nearest(pts.geometry.values, all_matches=False)
    matched = np.full(len(pts), -1)
    matched[near[0]] = near[1]
    keep = matched >= 0
    pts, matched = pts.iloc[keep], matched[keep]
    part_geoms = parts[matched]

    # coverage cut: the point must fall in its matched part's bbox
    boxes = shapely.box(*np.array([shapely.bounds(g) for g in part_geoms]).T)
    keep = shapely.intersects(boxes, pts.geometry.values)
    if not keep.any():
        return {}
    pts, part_geoms, matched = pts.iloc[keep], part_geoms[keep], matched[keep]

    # points on flow-bends missing from the geometry file cannot be arc-signed
    fls = np.array([fl_geoms.get(i) for i in pts["flowline_id"].values], dtype=object)
    ok = np.array([g is not None for g in fls])
    pts, part_geoms, matched, fls = pts.iloc[ok], part_geoms[ok], matched[ok], fls[ok]
    if len(pts) == 0:
        return {}
    dist = shapely.distance(pts.geometry.values, part_geoms)

    # arc convention: seaward of the delineation, along the flow-bend, = positive
    ref_pt = shapely.get_point(shapely.shortest_line(pts.geometry.values, part_geoms), -1)
    s_pt = shapely.line_locate_point(fls, pts.geometry.values)
    s_ref = shapely.line_locate_point(fls, ref_pt)
    out = {"arc": _zcut(np.where(s_pt >= s_ref, dist, -dist), z_max)}

    if bm is not None:
        sgn = bm.sign(pts.geometry.values, part_geoms, matched, scene_key)
        good = np.isfinite(sgn)
        out["basemap"] = _zcut(sgn[good] * dist[good], z_max)
    return out

def table1(points, scenes, fl_geoms, bm=None):
    """One Table row per year x mission group and sign variant, pooling all
    point distances. MAD is computed on the absolute distances."""
    by_primary = dict(tuple(points.groupby("primary_time")))
    by_pair = dict(tuple(points.groupby(["primary_time", "secondary_time"])))
    rows = []
    for rec in scenes.itertuples():
        group = calibrate.dinsar_group(rec.dinsar)
        for k in ("1", "2"):
            p, s = getattr(rec, f"p{k}"), getattr(rec, f"s{k}")
            # Sentinel names its exact pair; CSK constrains the primary only
            pts = by_primary.get(p) if pd.isna(s) else by_pair.get((p, s))
            if pts is None:
                continue
            res = scene_distances(rec.geometry, pts, fl_geoms, bm=bm, scene_key=rec.dinsar)
            for variant, d in res.items():
                if len(d):
                    rows.append(pd.DataFrame({"group": group, "dinsar": rec.dinsar,
                                              "variant": variant, "d": d}))
    df = pd.concat(rows, ignore_index=True)

    out = df.groupby(["variant", "group"]).agg(
        n_ifg=("dinsar", "nunique"),
        n_pts=("d", "size"),
        abs_median=("d", lambda x: np.median(np.abs(x))),
        abs_mean=("d", lambda x: np.mean(np.abs(x))),
        signed_median=("d", "median"),
        signed_mean=("d", "mean"),
        mad_abs=("d", lambda x: median_abs_deviation(np.abs(x))),
    )
    return out.round(1), df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--postprocessing-dir", type=pathlib.Path, required=True)
    ap.add_argument("--dinsars-dir", type=pathlib.Path, required=True)
    ap.add_argument("--flow-bends", type=pathlib.Path, required=True)
    ap.add_argument("--basemap", type=pathlib.Path, required=True)
    ap.add_argument("--csk-max-dt", type=float, default=None,
                    help="drop CSK scenes matched farther than this many days")
    args = ap.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    out_parquet = args.postprocessing_dir / "table1_distances_points.parquet"
    out_csv = args.postprocessing_dir / "table1_distances_points.csv"

    flowlines = calibration_truth.load_flowlines(args.flow_bends)
    fl_geoms = dict(zip(flowlines["Id"], flowlines.geometry))
    scenes = calibration_truth.load_dinsar(args.dinsars_dir)
    print(f"{len(scenes)} DInSAR scenes, {len(fl_geoms)} flow-bends")

    roi = gpd.read_file(pathlib.Path(__file__).resolve().parents[1] / "rois" / "LarsenC.geojson")
    roi = roi.to_crs("EPSG:3031")
    bm = BasemapSigner(gpd.read_file(args.basemap).clip(roi))
    print(f"basemap: {len(bm.geoms)} polygons after ROI clip")

    points = load_points(args.postprocessing_dir / "paper_points.parquet")
    matched = match_csk_pairs(scenes, points, max_dt_days=args.csk_max_dt)
    tab, df = table1(points, matched, fl_geoms, bm=bm)
    print(f"\n[paper] {len(points)} GL points")
    print(tab.to_string())

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.assign(kind="paper").to_parquet(out_parquet, index=False)
    df.assign(kind="paper").to_csv(out_csv, index=False)
    write_sidecar(
        out_parquet,
        repo=pathlib.Path(__file__).resolve().parents[1],
        inputs=[args.postprocessing_dir / "paper_points.parquet", args.dinsars_dir,
                args.flow_bends, args.basemap],
        configuration={"csk_max_dt_days": args.csk_max_dt, "sign": "basemap"},
        seed=int(config["CEBU_params"]["seed"]),
    )
    print("\nwrote", out_parquet)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
