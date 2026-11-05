"""Build per-comparison DInSAR reference-position uncertainties.

Manual Sentinel-1 delineations receive the configured mapping-error scale. Automatic
delineations are matched to the AIS CCI ensemble standard-deviation polygons by
acquisition metadata, and their uncertainty at each flow-bend crossing is the local
half-width of the matched polygon along its normal. Missing uncertainties are null.
Automatic delineations whose median half-width exceeds the configured maximum are
discarded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .run_metadata import write_sidecar


NORMAL_PROBE_REACH_M = 100_000.0
AUTOMATIC_MARKERS = ("_HH_", "_VV_")


def parse_automatic_uuid(value: object) -> dict:
    """Parse acquisition identity from an AIS CCI automatic-line or polygon UUID."""
    text = str(value)
    parts = text.split("_")
    if len(parts) < 9 or any(not parts[index] for index in range(9)):
        raise ValueError(f"unsupported automatic DInSAR UUID: {text!r}")
    sensor, _, relative_orbit, pass_code, look_code, processing_code = parts[:6]
    try:
        times = tuple(
            datetime.strptime(item, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            for item in parts[6:9]
        )
        orbit = int(relative_orbit)
        pass_direction = {"A": "ascending", "D": "descending"}[pass_code]
        look_direction = {"L": "left", "R": "right"}[look_code]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid automatic DInSAR UUID: {text!r}") from exc
    return {
        "uuid": text,
        "sensor": sensor,
        "relative_orbit": orbit,
        "pass_direction": pass_direction,
        "look_direction": look_direction,
        "processing_code": processing_code,
        "times": times,
        "dates": tuple(time.date() for time in times),
    }


def _match_key(metadata: dict, time_field: str) -> tuple:
    return (
        metadata["sensor"],
        metadata["relative_orbit"],
        metadata["pass_direction"],
        metadata["look_direction"],
        *metadata[time_field],
    )


def load_uncertainty_polygons(paths) -> gpd.GeoDataFrame:
    """Read polygon files and retain their UUID and source-file provenance."""
    frames = []
    for path in map(Path, paths):
        frame = gpd.read_file(path, engine="pyogrio", columns=["UUID"])
        if "UUID" not in frame or frame.geometry.name not in frame:
            raise ValueError(f"uncertainty file lacks UUID or geometry: {path}")
        geometry_name = frame.geometry.name
        frame = frame[["UUID", geometry_name]].copy()
        if geometry_name != "geometry":
            frame = frame.rename_geometry("geometry")
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return gpd.GeoDataFrame(
            columns=["UUID", "source_file", "geometry"],
            geometry="geometry",
            crs="EPSG:3031",
        )
    polygons = pd.concat(frames, ignore_index=True)
    polygons = gpd.GeoDataFrame(polygons, geometry="geometry", crs=frames[0].crs)
    present = ~(polygons.geometry.isna() | polygons.geometry.is_empty)
    allowed = polygons.loc[present].geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not allowed.all():
        kinds = sorted(polygons.loc[present].loc[~allowed].geometry.geom_type.unique())
        raise ValueError(f"uncertainty files contain non-polygon geometries: {kinds}")
    return polygons


def match_lines_to_uncertainty_polygons(
    scene_ids, polygons: gpd.GeoDataFrame
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Match automatic DInSAR lines to polygons by ordered acquisition metadata.

    Exact ordered acquisition times are used first. Ordered dates are a fallback only
    when they identify one line uniquely. The returned audit has one row per polygon.
    """
    exact = defaultdict(list)
    dated = defaultdict(list)
    for scene_id in dict.fromkeys(map(str, scene_ids)):
        if not any(marker in scene_id for marker in AUTOMATIC_MARKERS):
            continue
        metadata = parse_automatic_uuid(scene_id)
        exact[_match_key(metadata, "times")].append(scene_id)
        dated[_match_key(metadata, "dates")].append(scene_id)

    matches: dict[str, dict] = {}
    duplicate_scenes = set()
    audit = []
    for row in polygons.itertuples(index=False):
        source_uuid = str(row.UUID)
        record = {
            "source_uuid": source_uuid,
            "source_file": row.source_file,
            "match_status": None,
            "candidate_count": 0,
            "matched_dinsar": None,
        }
        try:
            metadata = parse_automatic_uuid(source_uuid)
        except ValueError:
            record["match_status"] = "invalid_metadata"
            audit.append(record)
            continue
        if row.geometry is None or row.geometry.is_empty:
            record["match_status"] = "empty_geometry"
            audit.append(record)
            continue

        candidates = exact.get(_match_key(metadata, "times"), [])
        method = "exact_ordered_acquisition_times"
        status = "matched_exact_time"
        if not candidates:
            candidates = dated.get(_match_key(metadata, "dates"), [])
            method = "unique_ordered_acquisition_dates"
            status = "matched_ordered_date"
        record["candidate_count"] = len(candidates)
        if not candidates:
            record["match_status"] = "unmatched"
        elif len(candidates) > 1:
            record["match_status"] = "ambiguous"
        else:
            scene_id = candidates[0]
            record.update(match_status=status, matched_dinsar=scene_id)
            if scene_id in matches or scene_id in duplicate_scenes:
                record["match_status"] = "duplicate_polygon"
                matches.pop(scene_id, None)
                duplicate_scenes.add(scene_id)
            else:
                matches[scene_id] = {
                    "geometry": row.geometry,
                    "source_uuid": source_uuid,
                    "source_file": row.source_file,
                    "match_method": method,
                }
        audit.append(record)
    return matches, pd.DataFrame.from_records(audit)


def perpendicular_polygon_sigma(point, region) -> float | None:
    """Return a polygon's local half-width along its normal through ``point``."""
    if point is None or region is None or point.is_empty or region.is_empty:
        return None
    polygons = [
        part for part in shapely.get_parts(region) if part.geom_type == "Polygon"
    ]
    containing = next((part for part in polygons if part.covers(point)), None)
    if containing is None:
        return None
    boundary = containing.boundary
    near = point.distance(boundary)
    if near <= 0:
        return None
    foot = boundary.interpolate(boundary.project(point))
    ux, uy = (point.x - foot.x) / near, (point.y - foot.y) / near
    probe = shapely.LineString(
        [
            (point.x, point.y),
            (
                point.x + ux * NORMAL_PROBE_REACH_M,
                point.y + uy * NORMAL_PROBE_REACH_M,
            ),
        ]
    )
    far = [
        point.distance(hit)
        for hit in shapely.get_parts(probe.intersection(boundary))
        if point.distance(hit) > 0
    ]
    return (near + min(far)) / 2.0 if far else None


def build_reference_uncertainty(
    truth: pd.DataFrame,
    flowlines: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    manual_sigma_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one reference-uncertainty record per S1 calibration unit."""
    units = truth.loc[truth["dinsar"].astype(str).str.startswith("SEN")].copy()
    lines = flowlines[["Id", flowlines.geometry.name]].copy()
    lines["Id"] = lines["Id"].astype(int)
    if lines["Id"].duplicated().any():
        raise ValueError("flow-bend file contains duplicate Id values")
    geometry_by_id = lines.set_index("Id").geometry
    missing_ids = sorted(set(units["Id"].astype(int)) - set(geometry_by_id.index))
    if missing_ids:
        raise ValueError(f"truth cache contains unknown flow-bend Id values: {missing_ids[:5]}")

    matches, audit = match_lines_to_uncertainty_polygons(units["dinsar"], polygons)
    rows = []
    for record in units.itertuples(index=False):
        scene_id = str(record.dinsar)
        automatic = any(marker in scene_id for marker in AUTOMATIC_MARKERS)
        common = {"dinsar": scene_id, "Id": int(record.Id)}
        if not automatic:
            rows.append(
                {
                    **common,
                    "reference_sigma_m": float(manual_sigma_m),
                    "uncertainty_kind": "manual_mapping_error",
                    "uncertainty_status": "applied",
                    "uncertainty_source_uuid": None,
                    "uncertainty_source_file": None,
                    "uncertainty_match_method": "configured_manual_value",
                }
            )
            continue

        match = matches.get(scene_id)
        if match is None:
            rows.append(
                {
                    **common,
                    "reference_sigma_m": np.nan,
                    "uncertainty_kind": "ensemble_standard_deviation_buffer",
                    "uncertainty_status": "no_matched_polygon",
                    "uncertainty_source_uuid": None,
                    "uncertainty_source_file": None,
                    "uncertainty_match_method": None,
                }
            )
            continue

        line = geometry_by_id.loc[int(record.Id)]
        point = line.interpolate(float(record.d_true_raw))
        sigma = perpendicular_polygon_sigma(point, match["geometry"])
        rows.append(
            {
                **common,
                "reference_sigma_m": np.nan if sigma is None else float(sigma),
                "uncertainty_kind": "ensemble_standard_deviation_buffer",
                "uncertainty_status": "applied" if sigma is not None else "crossing_outside_polygon",
                "uncertainty_source_uuid": match["source_uuid"],
                "uncertainty_source_file": match["source_file"],
                "uncertainty_match_method": match["match_method"],
            }
        )
    return pd.DataFrame.from_records(rows), audit


def discard_wide_delineations(
    uncertainty: pd.DataFrame, max_half_width_m: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop all rows of automatic delineations whose median finite half-width exceeds
    ``max_half_width_m``. Returns the retained rows and the discarded medians by DInSAR id."""
    automatic = uncertainty["uncertainty_kind"].eq("ensemble_standard_deviation_buffer")
    medians = (
        uncertainty.loc[automatic].groupby("dinsar")["reference_sigma_m"].median()
    )
    discarded = medians[medians > max_half_width_m]
    keep = ~uncertainty["dinsar"].isin(discarded.index)
    return uncertainty.loc[keep].reset_index(drop=True), discarded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--flow-bends", type=Path, required=True)
    parser.add_argument("--uncertainty-polygons", type=Path, nargs="+", required=True)
    parser.add_argument("--manual-sigma-m", type=float, required=True)
    parser.add_argument("--max-half-width-m", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    truth = pd.read_parquet(args.truth_cache)
    flowlines = gpd.read_file(args.flow_bends, engine="pyogrio", columns=["Id"])
    polygons = load_uncertainty_polygons(args.uncertainty_polygons)
    uncertainty, audit = build_reference_uncertainty(
        truth, flowlines, polygons, manual_sigma_m=args.manual_sigma_m
    )
    n_units = len(uncertainty)
    uncertainty, discarded = discard_wide_delineations(
        uncertainty, args.max_half_width_m
    )
    print(
        f"discarded {discarded.size} automatic delineations "
        f"({n_units - len(uncertainty)} units) with median half-width "
        f"> {args.max_half_width_m:.0f} m"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    uncertainty.to_parquet(args.out, index=False)
    audit.to_parquet(args.audit_out, index=False)
    for output in (args.out, args.audit_out):
        write_sidecar(
            output,
            repo=Path(__file__).resolve().parents[1],
            inputs=[args.truth_cache, args.flow_bends, *args.uncertainty_polygons],
            configuration={
                "manual_sigma_m": args.manual_sigma_m,
                "max_half_width_m": args.max_half_width_m,
                "half_width_statistic": "median over the delineation's crossings",
                "discarded_delineations": int(discarded.size),
            },
        )
    print(
        f"wrote {args.out}: {len(uncertainty)} units; "
        f"{uncertainty['reference_sigma_m'].notna().sum()} with reference uncertainty"
    )
    print(f"wrote {args.audit_out}: {len(audit)} polygon link records")


if __name__ == "__main__":
    main()
