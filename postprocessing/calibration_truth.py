"""Construct DInSAR truth positions along canonical flow-bend lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from tqdm import tqdm

from .profiles import canonical_flow_bends
from .run_metadata import write_sidecar


# Segment pairs compared in one block of the crossing search, bounding its temporaries.
MAX_SEGMENT_PAIRS = 4_000_000


def load_flowlines(path, id_key="Id", ref_id_key="ref_line_id"):
    """Load canonical flow bends and their mapped reference-line IDs."""
    columns = [id_key, ref_id_key]
    flowlines = gpd.read_file(path, engine="pyogrio", columns=columns)
    if flowlines.empty and Path(path).suffix == ".gpkg":
        flowlines = gpd.read_file(
            path, engine="pyogrio", columns=columns, rows=slice(0, 2**31 - 1)
        )
    flowlines, audit = canonical_flow_bends(flowlines, id_column=id_key)
    flowlines[id_key] = flowlines[id_key].astype(int)
    flowlines[ref_id_key] = flowlines[ref_id_key].astype(int)
    print(f"flow-bend canonicalisation: {audit}")
    return flowlines.reset_index(drop=True)


def load_reference_lines(path, id_key="Id"):
    """Load the mapped reference grounding-line parts."""
    lines = gpd.read_file(path, engine="pyogrio", columns=[id_key])
    lines[id_key] = lines[id_key].astype(int)
    return lines


def load_dinsar(dinsar_dir, pattern="*.geojson"):
    """Load one geometry and its two acquisition pairs per DInSAR scene."""
    rows = []
    for path in sorted(Path(dinsar_dir).glob(pattern)):
        frame = gpd.read_file(path, engine="pyogrio", datetime_as_string=True)
        record = frame.iloc[0]
        if "UUID" in frame.columns:
            t1, t2, t3 = (
                pd.to_datetime(record[column]).strftime("%Y%m%d")
                for column in ("T1", "T2", "T3")
            )
            scene_id, pairs = record["UUID"], [(t1, t2), (t2, t3)]
        else:
            t1, t2, t3, t4 = (
                pd.to_datetime(str(record[column])).strftime("%Y%m%d")
                for column in ("Primary1", "Secondary1", "Primary2", "Secondary2")
            )
            scene_id, pairs = record["DD"], [(t1, t2), (t3, t4)]
        rows.append({"dinsar": scene_id, "geometry": record.geometry, "pairs": pairs})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3031")


def _crossings(arc, separation, tolerance=5.0, gap=100.0):
    """Return distinct near-touching stretches as ``(arc, separation, span)``."""
    touching = np.where(
        separation <= max(separation.min() + tolerance, tolerance)
    )[0]
    if touching.size == 0:
        return []
    groups = np.split(touching, np.where(np.diff(arc[touching]) > gap)[0] + 1)
    return [
        (
            float(arc[group[np.argmin(separation[group])]]),
            float(separation[group].min()),
            float(arc[group].max() - arc[group].min()),
        )
        for group in groups
    ]


class FlowbendIndex:
    """Flattened full-resolution and decimated flow-bend vertices."""

    def __init__(self, geometries, ids, stride=200):
        self.ids = np.asarray(ids)
        self.geometries = np.asarray(geometries, dtype=object)
        vertex_counts = shapely.get_num_coordinates(self.geometries)
        self.starts = np.concatenate([[0], np.cumsum(vertex_counts)])
        self.coordinates = shapely.get_coordinates(self.geometries)

        steps = np.zeros(len(self.coordinates))
        steps[1:] = np.hypot(*np.diff(self.coordinates, axis=0).T)
        steps[self.starts[:-1]] = 0.0
        self.arc = np.cumsum(steps)
        self.arc -= np.repeat(self.arc[self.starts[:-1]], vertex_counts)

        coarse = []
        for start, stop in zip(self.starts[:-1], self.starts[1:]):
            indices = np.arange(start, stop, stride)
            if indices.size == 0 or indices[-1] != stop - 1:
                indices = np.append(indices, stop - 1)
            coarse.append(indices)
        self.coarse_indices = np.concatenate(coarse)
        self.coarse_starts = np.concatenate(
            [[0], np.cumsum([indices.size for indices in coarse])]
        )
        self.coarse_coordinates = self.coordinates[self.coarse_indices]
        self.margin = np.array(
            [
                np.diff(self.arc[indices]).max() / 2.0 if indices.size > 1 else 0.0
                for indices in coarse
            ]
        )

    def line_slice(self, index):
        start, stop = self.starts[index], self.starts[index + 1]
        return self.arc[start:stop], start, stop - start


def truths_for_dinsar_indexed(
    line, index, tree, threshold=100.0, densify=10.0, tolerance=5.0, workers=-1
):
    """Find DInSAR crossings with a coarse-to-fine vertex search."""
    candidates = np.atleast_1d(
        tree.query(line, predicate="dwithin", distance=threshold)
    )
    if candidates.size == 0:
        return {}

    reference = cKDTree(shapely.get_coordinates(shapely.segmentize(line, densify)))
    take = np.concatenate(
        [
            np.arange(index.coarse_starts[i], index.coarse_starts[i + 1])
            for i in candidates
        ]
    )
    coarse_separation, _ = reference.query(
        index.coarse_coordinates[take], workers=workers
    )
    coarse_offsets = np.concatenate(
        [
            [0],
            np.cumsum(
                index.coarse_starts[candidates + 1]
                - index.coarse_starts[candidates]
            ),
        ]
    )

    keep_distance = threshold + tolerance
    windows, refine = [], []
    for position, candidate in enumerate(candidates):
        separation = coarse_separation[
            coarse_offsets[position] : coarse_offsets[position + 1]
        ]
        hits = np.flatnonzero(
            separation <= keep_distance + index.margin[candidate]
        )
        if hits.size == 0:
            continue
        coarse_indices = index.coarse_indices[
            index.coarse_starts[candidate] : index.coarse_starts[candidate + 1]
        ]
        start = index.starts[candidate]
        count = index.starts[candidate + 1] - start
        mask = np.zeros(count, dtype=bool)
        for lower, upper in zip(
            coarse_indices[np.maximum(hits - 1, 0)] - start,
            coarse_indices[np.minimum(hits + 1, coarse_indices.size - 1)] - start,
        ):
            mask[lower : upper + 1] = True
        local = np.flatnonzero(mask)
        windows.append((candidate, local))
        refine.append(start + local)

    if not refine:
        return {}
    fine_separation, _ = reference.query(
        index.coordinates[np.concatenate(refine)], workers=workers
    )
    fine_offsets = np.concatenate(
        [[0], np.cumsum([local.size for _, local in windows])]
    )

    truths = {}
    for position, (candidate, local) in enumerate(windows):
        arc, _, count = index.line_slice(candidate)
        separation = np.full(count, np.inf)
        separation[local] = fine_separation[
            fine_offsets[position] : fine_offsets[position + 1]
        ]
        if separation.min() > threshold:
            continue
        crossings = _crossings(arc, separation, tolerance=tolerance)
        best = min(crossings, key=lambda crossing: crossing[1])
        truths[index.ids[candidate]] = (
            best[0],
            best[1],
            len(crossings),
            best[2],
        )
    return truths


def truths_for_dinsar(line, ids, tree, geometries, threshold=100.0, densify=10.0):
    """Reference vertex search used to validate the indexed implementation."""
    candidates = np.atleast_1d(
        tree.query(line, predicate="dwithin", distance=threshold)
    )
    if candidates.size == 0:
        return {}

    reference = cKDTree(shapely.get_coordinates(shapely.segmentize(line, densify)))
    subset = geometries[candidates]
    coordinates = shapely.get_coordinates(subset)
    bounds = np.concatenate(
        [[0], np.cumsum(shapely.get_num_coordinates(subset))]
    )
    separations, _ = reference.query(coordinates, workers=-1)

    truths = {}
    for position, candidate in enumerate(candidates):
        start, stop = bounds[position], bounds[position + 1]
        separation = separations[start:stop]
        if separation.min() > threshold:
            continue
        xy = coordinates[start:stop]
        arc = np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        )
        crossings = _crossings(arc, separation)
        best = min(crossings, key=lambda crossing: crossing[1])
        truths[ids[candidate]] = (best[0], best[1], len(crossings), best[2])
    return truths


def _crossing_arcs(line, part):
    """Return the arc coordinates at which ``line`` meets ``part``.
    """
    line_xy = shapely.get_coordinates(line)
    part_xy = shapely.get_coordinates(part)
    if len(line_xy) < 2 or len(part_xy) < 2:
        return np.empty(0)

    starts, ends = line_xy[:-1], line_xy[1:]
    lengths = np.hypot(*(ends - starts).T)
    arc_start = np.concatenate([[0.0], np.cumsum(lengths)])[:-1]
    low, high = np.minimum(starts, ends), np.maximum(starts, ends)

    bounds_low, bounds_high = part_xy.min(axis=0), part_xy.max(axis=0)
    nearby = np.flatnonzero(
        (low[:, 0] <= bounds_high[0])
        & (high[:, 0] >= bounds_low[0])
        & (low[:, 1] <= bounds_high[1])
        & (high[:, 1] >= bounds_low[1])
    )
    if nearby.size == 0:
        return np.empty(0)

    part_starts, part_ends = part_xy[:-1], part_xy[1:]
    part_low = np.minimum(part_starts, part_ends)
    part_high = np.maximum(part_starts, part_ends)

    arcs = []
    chunk = max(1, MAX_SEGMENT_PAIRS // len(part_starts))
    for base in range(0, nearby.size, chunk):
        block = nearby[base : base + chunk]
        overlapping = (
            (low[block, 0][:, None] <= part_high[None, :, 0])
            & (high[block, 0][:, None] >= part_low[None, :, 0])
            & (low[block, 1][:, None] <= part_high[None, :, 1])
            & (high[block, 1][:, None] >= part_low[None, :, 1])
        )
        rows, columns = np.nonzero(overlapping)
        if rows.size == 0:
            continue

        segments = block[rows]
        origin = starts[segments]
        direction = ends[segments] - origin
        part_origin = part_starts[columns]
        part_direction = part_ends[columns] - part_origin
        denominator = (
            direction[:, 0] * part_direction[:, 1]
            - direction[:, 1] * part_direction[:, 0]
        )
        offset = part_origin - origin
        with np.errstate(divide="ignore", invalid="ignore"):
            along = (
                offset[:, 0] * part_direction[:, 1]
                - offset[:, 1] * part_direction[:, 0]
            ) / denominator
            across = (
                offset[:, 0] * direction[:, 1] - offset[:, 1] * direction[:, 0]
            ) / denominator
        crossing = (
            (denominator != 0)
            & (along >= 0.0)
            & (along <= 1.0)
            & (across >= 0.0)
            & (across <= 1.0)
        )
        if crossing.any():
            hit = segments[crossing]
            arcs.append(arc_start[hit] + along[crossing] * lengths[hit])

        # Parallel segments meet along a stretch rather than at a point. Two-point
        # inputs carry no self-touching vertices, so GEOS resolves them cheaply.
        for index in np.flatnonzero(denominator == 0):
            segment, column = segments[index], columns[index]
            overlap = shapely.LineString(
                [starts[segment], ends[segment]]
            ).intersection(
                shapely.LineString([part_starts[column], part_ends[column]])
            )
            if overlap.is_empty:
                continue
            for xy in shapely.get_coordinates(overlap):
                arcs.append(
                    np.atleast_1d(
                        arc_start[segment]
                        + float(np.hypot(*(xy - starts[segment])))
                    )
                )

    if not arcs:
        return np.empty(0)
    return np.unique(np.round(np.concatenate(arcs), 6))


def _reference_arcs(flowlines, reference_lines, id_key="Id", ref_id_key="ref_line_id"):
    """Return all mapped reference crossings in each flow bend's arc coordinate."""
    missing = {id_key, ref_id_key}.difference(flowlines.columns)
    if missing:
        raise ValueError(f"flow-bends need columns {sorted(missing)}")

    parts_by_id = {}
    for reference_id, geometry in zip(
        reference_lines[id_key], reference_lines.geometry
    ):
        parts_by_id.setdefault(int(reference_id), []).extend(
            shapely.get_parts(geometry)
        )
    trees = {
        reference_id: (np.asarray(parts, dtype=object), STRtree(parts))
        for reference_id, parts in parts_by_id.items()
    }

    arcs_by_flowline = {}
    for flowline_id, reference_id, line in zip(
        flowlines[id_key], flowlines[ref_id_key], flowlines.geometry
    ):
        reference_id = int(reference_id)
        if reference_id not in trees:
            raise ValueError(
                f"reference grounding line has no {id_key}={reference_id}"
            )
        parts, tree = trees[reference_id]
        candidates = np.atleast_1d(tree.query(line, predicate="intersects"))
        found = [_crossing_arcs(line, parts[candidate]) for candidate in candidates]
        arcs = np.concatenate(found) if found else np.empty(0)
        if arcs.size == 0:
            raise ValueError(
                f"flow-bend {int(flowline_id)} does not cross reference line "
                f"{reference_id}"
            )
        arcs_by_flowline[int(flowline_id)] = np.unique(np.round(arcs, 6))
    return arcs_by_flowline


def build_truths(
    flowlines,
    dinsar,
    reference_lines=None,
    threshold=100.0,
    id_key="Id",
    ref_id_key="ref_line_id",
    reference_distance=20000.0,
    n_jobs=-1,
    stride=200,
):
    """Build one truth row per covered DInSAR/flow-bend pair."""
    geometries = np.asarray(flowlines.geometry.values, dtype=object)
    ids = flowlines[id_key].values
    reference_ids = dict(
        zip(flowlines[id_key].astype(int), flowlines[ref_id_key].astype(int))
    )
    reference_arcs = _reference_arcs(
        flowlines, reference_lines, id_key=id_key, ref_id_key=ref_id_key
    )
    tree = STRtree(list(geometries))
    index = FlowbendIndex(geometries, ids, stride=stride)
    print(
        f"truth: indexed {len(ids)} flow-bends, {len(index.coordinates)} vertices "
        f"({len(index.coarse_indices)} at stride {stride})"
    )

    workers = 1 if n_jobs != 1 else -1

    def one(record):
        truths = truths_for_dinsar_indexed(
            record.geometry, index, tree, threshold=threshold, workers=workers
        )
        (primary_1, secondary_1), (primary_2, secondary_2) = record.pairs
        rows = []
        for flowline_id, (
            raw_truth,
            separation,
            crossing_count,
            crossing_span,
        ) in truths.items():
            flowline_id = int(flowline_id)
            anchors = reference_arcs[flowline_id]
            reference_arc = float(anchors[np.argmin(np.abs(anchors - raw_truth))])
            truth = float(reference_distance + raw_truth - reference_arc)
            rows.append(
                {
                    "dinsar": record.dinsar,
                    "Id": flowline_id,
                    "ref_line_id": reference_ids[flowline_id],
                    "d_true": truth,
                    "d_true_raw": raw_truth,
                    "ref_arc": reference_arc,
                    "reference_distance": float(reference_distance),
                    "sep": separation,
                    "n_cross": crossing_count,
                    "d_span": crossing_span,
                    "p1": primary_1,
                    "s1": secondary_1,
                    "p2": primary_2,
                    "s2": secondary_2,
                }
            )
        return rows

    batches = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(record)
        for record in tqdm(dinsar.itertuples(), total=len(dinsar), desc="truth")
    )
    units = pd.DataFrame([row for batch in batches for row in batch])
    print(
        f"truth: {len(units)} units, "
        f"{100 * (units.n_cross > 1).mean():.1f}% multi-crossing, "
        f"median d_span {units.d_span.median():.0f} m"
    )

    return units


def main() -> None:
    """Build the truth cache used by the calibration and the posterior-spread summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--flow-bends", type=Path, required=True)
    parser.add_argument("--reference-gl", type=Path, required=True)
    parser.add_argument("--dinsars-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    import yaml

    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    calibration = config["Calibration"]
    threshold = float(calibration["coverage_threshold"])
    reference_distance = float(config["Flowline_params"]["landward"])

    flowlines = load_flowlines(args.flow_bends)
    reference_lines = load_reference_lines(args.reference_gl)
    dinsar = load_dinsar(args.dinsars_dir)
    print(f"{len(flowlines)} flow-bends, {len(dinsar)} DInSAR scenes")
    if args.validate:
        validate_truths(flowlines, dinsar, threshold=threshold)
    units = build_truths(
        flowlines,
        dinsar,
        reference_lines,
        threshold=threshold,
        reference_distance=reference_distance,
        n_jobs=args.threads,
        stride=args.stride,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    units.to_parquet(args.out, index=False)
    write_sidecar(
        args.out,
        repo=Path(__file__).resolve().parents[1],
        inputs=[args.flow_bends, args.reference_gl, args.dinsars_dir],
        configuration={
            "coverage_threshold": threshold,
            "reference_distance": reference_distance,
            "stride": args.stride,
        },
        seed=int(config["CEBU_params"]["seed"]),
    )
    print(f"truth: cached {len(units)} units to {args.out}")


def truths_exact(line, ids, tree, geometries, threshold=100.0):
    """Find exact GEOS nearest points for validation."""
    from shapely.ops import nearest_points

    truths = {}
    for candidate in np.atleast_1d(
        tree.query(line, predicate="dwithin", distance=threshold)
    ):
        flowline = geometries[candidate]
        point = nearest_points(line, flowline)[1]
        separation = point.distance(line)
        if separation <= threshold:
            truths[ids[candidate]] = (
                flowline.project(point),
                float(separation),
            )
    return truths


def validate_truths(
    flowlines, dinsar, threshold=100.0, n_scenes=3, max_span=100.0, id_key="Id"
):
    """Compare the vertex search with exact GEOS for identifiable units."""
    geometries = np.asarray(flowlines.geometry.values, dtype=object)
    ids = flowlines[id_key].values
    tree = STRtree(list(geometries))

    differences, separation_errors = [], []
    for line in dinsar.geometry.values[:n_scenes]:
        fast = truths_for_dinsar(
            line, ids, tree, geometries, threshold=threshold
        )
        exact = truths_exact(line, ids, tree, geometries, threshold=threshold)
        for flowline_id in set(fast) & set(exact):
            truth, separation, crossing_count, crossing_span = fast[flowline_id]
            if crossing_count != 1 or crossing_span > max_span:
                continue
            differences.append(abs(truth - exact[flowline_id][0]))
            separation_errors.append(separation - exact[flowline_id][1])
    differences = np.asarray(differences)
    separation_errors = np.asarray(separation_errors)
    print(
        f"validate_truths: {len(differences)} identifiable units over "
        f"{n_scenes} scenes; |d_true| diff median {np.median(differences):.2f} m "
        f"/ max {differences.max():.2f} m; sep over-estimate max "
        f"{separation_errors.max():+.2f} m"
    )
    assert differences.max() <= max_span, (
        f"KD-tree d_true disagrees with GEOS by {differences.max():.1f} m"
    )
    assert separation_errors.max() <= 6.0, (
        f"sep over-estimated by {separation_errors.max():.1f} m, expected <= densify/2"
    )
    return differences, separation_errors


if __name__ == "__main__":
    main()
