"""Build the paper point set and credible intervals from retained BIP files."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed

from .credible_intervals import central_intervals
from .posterior_io import load_result, pair_diagnostics, posterior_samples, result_times
from .profiles import canonical_flow_bends
from .run_metadata import write_sidecar


def load_paper_point_keys(path: str | Path) -> pd.MultiIndex:
    """Return unique flow-bend/acquisition-pair keys from the paper point product."""
    points = pd.read_parquet(
        path, columns=["flowline_id", "primary_time", "secondary_time"]
    )
    points["flowline_id"] = points["flowline_id"].astype(int)
    for column in ("primary_time", "secondary_time"):
        points[column] = pd.to_datetime(points[column]).dt.strftime("%Y%m%d")
    return pd.MultiIndex.from_frame(
        points[["flowline_id", "primary_time", "secondary_time"]]
    ).unique()


def _normalise_covariate_config(config: dict) -> tuple[dict, dict]:
    """Resolve configured covariate rasters into the format ``covariates`` expects."""
    from . import covariates

    block = config.get("Covariates") or {}
    data_root = Path(config["Directories"]["work_dir"]) / config['Directories']['data_dir'] / config['Offset_params']['roi']
    resolved = {"Directories": {"data_dir": data_root}, "Covariates": {}}
    for name, spec in block.items():
        if isinstance(spec, dict):
            item = dict(spec)
            for key in ("path",):
                if key in item:
                    item[key] = str(_resolve_data_path(item[key], data_root))
            if "paths" in item:
                item["paths"] = [str(_resolve_data_path(path, data_root)) for path in item["paths"]]
            resolved["Covariates"][name] = item
        else:
            resolved["Covariates"][name] = str(_resolve_data_path(spec, data_root))
    return covariates.split_covariate_sources(resolved)


def _resolve_data_path(path: str | Path, data_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else data_root / path


def _log_evidence(result: dict) -> float:
    if "log_evidence" in result:
        return float(result["log_evidence"])
    with np.errstate(divide="ignore"):
        return float(np.log(np.float64(result["evidence"])))


def _line_products(
    flow_bend_id: int,
    geometry,
    result_root: Path,
    ness_min: float,
    tide_min: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    points: list[dict] = []
    pair_intervals: list[dict] = []
    retained_samples: list[np.ndarray] = []

    directory = result_root / str(flow_bend_id)
    for path in sorted(directory.glob("*.zstd")):
        result = load_result(path)
        samples = posterior_samples(result)
        if samples is None:
            continue
        diagnostics = pair_diagnostics(result)
        if diagnostics["nESS"] < ness_min or abs(diagnostics["diff_tides"]) < tide_min:
            continue

        primary, secondary = result_times(path)
        primary_time = pd.to_datetime(primary, format="%Y%m%d")
        secondary_time = pd.to_datetime(secondary, format="%Y%m%d")
        map_position = float(np.asarray(result["max_kdes"])[3])
        points.append({
            "flowline_id": flow_bend_id,
            "primary_time": primary_time,
            "secondary_time": secondary_time,
            **diagnostics,
            "log_evidence": _log_evidence(result),
            "geometry": shapely.line_interpolate_point(geometry, map_position),
        })

        for level, (left, right) in central_intervals(samples).items():
            pair_intervals.append({
                "flowline_id": flow_bend_id,
                "primary_time": primary_time,
                "secondary_time": secondary_time,
                "level": level,
                "left": left,
                "right": right,
                "width": right - left,
                **diagnostics,
                "geometry": shapely.MultiPoint([
                    shapely.line_interpolate_point(geometry, left),
                    shapely.line_interpolate_point(geometry, right),
                ]),
            })
        retained_samples.append(samples)

    cumulative: list[dict] = []
    if len(retained_samples) >= 3:
        mixture = np.concatenate(retained_samples)
        for level, (left, right) in central_intervals(mixture).items():
            cumulative.append({
                "flowline_id": flow_bend_id,
                "level": level,
                "left": left,
                "right": right,
                "width": right - left,
                "n_pairs": len(retained_samples),
                "geometry": shapely.MultiPoint([
                    shapely.line_interpolate_point(geometry, left),
                    shapely.line_interpolate_point(geometry, right),
                ]),
            })
    return points, pair_intervals, cumulative


def build_products(
    flow_bends: gpd.GeoDataFrame,
    result_root: str | Path,
    *,
    ness_min: float,
    tide_min: float,
    workers: int = 1,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, int]]:
    """Build all canonical postprocessing tables in memory."""
    canonical, audit = canonical_flow_bends(flow_bends)
    result_root = Path(result_root)
    rows = Parallel(n_jobs=workers)(
        delayed(_line_products)(
            int(row.Id), row.geometry, result_root, float(ness_min), float(tide_min)
        )
        for row in canonical.itertuples(index=False)
    )

    point_rows = [item for group, _, _ in rows for item in group]
    pair_rows = [item for _, group, _ in rows for item in group]
    cumulative_rows = [item for _, _, group in rows for item in group]
    crs = flow_bends.crs
    points = gpd.GeoDataFrame(point_rows, geometry="geometry", crs=crs)
    pair_intervals = gpd.GeoDataFrame(pair_rows, geometry="geometry", crs=crs)
    cumulative_intervals = gpd.GeoDataFrame(cumulative_rows, geometry="geometry", crs=crs)

    key = ["flowline_id", "primary_time", "secondary_time"]
    if points.duplicated(key).any():
        raise ValueError("paper point set contains duplicate flow-bend/acquisition-pair keys")
    audit.update({
        "paper_points": int(len(points)),
        "paper_flow_bends": int(points["flowline_id"].nunique()),
        "paper_acquisition_pairs": int(points[["primary_time", "secondary_time"]]
                                       .drop_duplicates().shape[0]),
    })
    return points, pair_intervals, cumulative_intervals, audit


def write_products(
    flow_bends_path: str | Path,
    result_root: str | Path,
    output_directory: str | Path,
    *,
    ness_min: float,
    tide_min: float,
    workers: int,
    repo: str | Path,
    seed: int | None,
    config: dict | None = None,
) -> dict[str, Path]:
    """Build and write the three merged postprocessing tables and sidecars."""
    flow_bends_path = Path(flow_bends_path)
    result_root = Path(result_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    flow_bends = gpd.read_file(flow_bends_path, engine="pyogrio")
    points, pair_intervals, cumulative_intervals, audit = build_products(
        flow_bends,
        result_root,
        ness_min=ness_min,
        tide_min=tide_min,
        workers=workers,
    )
    if config is not None and config.get("Covariates"):
        from . import covariates

        static_sources, pair_sources = _normalise_covariate_config(config)
        covariate_start = float(config["Flowline_params"]["landward"])
        static = covariates.compute_covariates(
            flow_bends, static_sources, line_start=covariate_start
        )
        pair = covariates.compute_pair_covariates(
            flow_bends, pair_sources, line_start=covariate_start
        )
        pair_intervals = covariates.attach_covariates(pair_intervals, static)
        pair_intervals = covariates.attach_pair_covariates(pair_intervals, pair)
        covariates.check_covariates(
            pair_intervals,
            list(static_sources) + list(pair_sources),
            label="pair",
        )
    outputs = {
        "points": output_directory / "paper_points.parquet",
        "pair_intervals": output_directory / "pair_intervals.parquet",
        "cumulative_intervals": output_directory / "cumulative_intervals.parquet",
    }
    points.to_parquet(outputs["points"], index=False)
    pair_intervals.to_parquet(outputs["pair_intervals"], index=False)
    cumulative_intervals.to_parquet(outputs["cumulative_intervals"], index=False)
    configuration = {"nESS_min": ness_min, "inferred_delta_a_min_m": tide_min}
    for output in outputs.values():
        write_sidecar(
            output,
            repo=repo,
            inputs=[flow_bends_path, result_root],
            configuration=configuration,
            seed=seed,
            extra={"selection_audit": audit},
        )
    return outputs


def main() -> None:
    """Console entry point for canonical product generation."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowlines", type = Path, required = True)
    parser.add_argument("--inversion-dir", type = Path, required = True)
    parser.add_argument("--postprocessing-dir", type = Path, required = True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    pp = config["Posterior_params"]
    outputs = write_products(
        args.flowlines,
        args.inversion_dir,
        args.postprocessing_dir,
        ness_min=float(pp["nESS"]),
        tide_min=float(pp["tide"]),
        workers=int(args.workers if args.workers is not None else pp.get("workers", 1)),
        repo=Path(__file__).resolve().parents[1],
        seed=int(config["CEBU_params"]["seed"]),
        config=config,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")

if __name__ == "__main__":
    main()
