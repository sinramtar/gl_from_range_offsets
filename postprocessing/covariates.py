"""Sample raster covariates along flow-bend lines and attach them to the
credible-interval tables. Static covariates give one value per line; per-pair
covariates give one value per (line, acquisition pair) from a multi-band raster."""

import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import rasterio as rio

from . import filtering

# Columns the interval tables already carry: the interval and the per-inversion
# diagnostics of `posterior_io.PAIR_DIAGNOSTICS`. The attach steps drop a colliding
# column before joining, so a covariate may not use one of these names; `diff_tides` is
# the inferred tide, and the modelled one is the separate `cats_diff_tide`.
RESERVED_COLUMNS = frozenset({"flowline_id", "left", "right", "width", "geometry",
                              "primary_time", "secondary_time",
                              "diff_tides", "stddev", "nESS", "ks_dist", "ks_pval",
                              "evidence"})


def sample_line(src, line, spacing=None, line_start=0.0, band=1):
    """Mean value of raster band `band` along `line`, sampled every `spacing` metres
    from `line_start`. `spacing` defaults to the pixel size. Returns NaN if the line is
    empty or falls entirely outside the raster or on nodata."""
    if line is None or line.is_empty:
        return np.nan

    if spacing is None:
        spacing = abs(src.transform.a)  # assumes square pixels

    distances = np.arange(line_start, line.length + spacing, spacing)
    len_dist = len(distances)
    if len_dist == 0:
        return np.nan
    elif len_dist == 1:
        distances = [distances]

    coords = [line.interpolate(d).coords[0] for d in distances]
    values = np.array([v[0] for v in src.sample(coords, indexes=[band])], dtype=float)

    nodata = src.meta['nodata']
    if nodata is not None:
        values[values == nodata] = np.nan

    return np.nan if np.all(np.isnan(values)) else np.nanmean(values)


def flowline_means(flowlines, raster_path, line_start=0.0, band=1, id_key="Id"):
    """Mean of a raster along every flow-bend line, as a Series indexed by line id."""
    with rio.open(raster_path) as src:
        values = [sample_line(src, geom, line_start=line_start, band=band)
                  for geom in flowlines.geometry]
    return pd.Series(values, index=flowlines[id_key].astype(int), name="value")

def compute_covariates(flowlines, sources, line_start=0.0, id_key="Id"):
    """Table of per-line covariate means, one column per entry in `sources` and one row
    per flow-bend line (keyed `flowline_id`). `sources` maps a column name to a raster
    path or a `(path, band)` pair. A raster that cannot be read raises."""
    out = pd.DataFrame(index=pd.Index(flowlines[id_key].astype(int), name="flowline_id"))
    for name, source in sources.items():
        path, band = source if isinstance(source, (tuple, list)) else (source, 1)
        try:
            out[name] = flowline_means(flowlines, path, line_start=line_start,
                                       band=band, id_key=id_key).values
        except (rio.errors.RasterioIOError, OSError) as exc:
            raise RuntimeError(
                f"static covariate '{name}': cannot read band {band} of {path}") from exc
    return out.reset_index()


def attach_covariates(ci_df, covariates, id_key="flowline_id"):
    """Left-join per-line `covariates` onto credible-interval table `ci_df` on
    `flowline_id`, returning `ci_df` with the covariate columns added."""
    value_cols = [c for c in covariates.columns if c != id_key]
    _reject_reserved(value_cols)
    existing = [c for c in value_cols if c in ci_df.columns]
    if existing:
        ci_df = ci_df.drop(columns=existing)
    return ci_df.merge(covariates, on=id_key, how="left")


def sample_line_all_bands(src, line, line_start=0.0, mode="mean"):
    """Per-band value of multi-band raster `src` for one line, as a length-`src.count`
    array. `mode="mean"` averages along the line from `line_start`; `mode="endpoint"`
    takes the single value at the seaward tip (`line.length`), ignoring `line_start`.
    NaN per band where the line is empty, off the raster, or on nodata."""
    n_bands = src.count
    if line is None or line.is_empty:
        return np.full(n_bands, np.nan)

    if mode == "endpoint":
        coords = [line.interpolate(line.length).coords[0]]
    else:
        spacing = abs(src.transform.a)  # assumes square pixels
        distances = np.arange(line_start, line.length + spacing, spacing)
        if len(distances) == 0:
            return np.full(n_bands, np.nan)
        coords = [line.interpolate(d).coords[0] for d in distances]

    values = np.array(list(src.sample(coords)), dtype=float)  # (n_coords, n_bands)
    for band, nodata in enumerate(src.nodatavals):
        if nodata is not None:
            values[values[:, band] == nodata, band] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-nan band -> nan
        return np.nanmean(values, axis=0)


def compute_pair_covariates(flowlines, pair_sources, line_start=0.0, id_key="Id"):
    """Long-form covariate table keyed [flowline_id, primary_time, secondary_time], one
    column per entry in `pair_sources`. `pair_sources` maps a column name to
    {"paths": [path, ...], "mode": "mean"|"endpoint"}; each raster is multi-band with
    one band per acquisition pair, the band described "<primary>_<secondary>". Empty
    `pair_sources` returns a frame with only the key columns."""
    keys = ["flowline_id", "primary_time", "secondary_time"]
    if not pair_sources:
        return pd.DataFrame(columns=keys)

    ids = flowlines[id_key].astype(int).values
    geoms = flowlines.geometry.values

    per_source = []
    for name, spec in pair_sources.items():
        mode = spec.get("mode", "mean")
        frames = []
        for path in spec["paths"]:
            try:
                src = rio.open(path)
            except (rio.errors.RasterioIOError, OSError) as exc:
                raise RuntimeError(
                    f"per-pair covariate '{name}': cannot read {path}") from exc
            with src:
                descriptions = src.descriptions
                if not descriptions or any(d is None for d in descriptions):
                    raise ValueError(
                        f"per-pair covariate '{name}' raster {path} has no band "
                        "descriptions; cannot map bands to acquisition pairs")
                pairs = [d.split("_", 1) for d in descriptions]  # primary, secondary
                sampled = np.array([
                    sample_line_all_bands(src, geom, line_start=line_start, mode=mode)
                    for geom in geoms])  # (n_lines, n_bands)
            n_lines, n_bands = sampled.shape
            frames.append(pd.DataFrame({
                "flowline_id": np.repeat(ids, n_bands),
                "primary_time": np.tile([p[0] for p in pairs], n_lines),
                "secondary_time": np.tile([p[1] for p in pairs], n_lines),
                name: sampled.reshape(-1)}))
        if frames:
            per_source.append(pd.concat(frames, ignore_index=True))

    if not per_source:
        return pd.DataFrame(columns=keys)

    out = per_source[0]
    for frame in per_source[1:]:
        out = out.merge(frame, on=keys, how="outer")
    return filtering._normalise_times(out)


def attach_pair_covariates(ci_df, pair_covs):
    """Left-join per-pair `pair_covs` onto per-time credible-interval table `ci_df` on
    (flowline_id, primary_time, secondary_time), returning `ci_df` with the covariate
    columns added. The join only ADDS columns, so any static covariates already attached
    are carried through untouched. Returns `ci_df` unchanged when `pair_covs` has only
    the key columns."""
    keys = ["flowline_id", "primary_time", "secondary_time"]
    value_cols = [c for c in pair_covs.columns if c not in keys]
    if not value_cols:
        return ci_df
    _reject_reserved(value_cols)
    existing = [c for c in value_cols if c in ci_df.columns]
    if existing:
        ci_df = ci_df.drop(columns=existing)
    # both sides to YYYYMMDD strings, else the time join misses silently
    ci_df = filtering._normalise_times(ci_df)
    return ci_df.merge(pair_covs, on=keys, how="left")


def _reject_reserved(value_cols):
    """Refuse a covariate named after a column the interval table already owns; the
    attach steps drop the colliding column before joining, so it would be replaced."""
    clash = sorted(RESERVED_COLUMNS.intersection(value_cols))
    if clash:
        raise ValueError(
            f"covariate(s) {clash} would overwrite the interval table's own columns; "
            "rename them in the config's `Covariates` block")


def check_covariates(ci_df, expected, label=""):
    """Fail unless every configured covariate in `expected` is a column of `ci_df`, and
    report how much of each is populated. Called before the intervals are written,
    because the merge rule deletes the per-flow-bend files it built them from."""
    missing = [name for name in expected if name not in ci_df.columns]
    if missing:
        raise ValueError(f"covariates missing from the {label} intervals: {missing}")
    for name in expected:
        print(f"  {label} covariate '{name}': "
              f"{100 * float(ci_df[name].notna().mean()):.1f} % of rows populated")


def _resolve_path(path, data_dir):
    """`path` made absolute against `data_dir` if it is relative."""
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = pathlib.Path(os.path.normpath(data_dir / path))
    return path


def split_covariate_sources(config):
    """Resolve the `Covariates` config block into (static, pair) source dicts.

    Static maps {name: (path, band)}. Per-pair (entries flagged `per_pair: true`) maps
    {name: {"paths": [path, ...], "mode": "mean"|"endpoint"}}, taking a single `path` or a
    `paths` list and `sample: endpoint` for tip sampling. Absent block returns ({}, {})."""
    block = config.get("Covariates") or {}
    static, pair = {}, {}
    data_dir = config['Directories']['data_dir']
    for name, spec in block.items():
        if isinstance(spec, dict) and spec.get("per_pair"):
            raw = spec.get("paths", spec.get("path"))
            raw = raw if isinstance(raw, (list, tuple)) else [raw]
            pair[name] = {"paths": [_resolve_path(p, data_dir) for p in raw],
                          "mode": spec.get("sample", "mean")}
        elif isinstance(spec, dict):
            static[name] = (_resolve_path(spec["path"], data_dir), spec.get("band", 1))
        else:
            static[name] = (_resolve_path(spec, data_dir), 1)
    return static, pair
