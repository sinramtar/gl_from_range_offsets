import re
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio as rio


def correlation_bins(correlation: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Index of the equal-width correlation bin on [0, 1], closed on the right; values at
    or below 0 fall in the first bin and values above 1 in the last."""
    edges = np.linspace(0, 1, n_bins + 1)
    return np.clip(np.digitize(np.nan_to_num(correlation), edges, right=True) - 1,
                   0, n_bins - 1)


def fill_error_std(errors: np.ndarray,
                   correlation: np.ndarray,
                   valids_mask: np.ndarray,
                   n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Assign every valid pixel the standard deviation of the stable-pixel offsets in its
    correlation bin.

    The correlation range [0, 1] is split into `n_bins` equal bins, closed on the right
    ((0, 0.1], ..., (0.9, 1.0] for ten bins); values at or below 0 fall in the first bin
    and values above 1 in the last. Pixels with a non-finite correlation, or outside
    `valids_mask`, stay NaN, as do bins with fewer than two stable pixels.

    Parameters
    ----------
    errors : np.ndarray
        2-D range offsets of the stable pixels, NaN elsewhere.
    correlation : np.ndarray
        2-D normalised cross-correlation coefficient of the offset tracking.
    valids_mask : np.ndarray
        2-D boolean mask of the pixels that receive an error estimate.
    n_bins : int
        Number of equal-width correlation bins on [0, 1].

    Returns
    -------
    filled : np.ndarray
        Per-pixel error standard deviation (m).
    bin_stds : np.ndarray
        Standard deviation per bin, NaN for bins without stable pixels.
    """
    assignable = valids_mask & np.isfinite(correlation)
    bins = correlation_bins(correlation, n_bins)
    stable = assignable & np.isfinite(errors)

    filled = np.full(errors.shape, np.nan, dtype=float)
    bin_stds = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        in_bin = stable & (bins == b)
        if np.count_nonzero(in_bin) >= 2:
            bin_stds[b] = np.nanstd(errors[in_bin], ddof=0)
            filled[assignable & (bins == b)] = bin_stds[b]
    return filled, bin_stds


def get_data_error(velocity_mag, detrended_offsets, correlations_tif, output,
                   stable_speed=0.01):
    """Per acquisition pair, write the range-offset error raster `output`.

    Stable pixels are those with ice speed below `stable_speed` (m/day) in the
    `velocity_mag` raster. For every pair present in both the detrended offsets and the
    correlation raster (matched by band description `YYYYMMDD_YYYYMMDD`), the stable-pixel
    offsets are binned by correlation with `fill_error_std`. One output band per matched
    pair, in the order of `detrended_offsets`.
    """
    with rio.open(correlations_tif) as tif:
        meta = tif.meta.copy()
        bounds = tif.bounds
        correlation_band = {}
        for index, description in enumerate(tif.descriptions, start=1):
            correlation_band.setdefault(description, index)

    with rio.open(velocity_mag) as tif:
        window = rio.windows.from_bounds(*bounds, transform=tif.transform)
        speed = tif.read(1, window=window)

    matched = []
    for path in detrended_offsets:
        with rio.open(path) as tif:
            matched += [(path, index, description)
                        for index, description in enumerate(tif.descriptions, start=1)
                        if description in correlation_band]

    meta.update({"count": len(matched), "dtype": "float32", "nodata": np.nan})
    with rio.open(output, "w", **meta) as dest, rio.open(correlations_tif) as corr:
        for band, (path, index, description) in enumerate(matched, start=1):
            with rio.open(path) as tif:
                window = rio.windows.from_bounds(*bounds, transform=tif.transform)
                range_shift = tif.read(index, window=window)
            correlation = corr.read(correlation_band[description])
            stable = np.where(speed < stable_speed, range_shift, np.nan)
            error, _ = fill_error_std(errors=stable, correlation=correlation,
                                      valids_mask=~np.isnan(range_shift))
            dest.write_band(band, error.astype(np.float32))
            dest.set_band_description(band, description)


def error_sample(data_error_tif, correlations_tif, correlation_threshold, output):
    """Write the error sample the likelihood marginalises over.

    Collects the finite per-pixel errors of `data_error_tif` at pixels whose correlation
    coefficient is at least `correlation_threshold`, pairing each band with the
    correlation band of the same description. The Parquet `output` has the columns
    `error` (m) and `correlation`.
    """
    errors, correlations = [], []
    with rio.open(data_error_tif) as err, rio.open(correlations_tif) as corr:
        correlation_band = {}
        for index, description in enumerate(corr.descriptions, start=1):
            correlation_band.setdefault(description, index)
        for index, description in enumerate(err.descriptions, start=1):
            error = err.read(index)
            correlation = corr.read(correlation_band[description])
            keep = np.isfinite(error) & (correlation >= correlation_threshold)
            errors.append(error[keep])
            correlations.append(correlation[keep])
    sample = pd.DataFrame({"error": np.concatenate(errors),
                           "correlation": np.concatenate(correlations)})
    if sample.empty:
        raise ValueError(f"no finite errors at correlation >= {correlation_threshold}")
    sample.to_parquet(output, index=False)
    return sample


def get_sar_acquisition_times(raster_name: str) -> list:
    """Acquisition dates (`YYYYMMDD`) encoded in a raster or result filename."""
    matches = re.findall(r'\d{4}\d{2}\d{2}', raster_name)
    return [datetime.strptime(match, '%Y%m%d') for match in matches]


def detrend_offsets(offsets, bounds, temporal_baseline, range_pixel_spacing):
    """Stack the offset rasters of one temporal baseline (days) within `bounds`, convert
    them from range pixels per day to metres over the baseline, and subtract the per-pixel
    mean over the stack. Returns (offsets, acquisition times), sorted by primary time."""
    offsets_ts = []
    acq_times = []
    for offset in offsets:
        times = get_sar_acquisition_times(offset.name)
        if np.abs(times[0] - times[1]).days != temporal_baseline:
            continue
        with rio.open(offset) as tif:
            window = rio.windows.from_bounds(left = bounds[0], right = bounds[2], bottom = bounds[1], top = bounds[3], transform = tif.transform)
            band = tif.read(1, window = window)
            if tif.nodata is not None:
                band[band == tif.nodata] = np.nan
        offsets_ts.append(band)
        acq_times.append(times)

    offsets_ts = np.array(offsets_ts)
    acq_times = np.array(acq_times)
    sort_time = np.argsort(acq_times[:, 0])
    acq_times = acq_times[sort_time]
    offsets_ts = offsets_ts[sort_time]

    offsets_ts = offsets_ts * range_pixel_spacing * temporal_baseline
    offsets_ts = offsets_ts - np.nanmean(offsets_ts, axis = 0)
    return offsets_ts, acq_times


def get_vertical_component_from_offsets(offsets_ts, incidence_angles_raster, bounds):
    """Divide line-of-sight offsets by the cosine of the incidence angle (degrees)."""
    with rio.open(incidence_angles_raster) as tif:
        window = rio.windows.from_bounds(left = bounds[0], right = bounds[2], bottom = bounds[1], top = bounds[3], transform = tif.transform)
        incidence_angles = np.deg2rad(tif.read(1, window = window))
    return offsets_ts / np.cos(incidence_angles)


def save_complete_profile(offset, time, savedir):
    """Save one acquisition pair's profile as `<time>.npz` if it has no missing samples.

    The array holds two rows, the offsets and an observation mask of ones, the layout the
    inversion reads. Returns whether the profile was saved.
    """
    if np.isnan(offset).any():
        return False
    np.savez_compressed(savedir / f'{time}.npz', np.vstack([offset, np.ones_like(offset)]))
    return True
