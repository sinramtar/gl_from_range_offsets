"""Credible intervals of equally weighted posterior draws."""

from __future__ import annotations

import numpy as np


def weighted_quantile(samples, quantiles, weights):
    """Quantiles obtained by inverting a weighted empirical CDF."""
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(samples)
    cumulative = np.cumsum(weights[order])
    if cumulative.size == 0 or cumulative[-1] <= 0:
        raise ValueError("weighted quantiles need at least one positive weight")
    return np.interp(quantiles, cumulative / cumulative[-1], samples[order])


def central_intervals(samples, levels=(68, 95), weights=None):
    """Return central credible intervals keyed by percentage level."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        raise ValueError("credible intervals need at least one sample")
    intervals = {}
    for level in levels:
        alpha = 1.0 - float(level) / 100.0
        quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
        bounds = (
            np.quantile(samples, quantiles)
            if weights is None
            else weighted_quantile(samples, quantiles, weights)
        )
        intervals[int(level)] = tuple(float(value) for value in bounds)
    return intervals
