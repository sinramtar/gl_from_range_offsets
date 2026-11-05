"""Quality-control filters for the per (flow-bend, acquisition pair) inversion results.

The thresholds are read from the `Posterior_params` block of the configuration.
"""

import pandas as pd

# Interval width (m) below which a posterior counts as collapsed; reported, not cut.
COLLAPSED_CI_M = 10.0


def thresholds(config):
    """Return ``(nESS minimum, |inferred ΔA| minimum in m)`` from the configuration."""
    params = config["Posterior_params"]
    return float(params["nESS"]), float(params["tide"])


def _normalise_times(df, primary="primary_time", secondary="secondary_time"):
    """Return a copy of `df` with the `primary`/`secondary` time columns as `YYYYMMDD`
    strings, so tables that store ISO and compact dates join cleanly."""
    out = df.copy()
    for col in (primary, secondary):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y%m%d")
    return out


def converged_mask(points_df, ness_min):
    """Rows whose inversion converged: normalised effective sample size of at least
    `ness_min`."""
    return points_df.nESS >= ness_min


def informative_mask(points_df, tide_min):
    """Rows whose inferred differential tide is at least `tide_min` in magnitude."""
    return points_df.diff_tides.abs() >= tide_min
