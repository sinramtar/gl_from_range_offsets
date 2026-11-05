import numpy as np

from shapely import LineString

from scipy.ndimage import map_coordinates, gaussian_filter

def combine_velocity_and_bending_fields(velocity, offsets, lower_stddev_threshold, upper_stddev_threshold, lower_quantile = 0.05,
                                        upper_quantile = 0.95, sigma = 0.8):
    offsets_stddev = np.nanstd(offsets, axis = 0)
    offsets_stddev = gaussian_filter(offsets_stddev, sigma, order = 0, mode = "nearest")
    offsets_stddev_v, offsets_stddev_u = np.gradient(offsets_stddev)
    velocity_u, velocity_v = velocity[0, :, :], -velocity[1, :, :]

    combined_v = np.array(offsets_stddev_v)
    combined_v[offsets_stddev < lower_stddev_threshold] = velocity_v[offsets_stddev<lower_stddev_threshold]
    combined_v[offsets_stddev > upper_stddev_threshold] = velocity_v[offsets_stddev>upper_stddev_threshold]
    combined_u = np.array(offsets_stddev_u)
    combined_u[offsets_stddev < lower_stddev_threshold] = velocity_u[offsets_stddev < lower_stddev_threshold]
    combined_u[offsets_stddev > upper_stddev_threshold] = velocity_u[offsets_stddev > upper_stddev_threshold]
    
    combined_grad = np.linalg.norm(np.array([combined_u, combined_v]), axis=0)
    combined_grad_clipped = np.clip(combined_grad, np.nanquantile(combined_grad, lower_quantile), np.nanquantile(combined_grad, upper_quantile))
    combined_grad_norm = (combined_grad_clipped - np.nanmin(combined_grad_clipped))/(np.nanmax(combined_grad_clipped) - np.nanmin(combined_grad_clipped))
    return combined_grad_norm, combined_u, combined_v

def sample_raster_along_single_flowline(gdf, raster, transform, method="bilinear", nodata=None):
    """
    Sample a 3D raster (time x rows x cols) along a single flowline in a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain only one LineString geometry.
    raster : np.ndarray
        3D raster with shape (time, rows, cols)
    transform : affine.Affine
        Affine transform of raster
    method : str
        "nearest" or "bilinear"
    nodata : float or int, optional
        Nodata value in the raster, replaced with np.nan

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_time), where n_points is the number
        of vertices along the flowline.
    """
    if raster.ndim != 3:
        raise ValueError("Raster must be 3D (time x rows x cols)")
    if len(gdf) != 1:
        raise ValueError("GeoDataFrame must contain exactly one flowline")

    geom = gdf.geometry.iloc[0]
    coords = np.array(geom.coords, dtype=float)
    if coords.shape[0] == 0:
        return np.empty((0, raster.shape[0]), dtype=float)

    n_time, n_rows, n_cols = raster.shape
    xs, ys = coords[:, 0], coords[:, 1]

    # convert world coordinates to pixel indices
    ix = (xs - transform.c) / transform.a
    iy = (ys - transform.f) / transform.e

    # clip nearest indices to raster bounds
    ix_idx = np.clip(np.round(ix).astype(int), 0, n_cols - 1)
    iy_idx = np.clip(np.round(iy).astype(int), 0, n_rows - 1)

    result = np.empty((coords.shape[0], n_time), dtype=float)

    for t in range(n_time):
        if method == "nearest":
            vals = raster[t, iy_idx, ix_idx]
        else:
            vals = map_coordinates(raster[t], [iy, ix], order = 1, mode = "nearest")
        if nodata is not None:
            vals = np.where(vals == nodata, np.nan, vals)
        result[:, t] = vals

    return result

def build_interpolators(u_raster, v_raster, transform):
    """Prepare velocity fields and coordinate transforms for fast interpolation."""
    ny, nx = u_raster.shape
    x_coords = np.arange(nx) * transform.a + transform.c
    y_coords = np.arange(ny) * transform.e + transform.f
    return u_raster, v_raster, x_coords, y_coords


def sample_velocity(u_raster, v_raster, x_coords, y_coords, x, y, direction="forward"):
    """Sample velocity at given (x, y) using bilinear interpolation via map_coordinates."""
    ix = (x - x_coords[0]) / (x_coords[1] - x_coords[0])
    iy = (y - y_coords[0]) / (y_coords[1] - y_coords[0])
    velocity_u = map_coordinates(u_raster, [[iy], [ix]], order=1, mode="nearest")[0]
    velocity_v = map_coordinates(v_raster, [[iy], [ix]], order=1, mode="nearest")[0]
    if direction == "backward":
        velocity_u , velocity_v = -velocity_u, -velocity_v
    return velocity_u, velocity_v

def compute_flowline(
    u_raster,
    v_raster,
    x_coords,
    y_coords,
    start_x,
    start_y,
    length=1000,
    direction="forward",
    min_speed=1e-6,
    step=50,
    progress_window_steps=50,
    min_progress_fraction=0.5,
):
    """Compute one leg with a fixed-step RK2 integrator.

    ``length`` remains an arc-length target.  The progress guard separately
    requires the endpoint displacement over ``progress_window_steps`` to be at
    least ``min_progress_fraction`` of the corresponding path length.  This
    prevents a short oscillation from consuming the complete arc-length budget.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if progress_window_steps < 1:
        raise ValueError("progress_window_steps must be at least one")
    if not 0 <= min_progress_fraction <= 1:
        raise ValueError("min_progress_fraction must be between zero and one")

    coords = [(start_x, start_y)]
    x, y = start_x, start_y
    traveled = 0.0
    max_steps = int(np.ceil(length / step))

    for _ in range(max_steps):
        velocity_u, velocity_v = sample_velocity(u_raster, v_raster, x_coords, y_coords, x, y, direction)
        speed = np.sqrt(velocity_u**2 + velocity_v**2)
        if not np.isfinite(speed) or speed < min_speed:
            break

        step_length = min(step, length - traveled)
        dx = (velocity_u / speed) * step_length
        dy = (velocity_v / speed) * step_length

        xm = x + 0.5 * dx
        ym = y + 0.5 * dy
        um, vm = sample_velocity(u_raster, v_raster, x_coords, y_coords, xm, ym, direction)
        speed_m = np.sqrt(um**2 + vm**2)
        if not np.isfinite(speed_m) or speed_m < min_speed:
            break

        dx = (um / speed_m) * step_length
        dy = (vm / speed_m) * step_length

        x += dx
        y += dy
        coords.append((x, y))

        traveled += np.sqrt(dx**2 + dy**2)
        if traveled >= length:
            break

        if len(coords) > progress_window_steps:
            x0, y0 = coords[-(progress_window_steps + 1)]
            expected_progress = progress_window_steps * step
            if np.hypot(x - x0, y - y0) < min_progress_fraction * expected_progress:
                break
    return coords


def _resample_coordinates(coords, spacing):
    """Return coordinates at regular arc intervals, including both endpoints."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords.tolist()
    segment_lengths = np.hypot(*np.diff(coords, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = arc[-1]
    if total == 0:
        return coords[:1].tolist()
    distances = np.arange(0.0, total, spacing)
    if distances.size == 0 or not np.isclose(distances[-1], total):
        distances = np.append(distances, total)
    sampled = np.column_stack(
        [np.interp(distances, arc, coords[:, axis]) for axis in range(2)]
    )
    return sampled.tolist()


def reversal_fraction(flowline):
    """Fraction of consecutive turns greater than 90 degrees."""
    coordinates = np.asarray(flowline.coords, dtype=float)
    steps = np.diff(coordinates, axis=0)
    if len(steps) < 2:
        return 0.0
    nonzero = np.hypot(steps[:, 0], steps[:, 1]) > 0
    adjacent = nonzero[:-1] & nonzero[1:]
    if not np.any(adjacent):
        return 0.0
    dots = np.einsum("ij,ij->i", steps[:-1], steps[1:])
    return float(np.mean(dots[adjacent] < 0))


def flowline_passes_reversal_check(flowline, max_reversal_fraction=0.0):
    """Whether a non-empty flow-bend line satisfies the reversal limit."""
    return (
        flowline is not None
        and not flowline.is_empty
        and reversal_fraction(flowline) <= max_reversal_fraction
    )


def get_bidirectional_flowline_vector(
    start_point,
    ice_velocity,
    transform,
    lengths=[10000, 10000],
    step=2,
    integration_step=None,
    progress_window_steps=50,
    min_progress_fraction=0.5,
):
    if integration_step is None:
        integration_step = step
    u_interp, v_interp, x_coords, y_coords = build_interpolators(ice_velocity[0], -1 * ice_velocity[1], transform)
    landward_flow = compute_flowline(u_raster = u_interp, v_raster = v_interp, start_x = start_point.geometry.x.values[0], start_y = start_point.geometry.y.values[0],
                                     length = lengths[0], x_coords = x_coords, y_coords = y_coords, min_speed = 1e-6,
                                     step = integration_step, direction = 'backward', progress_window_steps = progress_window_steps,
                                     min_progress_fraction = min_progress_fraction)
    
    seaward_flow = compute_flowline(u_raster = u_interp, v_raster = v_interp, start_x = start_point.geometry.x.values[0], start_y = start_point.geometry.y.values[0],
                                     length = lengths[1], x_coords = x_coords, y_coords = y_coords, min_speed = 1e-6,
                                     step = integration_step, direction = 'forward', progress_window_steps = progress_window_steps,
                                     min_progress_fraction = min_progress_fraction)
    landward_flow = [coords for coords in landward_flow if np.isfinite(coords).all()]
    seaward_flow = [coords for coords in seaward_flow if np.isfinite(coords).all()]
    landward_flow = _resample_coordinates(landward_flow, step)
    seaward_flow = _resample_coordinates(seaward_flow, step)
    try:
        flowline = LineString(landward_flow[::-1] + seaward_flow[1:])

    except:
        flowline = LineString()
    return flowline

def get_flowline(
    start,
    ice_velocity,
    transform,
    lengths=[2e04, 2e04],
    step=2,
    integration_step=None,
    progress_window_steps=50,
    min_progress_fraction=0.5,
):
    flowline = get_bidirectional_flowline_vector(
        start_point=start,
        ice_velocity=ice_velocity,
        transform=transform,
        lengths=lengths,
        step=step,
        integration_step=integration_step,
        progress_window_steps=progress_window_steps,
        min_progress_fraction=min_progress_fraction,
    )
    return flowline
