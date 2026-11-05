import numpy as np
import geopandas as gpd
from affine import Affine
from shapely import LineString, Point

from gl_from_range_offsets.data_preparation import flowlines


def _grid():
    coordinates = np.arange(-1_000.0, 1_001.0, 100.0)
    return coordinates, coordinates


def test_progress_guard_stops_an_oscillating_trace():
    x_coordinates, y_coordinates = _grid()
    u = np.tile(-x_coordinates, (len(y_coordinates), 1))
    v = np.zeros_like(u)

    coordinates = flowlines.compute_flowline(
        u,
        v,
        x_coordinates,
        y_coordinates,
        start_x=50.0,
        start_y=0.0,
        length=2_000.0,
        step=200.0,
        progress_window_steps=4,
        min_progress_fraction=0.5,
    )

    assert len(coordinates) == 5
    assert LineString(coordinates).length == 800.0


def test_coarse_trace_is_resampled_and_keeps_reference_anchor():
    field = np.zeros((2, 21, 21), dtype=float)
    field[0] = 1.0
    start = gpd.GeoDataFrame(geometry=[Point(0.0, 0.0)], crs="EPSG:3031")

    line = flowlines.get_flowline(
        start,
        field,
        Affine.translation(-1_000.0, 1_000.0) * Affine.scale(100.0, -100.0),
        lengths=[1_000.0, 1_000.0],
        step=2.0,
        integration_step=200.0,
        progress_window_steps=4,
        min_progress_fraction=0.5,
    )

    coordinates = np.asarray(line.coords)
    assert len(coordinates) == 1_001
    assert np.allclose(coordinates[500], [0.0, 0.0])
    assert np.isclose(line.length, 2_000.0)


def test_reversal_quality_rejects_a_backtracking_line():
    good = LineString([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    bad = LineString([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)])

    assert flowlines.reversal_fraction(good) == 0.0
    assert flowlines.reversal_fraction(bad) == 1.0
    assert flowlines.flowline_passes_reversal_check(good, 0.0)
    assert not flowlines.flowline_passes_reversal_check(bad, 0.0)
