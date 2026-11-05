import geopandas as gpd
import numpy as np
from shapely import LineString

from gl_from_range_offsets.postprocessing.calibration_truth import _reference_arcs


def test_reference_arcs_uses_mapped_reference_line_id():
    flow_bends = gpd.GeoDataFrame(
        {"Id": [7], "ref_line_id": [20]},
        geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])],
        crs="EPSG:3031",
    )
    reference_lines = gpd.GeoDataFrame(
        {"Id": [10, 20]},
        geometry=[
            LineString([(2.0, -1.0), (2.0, 1.0)]),
            LineString([(8.0, -1.0), (8.0, 1.0)]),
        ],
        crs=flow_bends.crs,
    )

    arcs = _reference_arcs(flow_bends, reference_lines)

    assert np.allclose(arcs[7], [8.0])
