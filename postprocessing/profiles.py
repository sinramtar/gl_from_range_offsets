"""Canonical flow-bend profile selection."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely


def canonical_profile_table(flow_bends: gpd.GeoDataFrame, id_column: str = "Id") -> pd.DataFrame:
    """Return one row per flow-bend ID with its canonical physical-profile ID.

    Physical identity is exact geometry identity.  The lowest flow-bend ID is the
    canonical representative of all rows with the same coordinate sequence.
    """
    if id_column not in flow_bends:
        raise ValueError(f"flow-bend table has no {id_column!r} column")
    if flow_bends[id_column].isna().any() or flow_bends[id_column].duplicated().any():
        raise ValueError("flow-bend IDs must be present and unique")
    if flow_bends.geometry.isna().any() or flow_bends.geometry.is_empty.any():
        raise ValueError("flow-bend geometries must be present and non-empty")

    table = pd.DataFrame({
        "flow_bend_id": flow_bends[id_column].astype(int).to_numpy(),
        "geometry_key": shapely.to_wkb(flow_bends.geometry.to_numpy(), hex=True),
    })
    table["canonical_flow_bend_id"] = (
        table.groupby("geometry_key", sort=False)["flow_bend_id"].transform("min").astype(int)
    )
    return table


def canonical_flow_bends(
    flow_bends: gpd.GeoDataFrame, id_column: str = "Id"
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Return canonical flow-bend lines and a small selection audit."""
    invalid_geometry = flow_bends.geometry.isna() | flow_bends.geometry.is_empty
    valid_flow_bends = flow_bends.loc[~invalid_geometry].copy()
    table = canonical_profile_table(valid_flow_bends, id_column=id_column)
    keep = table["flow_bend_id"].eq(table["canonical_flow_bend_id"])
    canonical_ids = set(table.loc[keep, "flow_bend_id"])
    selected = flow_bends.loc[flow_bends[id_column].astype(int).isin(canonical_ids)].copy()
    selected = selected.sort_values(id_column).reset_index(drop=True)
    audit = {
        "input_flow_bends": int(len(flow_bends)),
        "invalid_geometry_flow_bends_removed": int(invalid_geometry.sum()),
        "canonical_flow_bends": int(len(selected)),
        "duplicate_flow_bends_removed": int((~keep).sum()),
        "duplicate_geometry_groups": int(
            table.loc[table["geometry_key"].duplicated(keep=False), "geometry_key"].nunique()
        ),
    }
    return selected, audit
