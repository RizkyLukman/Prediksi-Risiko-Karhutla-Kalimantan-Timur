from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from karhutla.grid import (
    GRID_COLUMNS,
    assign_dominant_adm2,
    build_grid,
    cell_id_from_indices,
    clip_grid_to_region,
    make_grid,
)

RES = 0.05


def _province() -> gpd.GeoDataFrame:
    # Two regencies side by side; province is their union.
    return gpd.GeoDataFrame(
        {"adm1_name": ["P"]}, geometry=[box(116.0, -1.0, 116.2, -0.9)], crs="EPSG:4326"
    )


def _adm2() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"adm2_name": ["West", "East"], "adm2_pcode": ["W1", "E1"], "adm1_name": ["P", "P"]},
        geometry=[box(116.0, -1.0, 116.07, -0.9), box(116.07, -1.0, 116.2, -0.9)],
        crs="EPSG:4326",
    )


# --- make_grid ---------------------------------------------------------------


def test_make_grid_aligns_cells_to_resolution_multiples() -> None:
    grid = make_grid((116.02, -1.03, 116.11, -0.96), RES)
    assert len(grid) == 6  # lon 116.00/116.05/116.10 x lat -1.05/-1.00
    assert np.allclose(grid["lon_min"] / RES, np.round(grid["lon_min"] / RES))
    assert np.allclose(grid["lat_min"] / RES, np.round(grid["lat_min"] / RES))
    assert set(np.round(grid["lon_min"], 6)) == {116.0, 116.05, 116.1}
    assert set(np.round(grid["lat_min"], 6)) == {-1.05, -1.0}


def test_make_grid_cells_are_squares_of_resolution() -> None:
    grid = make_grid((116.0, -1.0, 116.1, -0.9), RES)
    for geom in grid.geometry:
        minx, miny, maxx, maxy = geom.bounds
        assert maxx - minx == pytest.approx(RES)
        assert maxy - miny == pytest.approx(RES)
    assert grid.crs.to_epsg() == 4326


def test_make_grid_centroids_match_geometry() -> None:
    grid = make_grid((116.0, -1.0, 116.1, -0.9), RES)
    b = grid.geometry.bounds
    assert np.allclose(grid["lon_c"], (b["minx"] + b["maxx"]) / 2)
    assert np.allclose(grid["lat_c"], (b["miny"] + b["maxy"]) / 2)


# --- cell_id -----------------------------------------------------------------


def test_cell_ids_are_unique_and_deterministic() -> None:
    a = make_grid((113.8, -2.5, 119.0, 2.7), RES)
    b = make_grid((113.8, -2.5, 119.0, 2.7), RES)
    assert a["cell_id"].is_unique
    assert a["cell_id"].tolist() == b["cell_id"].tolist()
    assert a["cell_id"].dtype == np.int64


def test_cell_id_depends_only_on_position_not_on_bounds() -> None:
    wide = make_grid((115.0, -2.0, 118.0, 1.0), RES).set_index(["lon_min", "lat_min"])
    narrow = make_grid((116.0, -1.0, 116.1, -0.9), RES).set_index(["lon_min", "lat_min"])
    common = narrow.index.intersection(wide.index)
    assert len(common) == len(narrow)
    assert (wide.loc[common, "cell_id"] == narrow.loc[common, "cell_id"]).all()


def test_cell_id_formula_is_documented_and_invertible() -> None:
    grid = make_grid((116.0, -1.0, 116.1, -0.9), RES)
    expected = cell_id_from_indices(grid["col"].to_numpy(), grid["row"].to_numpy())
    assert (grid["cell_id"] == expected).all()
    assert np.allclose(grid["col"] * RES - 180, grid["lon_min"])
    assert np.allclose(grid["row"] * RES - 90, grid["lat_min"])


# --- clip_grid_to_region -----------------------------------------------------


def test_clip_keeps_intersecting_cells_with_area_fraction() -> None:
    grid = make_grid((115.9, -1.1, 116.3, -0.8), RES)
    clipped = clip_grid_to_region(grid, _province().geometry.iloc[0], min_fraction=0.0)
    # province spans lon 116.0-116.2 (4 cols) x lat -1.0 to -0.9 (2 rows) = 8 full cells
    assert len(clipped) == 8
    assert np.allclose(clipped["frac_in_region"], 1.0, atol=1e-6)


def test_clip_reports_partial_fraction_and_drops_outside() -> None:
    region = box(116.0, -1.0, 116.075, -0.9)  # 1.5 columns wide
    grid = make_grid((115.9, -1.1, 116.3, -0.8), RES)
    clipped = clip_grid_to_region(grid, region, min_fraction=0.0)
    fracs = sorted(np.round(clipped["frac_in_region"], 3).unique())
    assert fracs == [0.5, 1.0]
    assert len(clipped) == 4  # 2 rows x (1 full + 1 half)


def test_clip_min_fraction_filters_slivers() -> None:
    region = box(116.0, -1.0, 116.075, -0.9)
    grid = make_grid((115.9, -1.1, 116.3, -0.8), RES)
    clipped = clip_grid_to_region(grid, region, min_fraction=0.6)
    assert len(clipped) == 2
    assert (clipped["frac_in_region"] > 0.6).all()


# --- assign_dominant_adm2 ----------------------------------------------------


def test_dominant_adm2_is_the_one_with_largest_overlap() -> None:
    grid = make_grid((116.0, -1.0, 116.2, -0.9), RES)
    out = assign_dominant_adm2(grid, _adm2())
    by_lon = out.set_index(np.round(out["lon_min"], 6))
    assert by_lon.loc[116.0, "adm2_name"].unique().tolist() == ["West"]
    # cell 116.05-116.10 overlaps West by 0.02 and East by 0.03 -> East
    assert by_lon.loc[116.05, "adm2_name"].unique().tolist() == ["East"]
    assert by_lon.loc[116.1, "adm2_name"].unique().tolist() == ["East"]
    assert by_lon.loc[116.15, "adm2_pcode"].unique().tolist() == ["E1"]


def test_dominant_adm2_keeps_one_row_per_cell() -> None:
    grid = make_grid((116.0, -1.0, 116.2, -0.9), RES)
    out = assign_dominant_adm2(grid, _adm2())
    assert len(out) == len(grid)
    assert out["cell_id"].is_unique
    assert out["adm2_name"].notna().all()


# --- build_grid --------------------------------------------------------------


def test_build_grid_writes_parquet_with_schema(tmp_path: Path) -> None:
    admin_dir = tmp_path / "admin"
    admin_dir.mkdir()
    _province().to_parquet(admin_dir / "adm1.parquet")
    _adm2().to_parquet(admin_dir / "adm2.parquet")
    out = tmp_path / "grid" / "grid.parquet"
    fig = tmp_path / "figures" / "grid.png"

    grid = build_grid(
        admin_dir=admin_dir, out_path=out, figure_path=fig, resolution=RES, min_fraction=0.0
    )

    assert len(grid) == 8
    assert list(grid.columns) == GRID_COLUMNS
    saved = gpd.read_parquet(out)
    assert saved.crs.to_epsg() == 4326
    assert saved["cell_id"].is_unique
    assert fig.is_file()
