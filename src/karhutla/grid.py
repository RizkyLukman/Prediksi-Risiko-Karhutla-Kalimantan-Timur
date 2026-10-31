"""Regular 0.05-degree analysis grid clipped to the province.

Cells are axis-aligned squares whose lower-left corners sit on multiples of the
resolution, so the same ``cell_id`` always refers to the same place regardless
of the bounding box used to build the grid.

``cell_id`` is an int64 derived from global integer column/row indices:
``col = round((lon_min + 180) / res)``, ``row = round((lat_min + 90) / res)``,
``cell_id = col * 100_000 + row``. Both indices are stored so the id is invertible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
EQUAL_AREA = "EPSG:6933"  # NSIDC EASE-Grid 2.0 Global, used only for area ratios
ROW_MULTIPLIER = 100_000

GRID_COLUMNS = [
    "cell_id",
    "col",
    "row",
    "lon_min",
    "lat_min",
    "lon_c",
    "lat_c",
    "frac_in_region",
    "adm2_name",
    "adm2_pcode",
    "geometry",
]


def cell_id_from_indices(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Combine global column/row indices into one int64 id."""
    return col.astype(np.int64) * ROW_MULTIPLIER + row.astype(np.int64)


def make_grid(bounds: tuple[float, float, float, float], resolution: float) -> gpd.GeoDataFrame:
    """Build all cells whose squares intersect ``bounds`` (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    col0 = int(np.floor((minx + 180) / resolution + 1e-9))
    col1 = int(np.ceil((maxx + 180) / resolution - 1e-9))
    row0 = int(np.floor((miny + 90) / resolution + 1e-9))
    row1 = int(np.ceil((maxy + 90) / resolution - 1e-9))

    cols, rows = np.meshgrid(np.arange(col0, col1), np.arange(row0, row1), indexing="ij")
    cols = cols.ravel()
    rows = rows.ravel()
    lon_min = np.round(cols * resolution - 180, 10)
    lat_min = np.round(rows * resolution - 90, 10)

    frame = pd.DataFrame(
        {
            "cell_id": cell_id_from_indices(cols, rows),
            "col": cols.astype(np.int64),
            "row": rows.astype(np.int64),
            "lon_min": lon_min,
            "lat_min": lat_min,
            "lon_c": lon_min + resolution / 2,
            "lat_c": lat_min + resolution / 2,
        }
    )
    geoms = [
        box(x, y, x + resolution, y + resolution)
        for x, y in zip(frame["lon_min"], frame["lat_min"], strict=True)
    ]
    return gpd.GeoDataFrame(frame, geometry=geoms, crs=WGS84)


def clip_grid_to_region(grid: gpd.GeoDataFrame, region, *, min_fraction: float) -> gpd.GeoDataFrame:
    """Keep cells intersecting ``region`` and add ``frac_in_region`` (equal-area ratio).

    Cell geometries are kept as full squares; the fraction records coverage.
    """
    hits = grid[grid.intersects(region)].copy()
    if hits.empty:
        raise ValueError("no grid cells intersect the region")
    inter = hits.geometry.intersection(region)
    cell_area = hits.geometry.to_crs(EQUAL_AREA).area
    inter_area = gpd.GeoSeries(inter, crs=WGS84).to_crs(EQUAL_AREA).area
    hits["frac_in_region"] = (inter_area / cell_area).clip(0.0, 1.0).to_numpy()
    kept = hits[hits["frac_in_region"] > min_fraction]
    return kept.reset_index(drop=True)


def assign_dominant_adm2(grid: gpd.GeoDataFrame, adm2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach the regency/city with the largest overlap area to each cell."""
    pieces = gpd.overlay(
        grid[["cell_id", "geometry"]],
        adm2[["adm2_name", "adm2_pcode", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    pieces["_area"] = pieces.geometry.to_crs(EQUAL_AREA).area
    best = pieces.sort_values(
        ["cell_id", "_area", "adm2_pcode"], ascending=[True, False, True]
    ).drop_duplicates("cell_id")[["cell_id", "adm2_name", "adm2_pcode"]]
    out = grid.merge(best, on="cell_id", how="left")
    missing = out["adm2_name"].isna().sum()
    if missing:
        raise ValueError(f"{missing} cells have no overlapping ADM2 polygon")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs)


def plot_grid(grid: gpd.GeoDataFrame, adm2: gpd.GeoDataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 9))
    grid.plot(
        ax=ax,
        column="frac_in_region",
        cmap="YlGn",
        edgecolor="grey",
        linewidth=0.1,
        legend=True,
        legend_kwds={"label": "fraction inside province", "shrink": 0.6},
    )
    adm2.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title(f"{len(grid)} grid cells")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_grid(
    *,
    admin_dir: Path,
    out_path: Path,
    figure_path: Path,
    resolution: float,
    min_fraction: float,
) -> gpd.GeoDataFrame:
    """Build, clip, label, save, and plot the grid from the admin layers."""
    adm1 = gpd.read_parquet(admin_dir / "adm1.parquet")
    adm2 = gpd.read_parquet(admin_dir / "adm2.parquet")
    region = adm1.geometry.union_all()

    grid = make_grid(tuple(adm1.total_bounds), resolution)
    grid = clip_grid_to_region(grid, region, min_fraction=min_fraction)
    grid = assign_dominant_adm2(grid, adm2)
    grid = grid.sort_values("cell_id", ignore_index=True)[GRID_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(out_path)
    plot_grid(grid, adm2, figure_path)
    log.info("wrote %d cells to %s", len(grid), out_path)
    return grid


def main() -> None:
    import argparse

    from karhutla.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser(description="Build the analysis grid.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    processed = REPO_ROOT / cfg.paths.data_processed
    grid = build_grid(
        admin_dir=processed / "admin",
        out_path=processed / "grid" / "grid.parquet",
        figure_path=REPO_ROOT / cfg.paths.reports / "figures" / "grid.png",
        resolution=cfg.grid.resolution_deg,
        min_fraction=cfg.grid.min_area_fraction,
    )
    print(f"{len(grid)} cells at {cfg.grid.resolution_deg} deg")
    print(grid.groupby("adm2_name").size().rename("cells").to_string())
    full = (grid["frac_in_region"] >= 0.999).sum()
    print(f"fully inside province: {full}; partial boundary cells: {len(grid) - full}")


if __name__ == "__main__":
    main()
