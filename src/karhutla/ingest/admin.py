"""Administrative boundaries of East Kalimantan (province and 10 regencies/cities).

Source selection and column names are driven by ``configs/config.yaml``
(``sources.admin``). Raw downloads are cached under ``data/raw/admin`` and are
never modified; the standardized layers are written to ``data/processed/admin``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import requests

from karhutla.config import AdminSourceConfig

log = logging.getLogger(__name__)

Fetcher = Callable[[str, Path], None]
LayerReader = Callable[[Path, str], gpd.GeoDataFrame]


AdminSource = AdminSourceConfig

STANDARD_COLUMNS = ["adm2_name", "adm2_pcode", "adm1_name", "geometry"]
WGS84 = "EPSG:4326"


def _http_fetch(url: str, dest: Path, timeout: int = 600) -> None:
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    target: Path,
    *,
    fetch: Fetcher | None = None,
    source: str = "",
    params: dict | None = None,
) -> bool:
    """Download ``url`` to ``target`` unless it already exists.

    Writes ``<target>.meta.json`` with source, url, params, download time, and
    sha256. A failed download leaves no partial file behind.

    Returns:
        True if the file was downloaded, False if it was already present.
    """
    if target.exists():
        log.info("skip download, exists: %s", target)
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    fetcher = fetch or _http_fetch
    try:
        fetcher(url, target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise

    meta = {
        "source": source,
        "url": url,
        "params": params or {},
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
    }
    target.with_name(target.name + ".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info("downloaded %s -> %s", url, target)
    return True


def select_province(gdf: gpd.GeoDataFrame, *, province_col: str, province: str) -> gpd.GeoDataFrame:
    """Keep only rows whose ``province_col`` equals ``province``."""
    out = gdf[gdf[province_col] == province]
    if out.empty:
        available = sorted(gdf[province_col].dropna().unique().tolist())
        raise ValueError(
            f"no rows where {province_col} == {province!r}; available values: {available}"
        )
    return out.reset_index(drop=True)


def standardize_adm2(
    gdf: gpd.GeoDataFrame, *, name_col: str, pcode_col: str, province_col: str
) -> gpd.GeoDataFrame:
    """Rename source columns to the project schema and ensure WGS84."""
    if gdf.crs is None:
        raise ValueError("input GeoDataFrame has no CRS; refusing to guess")
    out = gdf[[name_col, pcode_col, province_col, gdf.geometry.name]].rename(
        columns={name_col: "adm2_name", pcode_col: "adm2_pcode", province_col: "adm1_name"}
    )
    out = out.set_geometry(gdf.geometry.name)
    if out.geometry.name != "geometry":
        out = out.rename_geometry("geometry")
    if out.crs.to_epsg() != 4326:
        out = out.to_crs(WGS84)
    return gpd.GeoDataFrame(out[STANDARD_COLUMNS], geometry="geometry", crs=WGS84).reset_index(
        drop=True
    )


def validate_adm2(gdf: gpd.GeoDataFrame, *, expected_count: int, name_col: str) -> None:
    """Fail loudly if the regency count or names are not what the config expects."""
    names = gdf[name_col].tolist()
    if len(names) != expected_count:
        raise ValueError(f"expected {expected_count} ADM2 units, got {len(names)}: {sorted(names)}")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate ADM2 names: {dupes}")


def plot_admin(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Save a labelled check map of the ADM2 polygons to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 9))
    gdf.plot(ax=ax, column="adm2_name", cmap="tab20", edgecolor="black", linewidth=0.5)
    for _, row in gdf.iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(row["adm2_name"], (pt.x, pt.y), ha="center", fontsize=7)
    ax.set_title(f"{gdf['adm1_name'].iloc[0]}: {len(gdf)} kabupaten/kota")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _vsi_path(path: Path) -> str:
    """Return a GDAL path; a zipped FileGDB is opened in place via /vsizip/."""
    if path.suffix != ".zip":
        return str(path)
    import zipfile

    with zipfile.ZipFile(path) as zf:
        tops = sorted({n.split("/")[0] for n in zf.namelist()})
    inner = next((t for t in tops if t.endswith(".gdb")), None)
    base = f"/vsizip/{path.as_posix()}"
    return f"{base}/{inner}" if inner else base


def _read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(_vsi_path(path), layer=layer)


def build_admin(
    source: AdminSource,
    *,
    province: str,
    raw_dir: Path,
    out_dir: Path,
    figure_path: Path,
    fetch: Fetcher | None = None,
    read_layer: LayerReader | None = None,
) -> gpd.GeoDataFrame:
    """Download (if needed), filter, standardize, validate, and save the boundaries.

    Writes ``adm2.parquet`` (regencies/cities) and ``adm1.parquet`` (dissolved
    province) as GeoParquet in EPSG:4326 under ``out_dir``, plus a check map.
    """
    raw_path = raw_dir / source.raw_filename
    download_file(
        source.url,
        raw_path,
        fetch=fetch,
        source=source.name,
        params={"layer": source.layer},
    )

    reader = read_layer or _read_layer
    raw = reader(raw_path, source.layer)
    selected = select_province(raw, province_col=source.province_col, province=province)
    validate_adm2(selected, expected_count=source.expected_adm2_count, name_col=source.name_col)
    adm2 = standardize_adm2(
        selected,
        name_col=source.name_col,
        pcode_col=source.pcode_col,
        province_col=source.province_col,
    ).sort_values("adm2_name", ignore_index=True)
    adm1 = adm2.dissolve(by="adm1_name", as_index=False)[["adm1_name", "geometry"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    adm2.to_parquet(out_dir / "adm2.parquet")
    adm1.to_parquet(out_dir / "adm1.parquet")
    plot_admin(adm2, figure_path)
    log.info("wrote %d ADM2 polygons to %s", len(adm2), out_dir)
    return adm2


def main() -> None:
    """CLI: build the admin layers using ``configs/config.yaml``."""
    import argparse

    from karhutla.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser(description="Download and standardize admin boundaries.")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    adm2 = build_admin(
        cfg.sources.admin,
        province=cfg.region.province,
        raw_dir=REPO_ROOT / cfg.paths.data_raw / "admin",
        out_dir=REPO_ROOT / cfg.paths.data_processed / "admin",
        figure_path=REPO_ROOT / cfg.paths.reports / "figures" / "admin.png",
    )
    print(f"{len(adm2)} ADM2 units for {cfg.region.province}:")
    for name, pcode in zip(adm2["adm2_name"], adm2["adm2_pcode"], strict=True):
        print(f"  {pcode}  {name}")


if __name__ == "__main__":
    main()
