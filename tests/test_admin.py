import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from karhutla.ingest.admin import (
    download_file,
    plot_admin,
    select_province,
    standardize_adm2,
    validate_adm2,
)


def _raw_gdf(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "ADM2_EN": ["Paser", "Berau", "Bulungan"],
            "ADM2_PCODE": ["ID6401", "ID6403", "ID6501"],
            "ADM1_EN": ["Kalimantan Timur", "Kalimantan Timur", "Kalimantan Utara"],
        },
        geometry=[box(116, -2, 117, -1), box(117, 1, 118, 2), box(117, 2, 118, 3)],
        crs=crs,
    )


# --- download_file -----------------------------------------------------------


def test_download_file_skips_when_target_exists(tmp_path: Path) -> None:
    target = tmp_path / "raw.bin"
    target.write_bytes(b"old")
    calls: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        calls.append(url)

    downloaded = download_file("https://example.org/x", target, fetch=fetch)

    assert downloaded is False
    assert calls == []
    assert target.read_bytes() == b"old"


def test_download_file_fetches_and_writes_metadata(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "raw.bin"

    def fetch(url: str, dest: Path) -> None:
        dest.write_bytes(b"payload")

    downloaded = download_file(
        "https://example.org/x", target, fetch=fetch, source="Example", params={"layer": "a"}
    )

    assert downloaded is True
    assert target.read_bytes() == b"payload"
    meta = json.loads((tmp_path / "sub" / "raw.bin.meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "Example"
    assert meta["url"] == "https://example.org/x"
    assert meta["params"] == {"layer": "a"}
    assert meta["sha256"]
    assert meta["downloaded_at"]


def test_download_file_cleans_up_partial_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "raw.bin"

    def fetch(url: str, dest: Path) -> None:
        dest.write_bytes(b"half")
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        download_file("https://example.org/x", target, fetch=fetch)
    assert not target.exists()


# --- select_province ---------------------------------------------------------


def test_select_province_keeps_only_matching_rows() -> None:
    out = select_province(_raw_gdf(), province_col="ADM1_EN", province="Kalimantan Timur")
    assert list(out["ADM2_EN"]) == ["Paser", "Berau"]


def test_select_province_raises_when_no_rows_match() -> None:
    with pytest.raises(ValueError, match="Kalimantan Tengah"):
        select_province(_raw_gdf(), province_col="ADM1_EN", province="Kalimantan Tengah")


# --- standardize_adm2 --------------------------------------------------------


def test_standardize_adm2_renames_columns_and_keeps_wgs84() -> None:
    out = standardize_adm2(
        _raw_gdf(), name_col="ADM2_EN", pcode_col="ADM2_PCODE", province_col="ADM1_EN"
    )
    assert list(out.columns) == ["adm2_name", "adm2_pcode", "adm1_name", "geometry"]
    assert out.crs.to_epsg() == 4326


def test_standardize_adm2_reprojects_to_wgs84() -> None:
    raw = _raw_gdf().to_crs("EPSG:3857")
    out = standardize_adm2(raw, name_col="ADM2_EN", pcode_col="ADM2_PCODE", province_col="ADM1_EN")
    assert out.crs.to_epsg() == 4326
    assert out.geometry.iloc[0].bounds == pytest.approx((116, -2, 117, -1), abs=1e-6)


def test_standardize_adm2_rejects_missing_crs() -> None:
    raw = _raw_gdf().set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="CRS"):
        standardize_adm2(raw, name_col="ADM2_EN", pcode_col="ADM2_PCODE", province_col="ADM1_EN")


# --- validate_adm2 -----------------------------------------------------------


def test_validate_adm2_passes_on_expected_count() -> None:
    gdf = _raw_gdf().iloc[:2]
    validate_adm2(gdf, expected_count=2, name_col="ADM2_EN")


def test_validate_adm2_raises_on_wrong_count_listing_names() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_adm2(_raw_gdf(), expected_count=10, name_col="ADM2_EN")
    msg = str(exc_info.value)
    assert "expected 10" in msg
    assert "got 3" in msg
    assert "Bulungan" in msg


def test_validate_adm2_raises_on_duplicate_names() -> None:
    gdf = _raw_gdf()
    gdf.loc[2, "ADM2_EN"] = "Paser"
    with pytest.raises(ValueError, match="duplicate"):
        validate_adm2(gdf, expected_count=3, name_col="ADM2_EN")


# --- plot_admin --------------------------------------------------------------


def test_plot_admin_writes_png(tmp_path: Path) -> None:
    gdf = standardize_adm2(
        _raw_gdf().iloc[:2], name_col="ADM2_EN", pcode_col="ADM2_PCODE", province_col="ADM1_EN"
    )
    out = tmp_path / "figs" / "admin.png"
    plot_admin(gdf, out)
    assert out.is_file()
    assert out.stat().st_size > 1000


# --- build_admin -------------------------------------------------------------


def test_build_admin_writes_parquet_and_png(tmp_path: Path) -> None:
    from karhutla.ingest.admin import AdminSource, build_admin

    src = AdminSource(
        name="synthetic",
        url="https://example.org/adm.zip",
        raw_filename="adm.zip",
        layer="adm2",
        province_col="ADM1_EN",
        name_col="ADM2_EN",
        pcode_col="ADM2_PCODE",
        expected_adm2_count=2,
    )
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    fig = tmp_path / "figures" / "admin.png"

    def fetch(url: str, dest: Path) -> None:
        dest.write_bytes(b"zip")

    def read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
        assert path == raw_dir / "adm.zip"
        assert layer == "adm2"
        return _raw_gdf()

    adm2 = build_admin(
        src,
        province="Kalimantan Timur",
        raw_dir=raw_dir,
        out_dir=out_dir,
        figure_path=fig,
        fetch=fetch,
        read_layer=read_layer,
    )

    assert len(adm2) == 2
    assert list(adm2.columns) == ["adm2_name", "adm2_pcode", "adm1_name", "geometry"]
    adm2_file = gpd.read_parquet(out_dir / "adm2.parquet")
    assert adm2_file.crs.to_epsg() == 4326
    assert sorted(adm2_file["adm2_name"]) == ["Berau", "Paser"]
    adm1_file = gpd.read_parquet(out_dir / "adm1.parquet")
    assert len(adm1_file) == 1
    assert adm1_file["adm1_name"].iloc[0] == "Kalimantan Timur"
    assert adm1_file.geometry.iloc[0].area == pytest.approx(2.0)
    assert fig.is_file()
    assert (raw_dir / "adm.zip.meta.json").is_file()
