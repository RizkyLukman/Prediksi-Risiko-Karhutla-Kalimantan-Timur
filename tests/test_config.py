from pathlib import Path

import pytest
import yaml

from karhutla.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _valid_dict() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_default_config_loads_locked_decisions() -> None:
    cfg = load_config()
    assert cfg.project == "karhutla-kaltim"
    assert cfg.seed == 42
    assert cfg.region.province == "Kalimantan Timur"
    assert cfg.grid.resolution_deg == 0.05
    assert 0 <= cfg.grid.min_area_fraction < 1
    assert cfg.time.week_convention == "iso"
    assert cfg.time.horizon_weeks == 1
    assert cfg.split.train == (2012, 2021)
    assert cfg.split.val == (2022, 2022)
    assert cfg.split.test == (2023, 2025)
    assert cfg.split.production_sim == (2026, 2026)
    assert cfg.labels.sensor == "VIIRS_SNPP"
    assert cfg.features.rain_windows_days == [7, 30, 90]
    assert cfg.evaluation.top_k_percent == [5, 10]


def test_load_config_accepts_explicit_path(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_dict())
    cfg = load_config(path)
    assert cfg.seed == 42


def test_missing_field_gives_clear_error(tmp_path: Path) -> None:
    data = _valid_dict()
    del data["split"]["val"]
    path = _write(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "split.val" in message
    assert "missing" in message.lower()
    assert str(path) in message


def test_missing_top_level_section_gives_clear_error(tmp_path: Path) -> None:
    data = _valid_dict()
    del data["features"]
    path = _write(tmp_path, data)
    with pytest.raises(ConfigError, match="features"):
        load_config(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    data = _valid_dict()
    data["grid"]["resolution_degree"] = 0.05  # typo of resolution_deg
    path = _write(tmp_path, data)
    with pytest.raises(ConfigError, match="grid.resolution_degree"):
        load_config(path)


def test_overlapping_splits_are_rejected(tmp_path: Path) -> None:
    data = _valid_dict()
    data["split"]["val"] = [2021, 2022]  # overlaps train end 2021
    path = _write(tmp_path, data)
    with pytest.raises(ConfigError, match="overlap"):
        load_config(path)


def test_missing_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_admin_source_section_is_validated() -> None:
    cfg = load_config()
    admin = cfg.sources.admin
    assert admin.expected_adm2_count == 10
    assert admin.url.startswith("https://")
    for field in ("raw_filename", "layer", "province_col", "name_col", "pcode_col"):
        assert getattr(admin, field), f"sources.admin.{field} must not be empty"


def test_missing_admin_source_field_gives_clear_error(tmp_path: Path) -> None:
    data = _valid_dict()
    del data["sources"]["admin"]["layer"]
    path = _write(tmp_path, data)
    with pytest.raises(ConfigError, match="sources.admin.layer"):
        load_config(path)


def test_firms_source_section_is_validated() -> None:
    cfg = load_config()
    firms = cfg.sources.firms
    assert firms.source == "VIIRS_SNPP_SP"
    assert 1 <= firms.day_range <= 5
    assert firms.api_base.startswith("https://")
    assert firms.timezone == "Asia/Makassar"
    assert firms.nrt_source == "VIIRS_SNPP_NRT"
    assert firms.nrt_refetch_days >= 0
