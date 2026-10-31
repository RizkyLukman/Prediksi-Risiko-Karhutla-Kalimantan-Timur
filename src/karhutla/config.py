"""Typed project configuration loaded from ``configs/config.yaml``.

Every path, year, and tunable parameter lives in the YAML file; code reads it
through :func:`load_config`. Unknown keys are rejected so typos surface early.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

YearRange = tuple[int, int]


class ConfigError(Exception):
    """Raised when the config file is missing, unreadable, or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    data_raw: Path
    data_interim: Path
    data_processed: Path
    reports: Path


class RegionConfig(_Strict):
    province: str
    bbox_buffer_deg: float = Field(ge=0)


class GridConfig(_Strict):
    resolution_deg: float = Field(gt=0)


class TimeConfig(_Strict):
    start_date: date
    week_convention: Literal["iso"]
    horizon_weeks: int = Field(ge=1)


class SplitConfig(_Strict):
    train: YearRange
    val: YearRange
    test: YearRange
    production_sim: YearRange

    @model_validator(mode="after")
    def _ranges_are_ordered_and_disjoint(self) -> SplitConfig:
        ordered = [
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
            ("production_sim", self.production_sim),
        ]
        for name, (start, end) in ordered:
            if start > end:
                raise ValueError(f"split.{name}: start year {start} is after end year {end}")
        for (prev_name, prev), (name, cur) in zip(ordered, ordered[1:], strict=False):
            if cur[0] <= prev[1]:
                raise ValueError(
                    f"split.{name} {list(cur)} overlaps or precedes split.{prev_name} {list(prev)}"
                )
        return self


class StaticSourceConfig(_Strict):
    cluster_eps_m: float = Field(gt=0)
    min_active_months: int = Field(ge=1)
    min_active_years: int = Field(ge=1)


class LabelsConfig(_Strict):
    sensor: str
    confidence_keep: list[str] = Field(min_length=1)
    static_source: StaticSourceConfig


class FeaturesConfig(_Strict):
    era5_latency_days: int = Field(ge=0)
    rain_windows_days: list[int] = Field(min_length=1)
    hotspot_lags_weeks: list[int] = Field(min_length=1)


class EvaluationConfig(_Strict):
    top_k_percent: list[int] = Field(min_length=1)


class Config(_Strict):
    project: str
    seed: int
    paths: PathsConfig
    region: RegionConfig
    grid: GridConfig
    time: TimeConfig
    split: SplitConfig
    labels: LabelsConfig
    features: FeaturesConfig
    evaluation: EvaluationConfig


def _format_validation_error(path: Path, error: ValidationError) -> str:
    lines = [f"Invalid config file {path}:"]
    for issue in error.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        if issue["type"] == "missing":
            lines.append(f"  - {loc}: missing required field")
        elif issue["type"] == "extra_forbidden":
            lines.append(f"  - {loc}: unknown field (check spelling)")
        else:
            lines.append(f"  - {loc}: {issue['msg']}")
    return "\n".join(lines)


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate the project config.

    Args:
        path: YAML file to read. Defaults to ``configs/config.yaml`` in the repo.

    Raises:
        ConfigError: if the file is missing, is not a YAML mapping, lacks a
            required field, contains an unknown field, or fails a sanity check.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {config_path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {config_path} must contain a top-level mapping")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(config_path, exc)) from exc
