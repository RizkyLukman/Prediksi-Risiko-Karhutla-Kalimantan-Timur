"""NASA FIRMS VIIRS S-NPP 375 m hotspot archive for the province bounding box.

The FIRMS area API (``/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/{days}/{date}``)
serves at most 5 days per call; the standard-processing source
``VIIRS_SNPP_SP`` covers 20 January 2012 onward. Each call is cached as one CSV
under ``data/raw/firms/<source>/<year>/`` so re-runs only fetch missing chunks.
A manifest records every chunk without ever storing the MAP_KEY.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

Fetcher = Callable[[str], str]
BBox = tuple[float, float, float, float]

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
]
OPTIONAL_COLUMNS = ["type"]  # standard-processing data only
CONFIDENCE_CODES = {"l", "n", "h"}  # verified against a live VIIRS_SNPP_SP response, 2026-11-01
DEDUP_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
STRING_COLUMNS = ["confidence", "satellite", "instrument", "version", "daynight", "acq_date"]


class FirmsApiError(RuntimeError):
    """The API answered with something other than a hotspot CSV."""


# --- request building --------------------------------------------------------


def iter_chunks(start: date, end: date, *, max_days: int) -> list[tuple[date, int]]:
    """Split [start, end] into consecutive (start_date, n_days) chunks."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if max_days < 1:
        raise ValueError("max_days must be >= 1")
    chunks: list[tuple[date, int]] = []
    cur = start
    while cur <= end:
        n = min(max_days, (end - cur).days + 1)
        chunks.append((cur, n))
        cur += timedelta(days=n)
    return chunks


def build_area_url(
    base: str, *, map_key: str, source: str, bbox: BBox, day_range: int, start: date
) -> str:
    area = ",".join(str(v) for v in bbox)
    return f"{base.rstrip('/')}/{map_key}/{source}/{area}/{day_range}/{start:%Y-%m-%d}"


_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """One keep-alive session for the whole run; avoids a new TLS handshake per chunk."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = "karhutla-kaltim/0.1 (research; hotspot archive)"
    return _session


def _http_fetch(url: str, *, timeout: int = 120, retries: int = 6) -> str:
    """GET with exponential backoff. Error text never includes the URL (it holds the key)."""
    for attempt in range(1, retries + 1):
        try:
            resp = _get_session().get(url, timeout=(30, timeout))
            if resp.status_code >= 500:
                raise requests.HTTPError(f"server error {resp.status_code}")
            resp.raise_for_status()
            return resp.text
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            reason = type(exc).__name__
            if attempt == retries:
                raise RuntimeError(f"giving up after {retries} attempts ({reason})") from None
            wait = 10 * 2 ** (attempt - 1)
            log.warning("fetch attempt %d failed (%s); retrying in %ds", attempt, reason, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def redact(text: str, secret: str) -> str:
    """Replace every occurrence of ``secret`` in ``text``."""
    return text.replace(secret, "***") if secret else text


# --- availability ------------------------------------------------------------


def parse_availability(text: str, source: str) -> tuple[date, date]:
    """Parse ``/api/data_availability/csv`` output (data_id,min_date,max_date)."""
    df = pd.read_csv(io.StringIO(text), dtype="string")
    for col in ("data_id", "min_date", "max_date"):
        if col not in df.columns:
            raise FirmsApiError(f"availability response lacks column {col!r}: {text[:200]!r}")
    row = df[df["data_id"] == source]
    if row.empty:
        raise FirmsApiError(f"source {source!r} not in availability response: {text[:200]!r}")
    return (
        date.fromisoformat(str(row["min_date"].iloc[0])),
        date.fromisoformat(str(row["max_date"].iloc[0])),
    )


def fetch_availability(
    base: str, *, map_key: str, source: str, fetch: Fetcher | None = None
) -> tuple[date, date]:
    """Return (min_date, max_date) the API currently serves for ``source``."""
    fetcher = fetch or _http_fetch
    url = f"{base.rstrip('/')}/{map_key}/{source}"
    try:
        text = fetcher(url)
    except Exception as exc:
        raise RuntimeError(f"availability check failed: {redact(str(exc), map_key)}") from None
    return parse_availability(text, source)


# --- parsing -----------------------------------------------------------------


def parse_csv(text: str) -> pd.DataFrame:
    """Parse an area-API CSV body; raise FirmsApiError on error messages."""
    head = text.lstrip().split("\n", 1)[0].strip().lower()
    if not head.startswith("latitude,longitude"):
        raise FirmsApiError(f"unexpected API response: {text.strip()[:200]!r}")
    df = pd.read_csv(io.StringIO(text), dtype=dict.fromkeys(STRING_COLUMNS, "string"))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FirmsApiError(f"response lacks required columns: {missing}")
    df["acq_time"] = df["acq_time"].astype("int64")
    if "type" in df.columns:
        df["type"] = df["type"].astype("Int64")
    return df


# --- download ----------------------------------------------------------------


def _chunk_path(out_dir: Path, source: str, start: date, n_days: int) -> Path:
    return out_dir / f"{start:%Y}" / f"{source}_{start:%Y-%m-%d}_{n_days}d.csv"


def download_archive(
    *,
    map_key: str,
    source: str,
    bbox: BBox,
    start: date,
    end: date,
    out_dir: Path,
    max_days: int,
    api_base: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
    fetch: Fetcher | None = None,
    sleep_seconds: float = 0.2,
    available_until: date | None = None,
) -> int:
    """Fetch every missing chunk in [start, end]. Returns the number written.

    Chunks are always aligned to ``start`` in ``max_days`` steps, so the same
    dates map to the same file names on every run. A chunk whose last day is
    after ``available_until`` is skipped (not cached) so it is fetched once the
    API actually serves those dates.
    """
    fetcher = fetch or _http_fetch
    manifest = out_dir / "manifest.jsonl"
    written = 0
    skipped_unavailable = 0
    chunks = iter_chunks(start, end, max_days=max_days)
    for i, (chunk_start, n_days) in enumerate(chunks, 1):
        chunk_end = chunk_start + timedelta(days=n_days - 1)
        if available_until is not None and chunk_end > available_until:
            skipped_unavailable += 1
            continue
        path = _chunk_path(out_dir, source, chunk_start, n_days)
        if path.exists():
            continue
        url = build_area_url(
            api_base,
            map_key=map_key,
            source=source,
            bbox=bbox,
            day_range=n_days,
            start=chunk_start,
        )
        try:
            text = fetcher(url)
        except Exception as exc:
            raise RuntimeError(
                f"chunk {chunk_start} ({n_days}d) failed: {redact(str(exc), map_key)}"
            ) from None
        df = parse_csv(text)  # validates before anything touches disk
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        record = {
            "source": source,
            "start": chunk_start.isoformat(),
            "days": n_days,
            "bbox": list(bbox),
            "file": path.relative_to(out_dir).as_posix(),
            "rows": int(len(df)),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        written += 1
        if written % 50 == 0:
            log.info("chunk %d/%d (%s): %d rows", i, len(chunks), chunk_start, len(df))
        if sleep_seconds:
            time.sleep(sleep_seconds)
    log.info(
        "%d new chunks written to %s (%d total, %d not yet available)",
        written,
        out_dir,
        len(chunks),
        skipped_unavailable,
    )
    return written


def load_raw(out_dir: Path) -> pd.DataFrame:
    files = sorted(out_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no chunk CSVs under {out_dir}")
    frames = [parse_csv(p.read_text(encoding="utf-8")) for p in files]
    return pd.concat(frames, ignore_index=True)


# --- cleaning ----------------------------------------------------------------


def clean_hotspots(df: pd.DataFrame, *, bbox: BBox, tz: str) -> pd.DataFrame:
    """Type, filter to bbox, de-duplicate, and add UTC/local time and ISO week keys."""
    bad = set(df["confidence"].dropna().unique()) - CONFIDENCE_CODES
    if bad:
        raise ValueError(
            f"unknown confidence codes {sorted(bad)}; expected {sorted(CONFIDENCE_CODES)}"
        )

    minx, miny, maxx, maxy = bbox
    inside = df["longitude"].between(minx, maxx) & df["latitude"].between(miny, maxy)
    out = df[inside].drop_duplicates(subset=DEDUP_COLUMNS).copy()

    hhmm = out["acq_time"].astype(int).map("{:04d}".format)
    utc = pd.to_datetime(out["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)
    local = utc.dt.tz_convert(tz)
    out["acq_datetime_utc"] = utc
    out["acq_date_local"] = local.dt.date
    out["week_start"] = (local - pd.to_timedelta(local.dt.day_of_week, unit="D")).dt.date
    out["year"] = local.dt.year.astype("int16")
    out["month"] = local.dt.month.astype("int8")
    return out.sort_values(["acq_datetime_utc", "latitude", "longitude"], ignore_index=True)


# --- summaries ---------------------------------------------------------------


def _pivot(df: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    table = df.pivot_table(
        index=index, columns="confidence", values="latitude", aggfunc="size", fill_value=0
    )
    table["total"] = table.sum(axis=1)
    table.columns.name = None
    return table.reset_index()


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts per year and per (year, month), split by confidence."""
    return _pivot(df, ["year"]), _pivot(df, ["year", "month"])


def plot_summary(by_year: pd.DataFrame, by_month: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [c for c in ["n", "h"] if c in by_year.columns]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    by_year.set_index("year")[keep].plot.bar(ax=ax1, stacked=True)
    ax1.set_title("VIIRS S-NPP hotspots per year (nominal + high confidence)")
    ax1.set_ylabel("hotspots")
    by_month.groupby("month")[keep].sum().sum(axis=1).plot.bar(ax=ax2)
    ax2.set_title("Hotspots per calendar month, all years (nominal + high)")
    ax2.set_ylabel("hotspots")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    import argparse
    import os

    import geopandas as gpd

    from karhutla.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser(description="Download and clean the FIRMS archive.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--end", type=date.fromisoformat, default=None, help="last date to fetch (default: today)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    src = cfg.sources.firms
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        raise SystemExit("FIRMS_MAP_KEY is not set; copy .env.example to .env and fill it in")

    adm1 = gpd.read_parquet(REPO_ROOT / cfg.paths.data_processed / "admin" / "adm1.parquet")
    pad = cfg.region.bbox_buffer_deg
    minx, miny, maxx, maxy = (float(v) for v in adm1.total_bounds)
    bbox: BBox = (
        round(minx - pad, 4),
        round(miny - pad, 4),
        round(maxx + pad, 4),
        round(maxy + pad, 4),
    )

    raw_dir = REPO_ROOT / cfg.paths.data_raw / "firms" / src.source
    end = args.end or datetime.now(UTC).date()
    avail_min, avail_max = fetch_availability(
        src.availability_base, map_key=map_key, source=src.source
    )
    log.info("%s available %s .. %s", src.source, avail_min, avail_max)
    download_archive(
        map_key=map_key,
        source=src.source,
        bbox=bbox,
        start=cfg.time.start_date,
        end=end,
        out_dir=raw_dir,
        max_days=src.day_range,
        api_base=src.api_base,
        sleep_seconds=src.sleep_seconds,
        available_until=avail_max,
    )

    clean = clean_hotspots(load_raw(raw_dir), bbox=bbox, tz=src.timezone)
    out = REPO_ROOT / cfg.paths.data_processed / "firms" / f"hotspots_{src.source.lower()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out, index=False)

    by_year, by_month = summarize(clean)
    rep = REPO_ROOT / cfg.paths.reports / "firms"
    rep.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(rep / "hotspots_per_year.csv", index=False)
    by_month.to_csv(rep / "hotspots_per_month.csv", index=False)
    plot_summary(by_year, by_month, REPO_ROOT / cfg.paths.reports / "figures" / "firms_summary.png")

    first, last = clean["acq_date_local"].min(), clean["acq_date_local"].max()
    print(f"bbox {bbox}; {len(clean)} hotspots {first} .. {last} -> {out}")
    print(by_year.to_string(index=False))


if __name__ == "__main__":
    main()
