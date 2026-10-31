import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from karhutla.ingest.firms import (
    FirmsApiError,
    build_area_url,
    clean_hotspots,
    download_archive,
    fetch_availability,
    iter_chunks,
    parse_availability,
    parse_csv,
    redact,
)

BBOX = (113.7, -2.6, 119.1, 2.8)  # minx, miny, maxx, maxy

HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
    "confidence,version,bright_ti5,frp,daynight,type"
)
ROW_A = "0.5,116.5,330.1,0.4,0.4,2015-09-13,1730,N,VIIRS,n,1,290.2,3.1,N,0"
ROW_B = "0.6,116.6,345.0,0.4,0.4,2015-09-13,0530,N,VIIRS,h,1,295.0,12.0,D,0"
ROW_LOW = "0.7,116.7,300.0,0.4,0.4,2015-09-13,0530,N,VIIRS,l,1,280.0,1.0,D,0"
ROW_OUT = "5.0,116.5,330.1,0.4,0.4,2015-09-13,1730,N,VIIRS,n,1,290.2,3.1,N,0"  # outside bbox


def _csv(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


# --- iter_chunks -------------------------------------------------------------


def test_iter_chunks_covers_range_without_gaps_or_overlap() -> None:
    chunks = iter_chunks(date(2012, 1, 20), date(2012, 2, 1), max_days=5)
    assert chunks[0] == (date(2012, 1, 20), 5)
    assert chunks[-1][0] + timedelta(days=chunks[-1][1] - 1) == date(2012, 2, 1)
    covered = [d for start, n in chunks for d in pd.date_range(start, periods=n)]
    expected = list(pd.date_range("2012-01-20", "2012-02-01"))
    assert covered == expected


def test_iter_chunks_respects_max_days() -> None:
    assert all(n <= 5 for _, n in iter_chunks(date(2020, 1, 1), date(2020, 12, 31), max_days=5))


def test_iter_chunks_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        iter_chunks(date(2020, 1, 2), date(2020, 1, 1), max_days=5)


# --- build_area_url ----------------------------------------------------------


def test_build_area_url_matches_documented_pattern() -> None:
    url = build_area_url(
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
        map_key="KEY123",
        source="VIIRS_SNPP_SP",
        bbox=BBOX,
        day_range=5,
        start=date(2015, 9, 13),
    )
    assert url == (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/KEY123/VIIRS_SNPP_SP/"
        "113.7,-2.6,119.1,2.8/5/2015-09-13"
    )


# --- parse_csv ---------------------------------------------------------------


def test_parse_csv_returns_typed_frame() -> None:
    df = parse_csv(_csv(ROW_A, ROW_B))
    assert len(df) == 2
    assert df["latitude"].dtype.kind == "f"
    assert df["acq_time"].dtype.kind in "iu"
    assert df["confidence"].tolist() == ["n", "h"]
    assert df["type"].tolist() == [0, 0]


def test_parse_csv_header_only_gives_empty_frame_with_columns() -> None:
    df = parse_csv(_csv())
    assert df.empty
    assert "latitude" in df.columns


def test_parse_csv_rejects_non_csv_api_message() -> None:
    with pytest.raises(FirmsApiError, match="Invalid MAP_KEY"):
        parse_csv("Invalid MAP_KEY.\n")


def test_parse_csv_rejects_missing_required_column() -> None:
    text = "latitude,longitude,acq_date\n0.5,116.5,2015-09-13\n"
    with pytest.raises(FirmsApiError, match="acq_time"):
        parse_csv(text)


# --- download_archive --------------------------------------------------------


def _fake_fetch(responses: dict[str, str], calls: list[str]):
    def fetch(url: str) -> str:
        calls.append(url)
        return responses.get(url.rsplit("/", 1)[-1], _csv())

    return fetch


def test_download_archive_writes_one_file_per_chunk_and_manifest(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({"2015-09-13": _csv(ROW_A, ROW_B)}, calls)
    out_dir = tmp_path / "firms"

    written = download_archive(
        map_key="KEY",
        source="VIIRS_SNPP_SP",
        bbox=BBOX,
        start=date(2015, 9, 13),
        end=date(2015, 9, 22),
        out_dir=out_dir,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
    )

    assert written == 2
    assert len(calls) == 2
    files = sorted(p.name for p in (out_dir / "2015").glob("*.csv"))
    assert files == ["VIIRS_SNPP_SP_2015-09-13_5d.csv", "VIIRS_SNPP_SP_2015-09-18_5d.csv"]
    lines = (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["start"] == "2015-09-13"
    assert rec["rows"] == 2
    assert "KEY" not in json.dumps(rec)  # the map key never lands on disk
    assert rec["sha256"]


def test_download_archive_is_idempotent(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({}, calls)
    kwargs = dict(
        map_key="KEY",
        source="VIIRS_SNPP_SP",
        bbox=BBOX,
        start=date(2015, 9, 13),
        end=date(2015, 9, 17),
        out_dir=tmp_path,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
    )
    assert download_archive(**kwargs) == 1
    assert download_archive(**kwargs) == 0
    assert len(calls) == 1


def test_download_archive_does_not_keep_file_when_api_errors(tmp_path: Path) -> None:
    def fetch(url: str) -> str:
        return "Invalid MAP_KEY.\n"

    with pytest.raises(FirmsApiError):
        download_archive(
            map_key="KEY",
            source="VIIRS_SNPP_SP",
            bbox=BBOX,
            start=date(2015, 9, 13),
            end=date(2015, 9, 13),
            out_dir=tmp_path,
            max_days=5,
            fetch=fetch,
            sleep_seconds=0,
        )
    assert list(tmp_path.rglob("*.csv")) == []


# --- clean_hotspots ----------------------------------------------------------


def test_clean_adds_utc_timestamp_local_date_and_iso_week_start() -> None:
    df = clean_hotspots(parse_csv(_csv(ROW_A)), bbox=BBOX, tz="Asia/Makassar")
    row = df.iloc[0]
    assert row["acq_datetime_utc"] == pd.Timestamp("2015-09-13 17:30", tz="UTC")
    # 17:30 UTC = 01:30 next day in WITA (UTC+8)
    assert row["acq_date_local"] == date(2015, 9, 14)
    assert row["week_start"] == date(2015, 9, 14)  # 2015-09-14 is a Monday


def test_clean_week_start_is_monday_of_local_date() -> None:
    sunday = "0.5,116.5,330.1,0.4,0.4,2015-09-13,0530,N,VIIRS,n,1,290.2,3.1,D,0"
    df = clean_hotspots(parse_csv(_csv(sunday)), bbox=BBOX, tz="Asia/Makassar")
    assert df.iloc[0]["acq_date_local"] == date(2015, 9, 13)
    assert df.iloc[0]["week_start"] == date(2015, 9, 7)


def test_clean_drops_rows_outside_bbox_and_exact_duplicates() -> None:
    df = clean_hotspots(parse_csv(_csv(ROW_A, ROW_A, ROW_OUT)), bbox=BBOX, tz="Asia/Makassar")
    assert len(df) == 1


def test_clean_keeps_all_confidence_levels() -> None:
    df = clean_hotspots(parse_csv(_csv(ROW_A, ROW_B, ROW_LOW)), bbox=BBOX, tz="Asia/Makassar")
    assert sorted(df["confidence"]) == ["h", "l", "n"]


def test_clean_rejects_unknown_confidence_code() -> None:
    bad = "0.5,116.5,330.1,0.4,0.4,2015-09-13,0530,N,VIIRS,x,1,290.2,3.1,D,0"
    with pytest.raises(ValueError, match="confidence"):
        clean_hotspots(parse_csv(_csv(bad)), bbox=BBOX, tz="Asia/Makassar")


def test_download_archive_never_leaks_map_key_in_errors(tmp_path: Path) -> None:
    def fetch(url: str) -> str:
        raise ConnectionError(f"could not reach {url}")

    with pytest.raises(RuntimeError) as exc_info:
        download_archive(
            map_key="SECRETKEY123",
            source="VIIRS_SNPP_SP",
            bbox=BBOX,
            start=date(2015, 9, 13),
            end=date(2015, 9, 13),
            out_dir=tmp_path,
            max_days=5,
            fetch=fetch,
            sleep_seconds=0,
        )
    assert "SECRETKEY123" not in str(exc_info.value)
    assert "2015-09-13" in str(exc_info.value)


def test_redact_handles_empty_secret() -> None:
    assert redact("abc", "") == "abc"
    assert redact("key=abc", "abc") == "key=***"


# --- availability ------------------------------------------------------------

AVAIL = "data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-07-31\n"


def test_parse_availability_returns_date_range() -> None:
    assert parse_availability(AVAIL, "VIIRS_SNPP_SP") == (date(2012, 1, 20), date(2026, 7, 31))


def test_parse_availability_rejects_unknown_source() -> None:
    with pytest.raises(FirmsApiError, match="MODIS_SP"):
        parse_availability(AVAIL, "MODIS_SP")


def test_fetch_availability_builds_url_and_redacts_errors() -> None:
    seen: list[str] = []

    def fetch(url: str) -> str:
        seen.append(url)
        return AVAIL

    out = fetch_availability(
        "https://x/api/data_availability/csv", map_key="K1", source="VIIRS_SNPP_SP", fetch=fetch
    )
    assert out[1] == date(2026, 7, 31)
    assert seen == ["https://x/api/data_availability/csv/K1/VIIRS_SNPP_SP"]

    def failing(url: str) -> str:
        raise ConnectionError(f"down {url}")

    with pytest.raises(RuntimeError) as exc_info:
        fetch_availability("https://x", map_key="SECRET9", source="VIIRS_SNPP_SP", fetch=failing)
    assert "SECRET9" not in str(exc_info.value)


# --- partial / volatile chunks (F-2.2) ---------------------------------------


def _calls_to_chunks(calls: list[str]) -> list[tuple[str, str]]:
    """(start_date, day_range) per request URL."""
    return [(u.rsplit("/", 1)[-1], u.rsplit("/", 2)[-2]) for u in calls]


def test_chunk_straddling_available_from_is_clipped_and_marked_partial(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({}, calls)
    # aligned chunks from 07-28: [07-28..08-01], [08-02..08-06]; NRT available from 08-01
    download_archive(
        map_key="KEY",
        source="VIIRS_SNPP_NRT",
        bbox=BBOX,
        start=date(2026, 7, 28),
        end=date(2026, 8, 6),
        out_dir=tmp_path,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
        available_from=date(2026, 8, 1),
        available_until=date(2026, 8, 31),
    )
    assert _calls_to_chunks(calls) == [("2026-08-01", "1"), ("2026-08-02", "5")]
    names = sorted(p.name for p in tmp_path.rglob("*.csv"))
    assert names == [
        "VIIRS_SNPP_NRT_2026-07-28_5d_partial.csv",
        "VIIRS_SNPP_NRT_2026-08-02_5d.csv",
    ]


def test_chunk_straddling_available_until_is_clipped_and_marked_partial(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({}, calls)
    download_archive(
        map_key="KEY",
        source="VIIRS_SNPP_SP",
        bbox=BBOX,
        start=date(2026, 7, 23),
        end=date(2026, 8, 6),
        out_dir=tmp_path,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
        available_until=date(2026, 7, 31),
    )
    assert _calls_to_chunks(calls) == [("2026-07-23", "5"), ("2026-07-28", "4")]
    names = sorted(p.name for p in tmp_path.rglob("*.csv"))
    assert names == [
        "VIIRS_SNPP_SP_2026-07-23_5d.csv",
        "VIIRS_SNPP_SP_2026-07-28_5d_partial.csv",
    ]


def test_partial_chunks_are_refetched_but_full_chunks_are_not(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({}, calls)
    kwargs = dict(
        map_key="KEY",
        source="VIIRS_SNPP_SP",
        bbox=BBOX,
        start=date(2026, 7, 23),
        end=date(2026, 8, 6),
        out_dir=tmp_path,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
        available_until=date(2026, 7, 31),
    )
    download_archive(**kwargs)
    download_archive(**kwargs)
    assert _calls_to_chunks(calls) == [
        ("2026-07-23", "5"),
        ("2026-07-28", "4"),
        ("2026-07-28", "4"),
    ]
    # once the API serves the full chunk, the partial file is replaced by a full one
    download_archive(**{**kwargs, "available_until": date(2026, 8, 31)})
    names = sorted(p.name for p in tmp_path.rglob("*.csv"))
    assert "VIIRS_SNPP_SP_2026-07-28_5d.csv" in names
    assert not any(n.endswith("_partial.csv") for n in names if "07-28" in n)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert [m["partial"] for m in manifest] == [False, True, True, False, False]


def test_recent_chunks_are_volatile(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({}, calls)
    download_archive(
        map_key="KEY",
        source="VIIRS_SNPP_NRT",
        bbox=BBOX,
        start=date(2026, 10, 1),
        end=date(2026, 10, 10),
        out_dir=tmp_path,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
        available_until=date(2026, 10, 10),
        volatile_from=date(2026, 10, 8),  # data newer than this may still be incomplete
    )
    names = sorted(p.name for p in tmp_path.rglob("*.csv"))
    assert names == [
        "VIIRS_SNPP_NRT_2026-10-01_5d.csv",
        "VIIRS_SNPP_NRT_2026-10-06_5d_partial.csv",
    ]


def test_prune_removes_chunks_before_available_from(tmp_path: Path) -> None:
    calls: list[str] = []
    fetch = _fake_fetch({}, calls)
    base = dict(
        map_key="KEY",
        source="VIIRS_SNPP_NRT",
        bbox=BBOX,
        start=date(2026, 8, 1),
        end=date(2026, 8, 10),
        out_dir=tmp_path,
        max_days=5,
        fetch=fetch,
        sleep_seconds=0,
        available_until=date(2026, 8, 31),
    )
    download_archive(**base, available_from=date(2026, 8, 1))
    assert len(list(tmp_path.rglob("*.csv"))) == 2
    # SP caught up through 08-05: NRT now only serves from 08-06
    download_archive(**base, available_from=date(2026, 8, 6), prune_before_from=True)
    names = sorted(p.name for p in tmp_path.rglob("*.csv"))
    assert names == ["VIIRS_SNPP_NRT_2026-08-06_5d.csv"]


# --- merge_sources -----------------------------------------------------------


def test_merge_sources_prefers_sp_and_drops_nrt_inside_sp_window() -> None:
    from karhutla.ingest.firms import merge_sources

    sp = clean_hotspots(parse_csv(_csv(ROW_A, ROW_B)), bbox=BBOX, tz="Asia/Makassar")
    nrt_rows = [
        ROW_A,  # same hotspot as SP (duplicate)
        "0.9,116.9,330.0,0.4,0.4,2015-09-20,0530,N,VIIRS,n,2.0NRT,290.0,2.0,D,0",  # after sp_max
        "0.8,116.8,330.0,0.4,0.4,2015-09-14,0530,N,VIIRS,n,2.0NRT,290.0,2.0,D,0",  # inside window
    ]
    nrt = clean_hotspots(parse_csv(_csv(*nrt_rows)), bbox=BBOX, tz="Asia/Makassar")
    nrt = nrt.drop(columns=["type"])  # NRT has no type column

    merged = merge_sources(sp, nrt, sp_max=date(2015, 9, 15))

    assert len(merged) == 3
    assert merged["source"].tolist().count("SP") == 2
    assert merged["source"].tolist().count("NRT") == 1
    assert merged.loc[merged["source"] == "NRT", "acq_date"].iloc[0] == "2015-09-20"
    assert merged["type"].isna().sum() == 1  # only the NRT row lacks type
    assert not merged.duplicated(
        ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    ).any()


# --- run_update (end to end with fakes) -------------------------------------


def test_run_update_twice_yields_identical_rows_without_duplicates(tmp_path: Path) -> None:
    from karhutla.ingest.firms import run_update

    sp_csv = _csv(ROW_A, ROW_B)
    nrt_csv = _csv("0.9,116.9,330.0,0.4,0.4,2015-09-20,0530,N,VIIRS,n,2.0NRT,290.0,2.0,D")
    avail = (
        "data_id,min_date,max_date\n"
        "VIIRS_SNPP_SP,2012-01-20,2015-09-17\n"
        "VIIRS_SNPP_NRT,2015-09-18,2015-09-22\n"
    )

    def fetch(url: str) -> str:
        if "/avail/" in url:
            return avail
        return nrt_csv if "NRT" in url else sp_csv

    kwargs = dict(
        map_key="KEY",
        sp_source="VIIRS_SNPP_SP",
        nrt_source="VIIRS_SNPP_NRT",
        bbox=BBOX,
        start=date(2015, 9, 13),
        today=date(2015, 9, 22),
        raw_root=tmp_path / "raw",
        out_path=tmp_path / "processed" / "hotspots.parquet",
        max_days=5,
        nrt_refetch_days=2,
        tz="Asia/Makassar",
        api_base="https://x/area",
        availability_base="https://x/avail",
        fetch=fetch,
        sleep_seconds=0,
    )
    first = run_update(**kwargs)
    second = run_update(**kwargs)

    assert len(first) == len(second) == 3
    assert first.drop(columns=["source"]).equals(second.drop(columns=["source"]))
    keys = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    assert not second.duplicated(keys).any()
    assert sorted(second["source"].unique()) == ["NRT", "SP"]
    meta = json.loads((tmp_path / "processed" / "hotspots.meta.json").read_text())
    assert meta["sp_max"] == "2015-09-17"
    assert meta["nrt_max"] == "2015-09-22"
