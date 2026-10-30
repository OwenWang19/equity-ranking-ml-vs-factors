"""Acquire small official public samples and risk benchmarks, not an alpha universe.

Run ``python -m equity_alpha.public_data --output data/access-YYYYMMDD``.
No account, personal API key, subscription, model fitting, or existing-file
replacement is involved. All sources are bounded to 8 MB per HTTP response.
The AAPL demonstration is not a substitute for historical, delisted stocks.
Current-vintage French factors are ex-post benchmarks, not certified PIT inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

MAX_BYTES = 8 * 1024 * 1024
START = "2005-01-01"
STOP = "2025-01-01"  # Exclusive; no reserved-period derived table is emitted.
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
SOURCES = {
    "ff5": {
        "url": FRENCH + "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "filename": "ff5_daily.zip",
        "documentation": FRENCH + "data_library.html",
        "role": "Current-vintage ex-post risk benchmarks; not individual-stock labels or PIT features",
        "redistribution": "Public academic download; no claim of an unrestricted raw-data license",
    },
    "momentum": {
        "url": FRENCH + "ftp/F-F_Momentum_Factor_daily_CSV.zip",
        "filename": "momentum_daily.zip",
        "documentation": FRENCH + "data_library.html",
        "role": "Current-vintage ex-post risk benchmark; not an individual-stock target",
        "redistribution": "Public academic download; no claim of an unrestricted raw-data license",
    },
    "sharadar_demo": {
        "url": "https://api.sharadar.com/v1.0/data/stocks?api_key=test-api-key"
        "&ticker=AAPL&from=2005-01-01&to=2024-12-31&sort=date.asc&limit=10000&format=csv",
        "filename": "sharadar_aapl_demo.csv",
        "documentation": "https://sharadar.com/docs/auth",
        "role": "Single surviving stock: HTTP/schema integration sample only; not approved research data",
        "redistribution": "https://sharadar.com/terms ; personal-use/publication restrictions apply",
    },
}


def acquire(url: str, destination: Path, *, opener=None) -> dict:
    """Download to a new file only; bound bytes and retain HTTP provenance."""
    if destination.exists():
        raise FileExistsError(destination)
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS downloads are permitted")
    request = urllib.request.Request(
        url, headers={"User-Agent": "EquityAlphaResearch/0.1 public-source-audit"}
    )
    open_url = opener or urllib.request.urlopen
    started = datetime.now(UTC).isoformat()
    with open_url(request, timeout=45) as response:
        final_url = response.geturl()
        if not final_url.startswith("https://"):
            raise ValueError("Non-HTTPS redirect")
        if response.status != 200:
            raise ValueError(f"Expected HTTP 200, received {response.status}")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_BYTES:
            raise ValueError("Response exceeds download size limit")
        payload = response.read(MAX_BYTES + 1)
        if not payload or len(payload) > MAX_BYTES:
            raise ValueError("Empty response or download size limit exceeded")
        if declared is not None and len(payload) != int(declared):
            raise ValueError("Truncated response: Content-Length mismatch")
        evidence = {
            "requested_url": url,
            "final_url": final_url,
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "started_at_utc": started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    with destination.open("xb") as handle:
        handle.write(payload)
    return evidence


def read_french_daily(path: Path, expected: list[str]) -> tuple[pd.DataFrame, dict]:
    """Parse the first daily CSV block, convert percent to decimal, preserve gaps.

    Monthly/annual sections and copyright prose are not interpreted as daily
    observations. ZIP members are inspected in memory, never extracted.
    """
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise ValueError("Expected exactly one CSV ZIP member")
        member = members[0]
        if member.file_size > MAX_BYTES:
            raise ValueError("Uncompressed CSV exceeds size limit")
        rows = csv.reader(io.StringIO(archive.read(member).decode("utf-8-sig")))
        header = None
        values = []
        dates = []
        for row in rows:
            cells = [cell.strip() for cell in row]
            if header is None:
                if cells and cells[1:] == expected:
                    header = cells
                continue
            if not cells or not cells[0]:
                if dates:
                    break
                continue
            if not re.fullmatch(r"\d{8}", cells[0]):
                raise ValueError("Unexpected row in daily factor block")
            if len(cells) != len(expected) + 1:
                raise ValueError("Malformed daily factor row")
            dates.append(cells[0])
            values.append([float(value) for value in cells[1:]])
    if header is None or not dates:
        raise ValueError("Daily factor header/data not found")
    index = pd.to_datetime(dates, format="%Y%m%d", errors="raise")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Daily factor dates must be unique and increasing")
    frame = pd.DataFrame(values, index=index, columns=expected, dtype=float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite factor values")
    missing = frame.isin([-99.99, -999.0])
    frame = frame.mask(missing) / 100.0
    frame.index.name = "date"
    return frame, {
        "zip_member": member.filename,
        "rows": len(frame),
        "first_date": index.min().date().isoformat(),
        "last_date": index.max().date().isoformat(),
        "source_unit": "percent",
        "derived_unit": "decimal return",
        "sentinel_missing_count": int(missing.to_numpy().sum()),
        "not_point_in_time_vintages": True,
    }


def development_benchmarks(ff5: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    """Fail rather than silently inner-joining mismatched development calendars."""
    if ff5.columns.tolist() != ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        raise ValueError("Unexpected FF5 column order/schema")
    if momentum.columns.tolist() != ["Mom"]:
        raise ValueError("Unexpected momentum column order/schema")
    ff5 = ff5.loc[(ff5.index >= START) & (ff5.index < STOP)]
    momentum = momentum.loc[(momentum.index >= START) & (momentum.index < STOP)]
    if ff5.empty or not ff5.index.equals(momentum.index):
        raise ValueError("Factor development calendars differ or are empty; do not silently drop dates")
    result = ff5.join(momentum, validate="one_to_one")
    if result.isna().any().any():
        raise ValueError("Development factors contain missing observations; no filling is permitted")
    result.columns = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]
    return result


def audit_stock_sample(path: Path, expected_calendar: pd.DatetimeIndex | None = None) -> dict:
    frame = pd.read_csv(path, dtype={"ticker": str})
    fields = {"ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj"}
    if not fields.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing sample stock data/schema (possibly an API error response)")
    if set(frame["ticker"]) != {"AAPL"}:
        raise ValueError("Public demonstration is restricted to AAPL")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Sample dates must be unique, present and increasing")
    if (dates < START).any() or (dates >= STOP).any():
        raise ValueError("Sample server ignored requested development bounds")
    numeric = sorted(fields - {"ticker", "date"})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Nonfinite sample price/volume")
    if (frame[numeric].drop(columns="volume") <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Nonpositive sample prices or negative volume")
    if (
        (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
    ).any():
        raise ValueError("Invalid OHLC ordering")
    audit = {
        "status": "schema_checked_demo_only",
        "rows": len(frame),
        "ticker_count": 1,
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "requested_first_date": START,
        "requested_last_date": "2024-12-31",
        "covers_requested_calendar": "not independently verified",
        "eligible_for_cross_sectional_research": False,
        "reason": "One surviving stock; no audited historical universe, identity or terminal-payout history",
        "approved_for_public_research_report": False,
        "model_fitted": False,
    }
    if expected_calendar is not None:
        observed = pd.DatetimeIndex(dates)
        absent = expected_calendar.difference(observed)
        extra = observed.difference(expected_calendar)
        audit["reference_calendar"] = "French benchmark dates, not an independent exchange calendar"
        audit["reference_calendar_rows"] = len(expected_calendar)
        audit["missing_reference_sessions"] = len(absent)
        audit["extra_sessions"] = len(extra)
        audit["matches_reference_calendar"] = not len(absent) and not len(extra)
        audit["observed_reference_coverage"] = (
            len(expected_calendar.intersection(observed)) / len(expected_calendar)
            if len(expected_calendar)
            else None
        )
        audit["query_history_truncated"] = len(absent) > 0
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing paths are rejected"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "raw"
    raw.mkdir()
    source_bytes = Path(__file__).read_bytes()
    (args.output / "acquisition_source.py").write_bytes(source_bytes)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Auxiliary benchmarks and access tests, not a broader equity-alpha experiment",
        "development_start": START,
        "reserved_start": STOP,
        "model_fitted": False,
        "broad_stock_dataset_obtained": False,
        "acquisition_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sources": {},
    }
    for name, spec in SOURCES.items():
        record = {**spec}
        manifest["sources"][name] = record
        try:
            record["download"] = acquire(spec["url"], raw / spec["filename"])
            record["status"] = "downloaded"
        except (OSError, ValueError) as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    table = None
    try:
        ff5, ff_audit = read_french_daily(
            raw / SOURCES["ff5"]["filename"], ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        )
        mom, mom_audit = read_french_daily(raw / SOURCES["momentum"]["filename"], ["Mom"])
        table = development_benchmarks(ff5, mom)
        target = args.output / "risk_benchmarks_2005_2024.csv"
        table.to_csv(target, float_format="%.10g")
        manifest["benchmarks"] = {
            "status": "auxiliary_only",
            "ff5_audit": ff_audit,
            "momentum_audit": mom_audit,
            "derived_rows": len(table),
            "derived_first_date": table.index.min().date().isoformat(),
            "derived_last_date": table.index.max().date().isoformat(),
            "derived_columns": table.columns.tolist(),
            "derived_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "raw_reserved_rows_ff5": int((ff5.index >= STOP).sum()),
            "raw_reserved_rows_momentum": int((mom.index >= STOP).sum()),
            "reserved_rows_in_derived_table": 0,
            "raw_files_are_current_vintage": True,
            "usage": "Ex-post risk attribution only; no feature or stock-universe qualification",
            "mkt_rf_definition": "Already excess of RF; do not subtract RF again",
        }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        manifest["benchmarks"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        manifest["stock_sample"] = audit_stock_sample(
            raw / SOURCES["sharadar_demo"]["filename"], table.index if table is not None else None
        )
    except (OSError, ValueError) as exc:
        manifest["stock_sample"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    manifest["acquisition_checks_passed"] = (
        all(record["status"] == "downloaded" for record in manifest["sources"].values())
        and manifest["benchmarks"]["status"] == "auxiliary_only"
        and manifest["stock_sample"]["status"] == "schema_checked_demo_only"
    )
    with (args.output / "acquisition_audit.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    print(json.dumps(manifest, indent=2, allow_nan=False))
    if not manifest["acquisition_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
