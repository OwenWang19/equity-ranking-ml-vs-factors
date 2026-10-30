"""Read a Qlib US ZIP without extracting files or executing provider code.

The provider's normalized adjusted closes are useful return indices. Historical
membership intervals are reconstructed data, not certified point-in-time records.
Missing observations remain missing, including observations after membership exit.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

SOURCE_URL = "https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_us_1d_latest.zip"
NORMALIZATION_URL = "https://github.com/microsoft/qlib/blob/main/scripts/data_collector/yahoo/collector.py"
_QUALIFIED_ARCHIVE_SHA256 = "20e3283009784843dcfe690488bf3aa739e64159f1da51a9ec33dd3fb647187f"
_FIELDS = ("open", "high", "low", "close", "volume", "factor")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def classify_membership_quality(
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], source_sha256: str
) -> dict:
    """Attach verified source-specific findings without generalizing to other ZIPs.

    A date-shaped source record does not establish historical membership. The
    archived BRK.B chronology has been checked against an issuer announcement.
    Unknown source hashes remain unverified, even if they use similar labels.
    """
    result = {
        "status": "unverified",
        "historical_index_universe_claim_supported": False,
        "known_issues": [],
    }
    if source_sha256 != _QUALIFIED_ARCHIVE_SHA256 or "BRK.B" not in intervals:
        return result
    first_start = min(begin for begin, _ in intervals["BRK.B"])
    if first_start < pd.Timestamp("2010-01-26"):
        result["status"] = "known_inaccurate"
        result["known_issues"].append(
            {
                "ticker": "BRK.B",
                "source_first_membership_start": first_start.date().isoformat(),
                "issuer_new_inclusion_announcement_date": "2010-01-26",
                "issuer_news_release_date": "2010-01-28",
                "source_url": "https://www.berkshirehathaway.com/news/jan2810.pdf",
                "issue": "Source membership precedes the issuer's announcement of its new S&P 500 inclusion",
                "scope": "Verified for this archive SHA-256; not a correction of its membership intervals",
            }
        )
    return result


def _date(value: str, context: str) -> pd.Timestamp:
    if not _DATE.fullmatch(value):
        raise ValueError(f"Invalid ISO date in {context}: {value!r}")
    return pd.Timestamp(value)


def _safe_names(archive: zipfile.ZipFile) -> set[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP member paths")
    for name in names:
        path = PurePosixPath(name)
        if (
            "\\" in name
            or name.startswith("/")
            or ":" in name
            or ".." in path.parts
            or "\x00" in name
            or any(part in ("", ".") for part in name.rstrip("/").split("/"))
        ):
            raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return set(names)


def _text(archive: zipfile.ZipFile, name: str) -> list[str]:
    info = archive.getinfo(name)
    if info.file_size > 8_000_000:
        raise ValueError(f"Oversized metadata member: {name}")
    return archive.read(info).decode("utf-8").splitlines()


def _read_series(archive: zipfile.ZipFile, name: str, n_dates: int) -> tuple[np.ndarray, int]:
    info = archive.getinfo(name)
    if info.file_size < 8 or info.file_size % 4 or info.file_size > 4 * (n_dates + 1):
        raise ValueError(f"Invalid/truncated binary length: {name}")
    binary = np.frombuffer(archive.read(info), dtype="<f4")
    start = float(binary[0])
    if not np.isfinite(start) or start != int(start) or start < 0:
        raise ValueError(f"Invalid calendar start index: {name}")
    start = int(start)
    values = binary[1:].astype(float)
    if start >= n_dates or start + len(values) > n_dates:
        raise ValueError(f"Binary span exceeds calendar: {name}")
    output = np.full(n_dates, np.nan)
    output[start : start + len(values)] = values
    return output, start


def load_qlib_archive(
    path: Path,
    symbol_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return close, volume, factor, literal membership, and JSON-safe audit.

    Economic-invalid observations become NaN with an explicit per-name/date audit.
    All matrices share the full archive calendar and all historical S&P 500
    labels. Quotes outside membership are retained so existing positions can be
    marked/closed. No quote is carried forward or backfilled. An optional explicit
    one-to-one vendor symbol map changes only archive lookup; membership labels
    remain unchanged, and recovery relative to literal lookups is audited.

    Qlib stores little-endian float32 arrays: first float = calendar offset,
    following floats = values. Only the six named fields for historical S&P 500
    labels are read. The source archive is immutable and is hashed in chunks.
    """
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    with zipfile.ZipFile(source) as archive:
        names = _safe_names(archive)
        for required in ("calendars/day.txt", "instruments/sp500.txt"):
            if required not in names:
                raise ValueError(f"Missing required archive member: {required}")
        date_lines = _text(archive, "calendars/day.txt")
        if not date_lines:
            raise ValueError("Archive calendar is empty")
        dates = pd.DatetimeIndex([_date(value, "calendar") for value in date_lines], name="date")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Calendar must be strictly increasing and unique")

        intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for line in _text(archive, "instruments/sp500.txt"):
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError("Membership records require ticker, start, end separated by tabs")
            ticker, first, last = parts
            if (
                not ticker
                or ticker.strip() != ticker
                or any(char.isspace() for char in ticker)
                or any(char in ticker for char in ("/", "\\", ":", "\x00"))
                or ticker in (".", "..")
            ):
                raise ValueError(f"Unsafe membership ticker: {ticker!r}")
            begin, end = _date(first, "membership"), _date(last, "membership")
            if begin > end:
                raise ValueError(f"Membership start exceeds end: {ticker}")
            intervals.setdefault(ticker, []).append((begin, end))
        if not intervals:
            raise ValueError("No historical membership records")
        tickers = sorted(intervals)
        if len({ticker.lower() for ticker in tickers}) != len(tickers):
            raise ValueError("Membership tickers collide under lowercase archive paths")
        if symbol_map is not None and not isinstance(symbol_map, dict):
            raise TypeError("symbol_map must be a dictionary of membership to vendor symbols")
        vendor_map = dict(symbol_map or {})
        for ticker, vendor in vendor_map.items():
            if not isinstance(ticker, str) or not isinstance(vendor, str):
                raise TypeError("Symbol map keys and values must be strings")
            if ticker not in intervals:
                raise ValueError(f"Unknown membership symbol in symbol_map: {ticker}")
            if not re.fullmatch(r"[A-Za-z0-9._^=-]+", vendor) or vendor in (".", ".."):
                raise ValueError(f"Unsafe vendor symbol: {vendor!r}")
            if f"features/{vendor.lower()}/close.day.bin" not in names:
                raise ValueError(f"Mapped vendor close member is missing: {vendor}")
        lookup_symbols = {ticker: vendor_map.get(ticker, ticker) for ticker in tickers}
        if len({vendor.lower() for vendor in lookup_symbols.values()}) != len(tickers):
            raise ValueError("Symbol map creates a vendor-symbol collision; lookups must be one-to-one")

        shape = len(dates), len(tickers)
        data = {field: np.full(shape, np.nan) for field in ("close", "volume", "factor")}
        membership = np.zeros(shape, dtype=bool)
        missing_fields: dict[str, list[str]] = {}
        field_counts = {field: 0 for field in _FIELDS}
        binary_start_indices: dict[str, dict[str, int]] = {}
        invalid_counts: dict[str, dict[str, int]] = {}
        invalid_by_date: dict[str, dict[str, int]] = {}
        invalid_details: dict[str, dict[str, list[str]]] = {}
        missing_after_exit: list[dict] = []
        per_ticker: dict[str, dict] = {}
        mapping_comparison: dict[str, dict] = {}
        before_missing_close: list[str] = []
        before_member_close_count = 0
        for column, ticker in enumerate(tickers):
            for begin, end in intervals[ticker]:
                membership[:, column] |= (dates >= begin) & (dates <= end)
            values: dict[str, np.ndarray] = {}
            missing_fields[ticker] = []
            binary_start_indices[ticker] = {}
            invalid_counts[ticker] = {}
            invalid_details[ticker] = {}
            for field in _FIELDS:
                name = f"features/{lookup_symbols[ticker].lower()}/{field}.day.bin"
                if name not in names:
                    missing_fields[ticker].append(field)
                    values[field] = np.full(len(dates), np.nan)
                    continue
                vector, start = _read_series(archive, name, len(dates))
                binary_start_indices[ticker][field] = start
                invalid = np.isinf(vector) | (vector < 0)
                if field != "volume":
                    invalid |= vector == 0
                invalid_counts[ticker][field] = int(invalid.sum())
                if invalid.any():
                    invalid_details[ticker][field] = []
                    for position in np.flatnonzero(invalid):
                        day = dates[position].date().isoformat()
                        invalid_details[ticker][field].append(day)
                        invalid_by_date.setdefault(day, {}).setdefault(field, 0)
                        invalid_by_date[day][field] += 1
                    vector[invalid] = np.nan
                values[field] = vector
                field_counts[field] += int(np.isfinite(vector).sum())
            for field, matrix in data.items():
                matrix[:, column] = values[field]
            active = membership[:, column]
            valid = np.isfinite(values["close"])
            valid_indices = np.flatnonzero(valid)
            expected = int(active.sum())
            missing = int((active & ~valid).sum())
            literal_valid = valid
            if ticker in vendor_map:
                literal_name = f"features/{ticker.lower()}/close.day.bin"
                if literal_name in names:
                    literal_values, _ = _read_series(archive, literal_name, len(dates))
                    literal_valid = np.isfinite(literal_values) & (literal_values > 0)
                else:
                    literal_valid = np.zeros(len(dates), dtype=bool)
                mapping_comparison[ticker] = {
                    "vendor_symbol": vendor_map[ticker],
                    "literal_observed_close_sessions": int(literal_valid.sum()),
                    "mapped_observed_close_sessions": int(valid.sum()),
                    "literal_member_close_sessions": int((active & literal_valid).sum()),
                    "mapped_member_close_sessions": int((active & valid).sum()),
                    "recovered_member_sessions": int((active & valid & ~literal_valid).sum()),
                    "lost_member_sessions": int((active & ~valid & literal_valid).sum()),
                }
            before_member_close_count += int((active & literal_valid).sum())
            if not literal_valid.any():
                before_missing_close.append(ticker)
            next_valid = np.r_[valid[1:], False]
            exits = np.flatnonzero(active[:-1] & ~active[1:])
            for position in exits:
                if not valid[position + 1]:
                    missing_after_exit.append(
                        {
                            "ticker": ticker,
                            "last_member_date": dates[position].date().isoformat(),
                            "next_session": dates[position + 1].date().isoformat(),
                        }
                    )
            per_ticker[ticker] = {
                "membership_sessions": expected,
                "member_sessions_with_close": expected - missing,
                "member_sessions_missing_close": missing,
                "member_sessions_without_next_session_close": int((active & ~next_valid).sum()),
                "observed_close_sessions": len(valid_indices),
                "first_close_date": dates[valid_indices[0]].date().isoformat()
                if len(valid_indices)
                else None,
                "last_close_date": dates[valid_indices[-1]].date().isoformat()
                if len(valid_indices)
                else None,
            }

    frames = {field: pd.DataFrame(array, index=dates, columns=tickers) for field, array in data.items()}
    member_frame = pd.DataFrame(membership, index=dates, columns=tickers)
    covered = member_frame & frames["close"].notna()
    expected = int(membership.sum())
    audit = {
        "schema_version": 1,
        "source_path": str(source),
        "source_url": SOURCE_URL,
        "source_sha256": digest.hexdigest(),
        "source_bytes": source.stat().st_size,
        "archive_member_count": len(names),
        "calendar_start": dates[0].date().isoformat(),
        "calendar_end": dates[-1].date().isoformat(),
        "n_dates": len(dates),
        "n_tickers": len(tickers),
        "membership_record_count": sum(map(len, intervals.values())),
        "membership_quality": classify_membership_quality(intervals, digest.hexdigest()),
        "tickers": tickers,
        "vendor_symbol_map": dict(sorted(vendor_map.items())),
        "vendor_symbol_map_sha256": hashlib.sha256(
            json.dumps(vendor_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "symbol_map_comparison": {
            "per_ticker": mapping_comparison,
            "tickers_without_close_before_mapping": before_missing_close,
            "recovered_names": [
                ticker
                for ticker, comparison in mapping_comparison.items()
                if comparison["literal_observed_close_sessions"] == 0
                and comparison["mapped_observed_close_sessions"] > 0
            ],
            "member_close_sessions_before_mapping": before_member_close_count,
            "member_close_sessions_after_mapping": int(covered.to_numpy().sum()),
            "member_close_coverage_before_mapping": before_member_close_count / expected
            if expected
            else None,
            "recovered_member_sessions": sum(
                item["recovered_member_sessions"] for item in mapping_comparison.values()
            ),
            "lost_member_sessions": sum(item["lost_member_sessions"] for item in mapping_comparison.values()),
        },
        "missing_fields": {ticker: fields for ticker, fields in missing_fields.items() if fields},
        "tickers_without_close_data": [
            ticker for ticker in tickers if not frames["close"][ticker].notna().any()
        ],
        "field_finite_observations": field_counts,
        "binary_start_indices": binary_start_indices,
        "economic_invalid_observations": {
            "count": sum(sum(fields.values()) for fields in invalid_counts.values()),
            "per_ticker_field": {
                ticker: {field: count for field, count in fields.items() if count}
                for ticker, fields in invalid_counts.items()
                if any(fields.values())
            },
            "per_date_field": invalid_by_date,
            "dates_by_ticker_field": {ticker: fields for ticker, fields in invalid_details.items() if fields},
            "treatment": "Convert to NaN in returned views; retain unchanged source archive bytes",
        },
        "membership_observations": expected,
        "membership_observations_with_close": int(covered.to_numpy().sum()),
        "membership_close_coverage": float(covered.to_numpy().sum() / expected) if expected else None,
        "membership_by_date": {
            day.date().isoformat(): {
                "members": int(member_frame.loc[day].sum()),
                "members_with_close": int(covered.loc[day].sum()),
            }
            for day in dates
        },
        "per_ticker": per_ticker,
        "membership_exits_without_next_session_close": missing_after_exit,
        "zero_volume_observations": int(frames["volume"].eq(0).to_numpy().sum()),
        "membership_convention": "Inclusive literal intervals from archive; reconstructed, not certified PIT",
        "normalization_source": NORMALIZATION_URL,
        "raw_price_formula": "normalized OHLC / normalized factor",
        "raw_volume_formula": "normalized volume * normalized factor",
        "dollar_turnover_formula": "normalized close * normalized volume",
        "constraints": [
            "Yahoo Finance sample; not a licensed survivor-complete or independently validated return database.",
            "Historical membership is reconstructed; announcement timing and constituent identities are unverified.",
            "Ticker reuse, renames, delistings, terminal payouts and security-type history are not resolved.",
            "No delisting return, borrow availability, execution price, spread or share-level corporate-action audit.",
            "Missing prices remain NaN, including missing next-session exit prices; no future availability filter.",
            "Normalized price ratios reflect provider adjustments, not independently certified total returns.",
            "Microsoft code MIT license does not establish redistribution rights over upstream Yahoo data.",
        ],
    }
    return frames["close"], frames["volume"], frames["factor"], member_frame, audit
