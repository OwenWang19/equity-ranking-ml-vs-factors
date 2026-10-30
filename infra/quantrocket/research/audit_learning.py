"""Derived-only audit of the completed QuantRocket 2007-2011 learning bundle.

Run with the QuantRocket Zipline Python environment, not zipline-reloaded. Check
the bundle's completion message in flightlog before passing --ingestion-complete.
The flag records the caller's attestation; list_bundles alone does not prove that
an asynchronous ingestion finished. This script does not ingest, export bars,
fit models, qualify return labels, or modify the original 2025 holdout protocol.

Official interfaces/source verified:
https://www.quantrocket.com/docs/api/#module-zipline.research
https://github.com/quantrocket-llc/zipline/blob/master/zipline/research/bundle.py
https://github.com/quantrocket-llc/zipline/blob/master/zipline/data/bcolz_daily_bars.py
https://github.com/quantrocket-llc/zipline/blob/master/zipline/assets/assets.py
https://github.com/quantrocket-llc/zipline/blob/master/zipline/data/_adjustments.pyx
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = "usstock-learn-1d"
FIELDS = ("open", "high", "low", "close", "volume")
IDENTITY_SYMBOLS = ("LEH", "KKD", "MON")
MAX_SESSIONS = 1500
MAX_SECURITIES = 50000


def date_string(value):
    """Normalize metadata/session timestamps without interpreting missing dates."""
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.date().isoformat()


def validate_sessions(values):
    """Reject another date scope before reading bars; retain reader-native labels."""
    sessions = pd.DatetimeIndex(values)
    if sessions.empty or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("Bundle sessions must be nonempty, unique, and increasing")
    if len(sessions) > MAX_SESSIONS:
        raise ValueError("Unexpected learning-bundle session count")
    first, last = date_string(sessions[0]), date_string(sessions[-1])
    if first < "2007-01-01" or last > "2011-12-31":
        raise ValueError("Only the separate 2007-2011 learning scope is authorized")
    return sessions


def summarize_bars(sessions, arrays, expected_assets):
    """Reduce unadjusted reader arrays to dates/counts, never price/volume values.

    A close observation is a finite, strictly positive close. A complete OHLCV
    observation has four finite positive prices and finite nonnegative volume.
    Zero volume is retained as distinct from a positively traded session. Missing
    bars are not forward-filled. First/last observations are not delisting dates.
    """
    if len(arrays) != len(FIELDS):
        raise ValueError("Expected exactly the five OHLCV arrays")
    values = [np.asarray(array) for array in arrays]
    shape = (len(sessions), expected_assets)
    if any(value.shape != shape for value in values):
        raise ValueError("Unexpected bar-array shape")
    opens, highs, lows, closes, volumes = values
    prices = np.stack(values[:4])
    finite_prices = np.isfinite(prices).all(axis=0)
    positive_prices = finite_prices & (prices > 0).all(axis=0)
    finite_volume = np.isfinite(volumes)
    observed_close = np.isfinite(closes) & (closes > 0)
    complete = positive_prices & finite_volume & (volumes >= 0)
    traded = complete & (volumes > 0)
    incoherent = positive_prices & (
        (highs < lows) | (highs < opens) | (highs < closes) | (lows > opens) | (lows > closes)
    )
    rows = []
    for column in range(expected_assets):
        close_locations = np.flatnonzero(observed_close[:, column])
        complete_locations = np.flatnonzero(complete[:, column])
        rows.append(
            {
                "observed_close_sessions": int(observed_close[:, column].sum()),
                "first_observed_close": date_string(sessions[close_locations[0]])
                if close_locations.size
                else None,
                "last_observed_close": date_string(sessions[close_locations[-1]])
                if close_locations.size
                else None,
                "complete_ohlcv_sessions": int(complete[:, column].sum()),
                "first_complete_ohlcv": date_string(sessions[complete_locations[0]])
                if complete_locations.size
                else None,
                "last_complete_ohlcv": date_string(sessions[complete_locations[-1]])
                if complete_locations.size
                else None,
                "positive_volume_ohlcv_sessions": int(traded[:, column].sum()),
                "zero_volume_with_complete_ohlc_sessions": int(
                    (positive_prices[:, column] & finite_volume[:, column] & (volumes[:, column] == 0)).sum()
                ),
                "negative_volume_sessions": int((finite_volume[:, column] & (volumes[:, column] < 0)).sum()),
                "ohlc_order_violation_sessions": int(incoherent[:, column].sum()),
            }
        )
    return rows


def asset_metadata(asset):
    """These are latest bundle metadata, not historical point-in-time attributes."""
    return {
        "real_sid": str(asset.real_sid),
        "zipline_sid": int(asset.sid),
        "latest_symbol": asset.symbol,
        "asset_name": asset.asset_name,
        "country_code": asset.country_code,
        "start_date": date_string(asset.start_date),
        "end_date": date_string(asset.end_date),
        "auto_close_date": date_string(asset.auto_close_date),
    }


def summarize_lifetimes(rows, cutoff):
    """Separate metadata exits, observation endings, and cutoff censoring."""

    def before(row, key):
        return row.get(key) is not None and row[key] < cutoff

    def on(row, key):
        return row.get(key) == cutoff

    return {
        "securities": len(rows),
        "common_stock_count": None,
        "common_stock_classification": "unknown; equity metadata also includes non-common-stock securities",
        "metadata_auto_close_before_cutoff": sum(before(row, "auto_close_date") for row in rows),
        "metadata_auto_close_on_cutoff": sum(on(row, "auto_close_date") for row in rows),
        "metadata_auto_close_after_cutoff": sum(
            row.get("auto_close_date") is not None and row["auto_close_date"] > cutoff for row in rows
        ),
        "metadata_auto_close_missing": sum(row.get("auto_close_date") is None for row in rows),
        "metadata_end_before_cutoff": sum(before(row, "end_date") for row in rows),
        "metadata_end_on_cutoff": sum(on(row, "end_date") for row in rows),
        "last_observed_close_before_cutoff": sum(before(row, "last_observed_close") for row in rows),
        "observed_close_on_cutoff_right_censoring_candidates": sum(
            on(row, "last_observed_close") for row in rows
        ),
        "no_observed_close": sum(row.get("last_observed_close") is None for row in rows),
        "verified_delisted_security_count": None,
        "interpretation": (
            "Metadata exits are candidates, not verified delistings or point-in-time status. "
            "Observation endings may reflect inactivity, missing coverage, or delisting. "
            "An end at the bundle cutoff is censoring, not evidence of death. "
            "Future metadata must not be used as a contemporaneous selection feature."
        ),
    }


def resolve_identity_examples(finder, rows, lookup_errors=(LookupError,)):
    """Use bundle symbol ownership, never concatenate same-ticker price series."""
    by_sid = {row["zipline_sid"]: row for row in rows}
    examples = {}
    for symbol in IDENTITY_SYMBOLS:
        try:
            asset = finder.lookup_symbol(symbol)
        except lookup_errors as exc:
            # Do not guess a company or join data when symbol ownership is ambiguous.
            examples[symbol] = {"resolution": "unresolved", "error_type": type(exc).__name__}
            continue
        row = by_sid.get(int(asset.sid))
        examples[symbol] = {
            "resolution": "unique_bundle_symbol_mapping" if row is not None else "outside_audited_equities",
            "company_identity_independently_verified": False,
            "point_in_time_symbol_verified": False,
            "metadata_and_observation_summary": row,
        }
    return examples


def collect_audit(bundle_data, chunk_size=256, symbol_lookup_errors=(LookupError,)):
    """Read existing local readers only; no client, ingestion, broker, or HTTP calls."""
    if not 1 <= chunk_size <= 512:
        raise ValueError("chunk_size must be between 1 and 512")
    reader = bundle_data.equity_daily_bar_reader
    sessions = validate_sessions(reader.sessions)
    finder = bundle_data.asset_finder
    zipline_sids = list(finder.equities_sids)
    if not 1 <= len(zipline_sids) <= MAX_SECURITIES or len(set(zipline_sids)) != len(zipline_sids):
        raise ValueError("Unexpected or duplicate equity IDs")
    assets = finder.retrieve_all(zipline_sids)
    if len(assets) != len(zipline_sids) or [int(asset.sid) for asset in assets] != zipline_sids:
        raise ValueError("AssetFinder returned assets in an unexpected order")
    rows = [asset_metadata(asset) for asset in assets]
    if len({row["real_sid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate persistent QuantRocket security IDs")
    daily_observed = np.zeros(len(sessions), dtype=np.int64)
    for offset in range(0, len(zipline_sids), chunk_size):
        chunk = zipline_sids[offset : offset + chunk_size]
        arrays = reader.load_raw_arrays(list(FIELDS), sessions[0], sessions[-1], chunk)
        derived = summarize_bars(sessions, arrays, len(chunk))
        closes = np.asarray(arrays[3])
        daily_observed += (np.isfinite(closes) & (closes > 0)).sum(axis=1)
        for row, stats in zip(rows[offset : offset + len(chunk)], derived, strict=True):
            row.update(stats)
        del arrays, closes

    country_codes = tuple(sorted({asset.country_code for asset in assets if asset.country_code}))
    # QuantRocket's AssetFinder.lifetimes uses start_date and AUTO_CLOSE_DATE,
    # not end_date. Include first trading dates for this metadata-only coverage audit.
    life = finder.lifetimes(sessions, include_start_date=True, country_codes=country_codes)
    if not life.index.equals(sessions):
        raise ValueError("Unexpected lifetime session labels")
    life = life.reindex(columns=zipline_sids)
    if life.isna().any().any():
        raise ValueError("Missing lifetime entries for audited assets")
    alive_counts = life.sum(axis=1)
    cutoff = date_string(sessions[-1])
    observed_sessions = np.flatnonzero(daily_observed)
    return {
        "schema_version": 1,
        "bundle": BUNDLE,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "learning_only": True,
            "authorized_window": ["2007-01-01", "2011-12-31"],
            "original_2025_holdout_protocol_modified": False,
            "raw_data_exported": False,
            "model_fitted": False,
        },
        "sessions": {
            "first": date_string(sessions[0]),
            "last": cutoff,
            "count": len(sessions),
            "reader_first_trading_day": date_string(reader.first_trading_day),
            "reader_last_available_dt": date_string(reader.last_available_dt),
            "first_with_any_observed_close": date_string(sessions[observed_sessions[0]])
            if observed_sessions.size
            else None,
            "last_with_any_observed_close": date_string(sessions[observed_sessions[-1]])
            if observed_sessions.size
            else None,
            "sessions_without_any_observed_close": int((daily_observed == 0).sum()),
        },
        "coverage": summarize_lifetimes(rows, cutoff),
        "observation_totals": {
            field: sum(row[field] for row in rows)
            for field in (
                "observed_close_sessions",
                "complete_ohlcv_sessions",
                "positive_volume_ohlcv_sessions",
                "zero_volume_with_complete_ohlc_sessions",
                "negative_volume_sessions",
                "ohlc_order_violation_sessions",
            )
        },
        "metadata_lifetimes": {
            "definition": "AssetFinder.lifetimes, start_date <= session <= auto_close_date; missing end is open",
            "point_in_time_verified": False,
            "daily_alive_first": int(alive_counts.iloc[0]),
            "daily_alive_last": int(alive_counts.iloc[-1]),
            "daily_alive_min": int(alive_counts.min()),
            "daily_alive_max": int(alive_counts.max()),
        },
        "observation_definition": (
            "Direct unadjusted daily-reader arrays, no forward-fill. Observed close = finite and > 0. "
            "Complete OHLCV = four finite positive prices and finite nonnegative volume. "
            "Only derived counts and dates are serialized; prices and volume values are not."
        ),
        "identity_examples": resolve_identity_examples(finder, rows, symbol_lookup_errors),
        "qualifications": {
            "model_ready": False,
            "point_in_time_metadata": "unqualified",
            "common_stock_universe": "unqualified",
            "delisting_events_and_terminal_returns": "unqualified",
            "adjusted_return_labels": "unqualified",
            "corporate_action_completeness": "unqualified",
            "notes": [
                "Security membership and historical observations do not qualify investability or survivorship controls.",
                "Do not infer a zero terminal return, reinvestment proceeds, or delisting return from the last close.",
                "Raw reader arrays are unadjusted; returns crossing corporate actions need separate validation.",
                "Zipline query/history adjustments are as of the query end or simulation time, not necessarily today.",
                (
                    "Split adjustments multiply past prices by a ratio and past volume by its inverse; dividends "
                    "and mergers affect prices only. Do not apply a dividend price factor to volume."
                ),
                "Latest ticker/name and future auto-close metadata are not proven point-in-time classifications.",
                "Do not combine this learning exercise with, or claim results for, the original 2025 holdout.",
            ],
        },
    }


def write_audit(report, output_dir):
    """Require a new directory, fail rather than overwrite any existing audit."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "learning_bundle_audit.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingestion-complete", action="store_true", help="Attest completion log was verified"
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    default_output = Path("/codeload/equity-alpha-audits") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    parser.add_argument(
        "--output-dir", type=Path, default=default_output, help="A NEW derived-output directory"
    )
    args = parser.parse_args(argv)
    if not args.ingestion_complete:
        parser.error("Verify the completed-ingestion flightlog message, then pass --ingestion-complete")
    if not 1 <= args.chunk_size <= 512:
        parser.error("--chunk-size must be between 1 and 512")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    # Import only inside the in-platform command, so pure helpers remain testable
    # offline without installing a different Zipline implementation.
    from quantrocket.zipline import list_bundles
    from zipline.data import bundles
    from zipline.errors import MultipleSymbolsFound, SymbolNotFound
    from zipline.research import use_bundle

    if list_bundles().get(BUNDLE) is not True:
        raise RuntimeError("Learning bundle is not listed as ingested; no audit has been written")
    use_bundle(BUNDLE)  # Registers QuantRocket's extension before bundles.load.
    report = collect_audit(bundles.load(BUNDLE), args.chunk_size, (SymbolNotFound, MultipleSymbolsFound))
    report["ingestion_completion"] = {
        "caller_attested_completion_log_verified": True,
        "api_listed_as_ingested": True,
        "independently_verified_by_this_script": False,
    }
    destination = write_audit(report, args.output_dir)
    print(json.dumps({"derived_audit": str(destination), "model_ready": False}))


if __name__ == "__main__":
    main()
