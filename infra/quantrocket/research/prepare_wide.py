"""Build a bounded, platform-only development matrix from the learning bundle.

The 2011 raw sessions are excluded before any feature or label calculation.
Only persistent security IDs and past/current observations determine the dynamic
universe; latest symbols, classifications and lifetime/end metadata are unused.
All row-level derived caches must remain inside the QuantRocket platform.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

AUTHORIZED_BUNDLE = "usstock-learn-1d"
RESERVED_START = pd.Timestamp("2011-01-01")
AUTHORIZED_START = pd.Timestamp("2007-01-03")
PLATFORM_SOURCE = Path("/codeload/equity-alpha-src")
PLATFORM_OUTPUT = Path("/codeload/equity-alpha-wide")
RAW_FIELDS = ("open", "high", "low", "close", "volume")
N_FEATURES = 12


def validate_config(config):
    """Fail closed on authorized scope and bounded resource/universe settings."""
    if config["bundle"] != AUTHORIZED_BUNDLE:
        raise ValueError("Only the authorized learning bundle is supported")
    end = pd.Timestamp(config["development_end_exclusive"])
    start = pd.Timestamp(config["authorized_start"])
    reserved = pd.Timestamp(config["reserved_start"])
    if start != AUTHORIZED_START or reserved != RESERVED_START or end != RESERVED_START:
        raise ValueError("Development must stop strictly before reserved 2011 sessions")
    for key in (
        "chunk_size",
        "max_assets_per_date",
        "warmup_sessions",
        "max_cache_bytes",
        "min_free_disk_bytes",
    ):
        if isinstance(config[key], bool) or int(config[key]) != config[key] or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["chunk_size"] > 256 or config["max_assets_per_date"] > 2500:
        raise ValueError("Chunk size and daily universe exceed authorized limits")
    if config["max_cache_bytes"] > 2_000_000_000 or config["min_free_disk_bytes"] < 8_000_000_000:
        raise ValueError("Cache cap/free-disk floor exceeds authorized resource bounds")
    for key in ("min_price", "min_adv20"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(config["bundle_reference_hash"], str) or len(config["bundle_reference_hash"]) != 64:
        raise ValueError("A prior bundle reference SHA-256 is required")
    return config


def development_sessions(values, config):
    """Return only authorized raw-session labels before any reader calculation."""
    sessions = pd.DatetimeIndex(values)
    if sessions.tz is not None:
        sessions = sessions.tz_convert("UTC").tz_localize(None)
    if sessions.empty or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("Sessions must be nonempty, unique and increasing")
    if sessions[0] < pd.Timestamp(config["authorized_start"]) or sessions[-1] > pd.Timestamp("2011-12-31"):
        raise ValueError("Bundle is outside the authorized 2007-2011 learning scope")
    selected = sessions[sessions < pd.Timestamp(config["development_end_exclusive"])]
    if len(selected) <= config["warmup_sessions"]:
        raise ValueError("Insufficient development sessions after scope truncation")
    return selected


def persistent_asset_order(assets):
    """Retain every asset exactly once, sorting only by persistent identity."""
    assets = list(assets)
    identities = [str(asset.real_sid) for asset in assets]
    if not assets or any(not sid or sid == "None" for sid in identities):
        raise ValueError("Every asset requires a persistent security ID")
    if len(set(identities)) != len(identities) or len({int(a.sid) for a in assets}) != len(assets):
        raise ValueError("Duplicate persistent or bundle-local security IDs")
    return sorted(assets, key=lambda asset: str(asset.real_sid))


def causal_eligibility(history_valid, current_close, adv20, config):
    """No target/status/end-date input is accepted by universe selection."""
    history_valid = np.asarray(history_valid, dtype=bool)
    close = np.asarray(current_close)
    adv = np.asarray(adv20)
    if history_valid.shape != close.shape or close.shape != adv.shape or close.ndim != 2:
        raise ValueError("Eligibility inputs must share [date, asset] shape")
    eligible = history_valid & np.isfinite(close) & (close >= config["min_price"])
    eligible &= np.isfinite(adv) & (adv >= config["min_adv20"])
    eligible[: config["warmup_sessions"]] = False
    return eligible


def select_daily_assets(eligible, adv20, persistent_sids, maximum):
    """ADV descending, stable-ID ascending ties, output in stable-ID order."""
    eligible = np.asarray(eligible, dtype=bool)
    adv20 = np.asarray(adv20)
    ids = np.asarray(persistent_sids)
    if eligible.ndim != 1 or eligible.shape != adv20.shape or ids.shape != eligible.shape:
        raise ValueError("Daily selection inputs must share one-dimensional shape")
    if maximum < 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid cap or duplicate persistent IDs")
    candidates = np.flatnonzero(eligible)
    if not np.isfinite(adv20[candidates]).all():
        raise ValueError("Eligible ADV must be finite")
    order = np.lexsort((ids[candidates], -adv20[candidates]))
    chosen = candidates[order[:maximum]]
    return chosen[np.argsort(ids[chosen], kind="stable")]


def rank_feature_cross_section(values):
    """Average ties, per-date selected cohort only; singleton/constant maps to 0."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Selected features must be a finite matrix")
    if len(values) <= 1:
        return np.zeros(values.shape, dtype=np.float32)
    ranks = pd.DataFrame(values).rank(method="average", axis=0).to_numpy()
    return ((ranks - 1.0) / (len(values) - 1.0) - 0.5).astype(np.float32)


def estimated_cache_bytes(n_dates, n_assets, n_features, maximum):
    """Conservative .npy payload/header + metadata estimate, including wide caches."""
    rows = n_dates * min(n_assets, maximum)
    wide = n_dates * n_assets * (4 * n_features + 4 + 1 + 4 + 1)
    flat = rows * (4 * n_features + 4 + 1 + 4 + 4)
    return int(wide + flat + n_dates * 8 + n_assets * 1024 + 2_000_000)


def check_resources(output_dir, estimated_bytes, config, disk_usage=shutil.disk_usage):
    """Check before making a new directory or allocating any data cache."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"Output directory must be new: {target}")
    if estimated_bytes > config["max_cache_bytes"]:
        raise ValueError("Estimated derived cache exceeds max_cache_bytes")
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    free = disk_usage(parent).free
    if free - estimated_bytes < config["min_free_disk_bytes"]:
        raise OSError("Insufficient disk: projected cache would breach the free-space floor")
    return int(free)


def _event_tables(reader, sessions):
    """Read local tables only; remove reserved/future events before core calls."""
    tables = {}
    end_seconds = pd.Timestamp(sessions[-1]).timestamp()
    for table in ("splits", "dividends", "mergers", "dividend_payouts"):
        frame = reader.get_df_from_table(table)
        date_column = "ex_date" if table == "dividend_payouts" else "effective_date"
        tables[table] = frame.loc[frame[date_column] <= end_seconds].copy()
    return tables


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare_wide(bundle_data, config, output_dir, core, source_hashes=None, disk_usage=shutil.disk_usage):
    """Prepare all-ID wide caches then flatten causal, dynamically selected rows."""
    validate_config(config)
    reader = bundle_data.equity_daily_bar_reader
    sessions = development_sessions(reader.sessions, config)
    finder = bundle_data.asset_finder
    all_sids = list(finder.equities_sids)
    assets = persistent_asset_order(finder.retrieve_all(all_sids))
    if len(assets) != len(all_sids) or {int(a.sid) for a in assets} != set(all_sids):
        raise ValueError("Asset retrieval must preserve the entire learning bundle")
    if len(core.FEATURE_NAMES) != N_FEATURES:
        raise ValueError("Expected exactly twelve registered feature names")
    n_dates, n_assets, n_features = len(sessions), len(assets), len(core.FEATURE_NAMES)
    estimate = estimated_cache_bytes(n_dates, n_assets, n_features, config["max_assets_per_date"])
    free_before = check_resources(output_dir, estimate, config, disk_usage)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    specs = {
        "features_wide": (np.float32, (n_dates, n_assets, n_features)),
        "adv20_wide": (np.float32, (n_dates, n_assets)),
        "eligible_wide": (np.bool_, (n_dates, n_assets)),
        "y_wide": (np.float32, (n_dates, n_assets)),
        "status_wide": (np.uint8, (n_dates, n_assets)),
    }
    for name, (dtype, shape) in specs.items():
        arrays[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    tables = _event_tables(bundle_data.adjustment_reader, sessions)
    adjustment_audits = []
    for offset in range(0, n_assets, config["chunk_size"]):
        chunk = assets[offset : offset + config["chunk_size"]]
        sids = [int(asset.sid) for asset in chunk]
        raw_arrays = reader.load_raw_arrays(list(RAW_FIELDS), sessions[0], sessions[-1], sids)
        raw = dict(zip(RAW_FIELDS, raw_arrays, strict=True))
        if any(np.asarray(value).shape != (n_dates, len(chunk)) for value in raw.values()):
            raise ValueError("Reader returned raw rows outside the development window")
        chunk_tables = {name: frame.loc[frame.sid.isin(sids)] for name, frame in tables.items()}
        qprice, qsplit, bad, audit = core.adjustment_factors(
            sessions,
            sids,
            chunk_tables["splits"],
            chunk_tables["dividends"],
            chunk_tables["mergers"],
            chunk_tables["dividend_payouts"],
        )
        result = core.features_and_labels(raw, sessions, qprice, qsplit, bad)
        shape = (n_dates, len(chunk))
        for key in ("adv20", "history_valid", "label_proxy", "label_status"):
            if np.asarray(result[key]).shape != shape:
                raise ValueError(f"Unexpected {key} result shape")
        if np.asarray(result["features"]).shape != (*shape, n_features):
            raise ValueError("Unexpected feature tensor shape")
        if np.any(np.asarray(result["label_status"]) >= len(core.LABEL_STATUS_NAMES)):
            raise ValueError("Unknown label status code")
        column = slice(offset, offset + len(chunk))
        arrays["features_wide"][:, column] = result["features"]
        arrays["adv20_wide"][:, column] = result["adv20"]
        arrays["eligible_wide"][:, column] = causal_eligibility(
            result["history_valid"],
            raw["close"],
            result["adv20"],
            config,
        )
        arrays["y_wide"][:, column] = result["label_proxy"]
        arrays["status_wide"][:, column] = result["label_status"]
        adjustment_audits.append({"chunk_offset": offset, "asset_count": len(chunk), "audit": audit})
        del raw_arrays, raw, result, qprice, qsplit, bad
    for array in arrays.values():
        array.flush()
    ids = np.asarray([str(asset.real_sid) for asset in assets])
    eligible_counts = np.asarray(arrays["eligible_wide"].sum(axis=1), dtype=np.int64)
    selected_counts = np.minimum(eligible_counts, config["max_assets_per_date"])
    n_rows = int(selected_counts.sum())
    if n_rows == 0:
        raise ValueError("No eligible development rows; no model matrix was produced")
    flat_specs = {
        "X": (np.float32, (n_rows, n_features)),
        "y": (np.float32, (n_rows,)),
        "status": (np.uint8, (n_rows,)),
        "date_index": (np.int32, (n_rows,)),
        "asset_index": (np.int32, (n_rows,)),
    }
    flat = {
        name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in flat_specs.items()
    }
    status_counts = np.zeros(len(core.LABEL_STATUS_NAMES), dtype=np.int64)
    per_date_status = np.zeros((n_dates, len(core.LABEL_STATUS_NAMES)), dtype=np.int64)
    cursor = 0
    for date in range(n_dates):
        chosen = select_daily_assets(
            arrays["eligible_wide"][date], arrays["adv20_wide"][date], ids, config["max_assets_per_date"]
        )
        end = cursor + len(chosen)
        if len(chosen):
            flat["X"][cursor:end] = rank_feature_cross_section(arrays["features_wide"][date, chosen])
            flat["y"][cursor:end] = arrays["y_wide"][date, chosen]
            statuses = arrays["status_wide"][date, chosen]
            flat["status"][cursor:end] = statuses
            flat["date_index"][cursor:end] = date
            flat["asset_index"][cursor:end] = chosen
            per_date_status[date] = np.bincount(statuses, minlength=len(core.LABEL_STATUS_NAMES))
            status_counts += per_date_status[date]
        cursor = end
    for array in flat.values():
        array.flush()
    np.save(output / "dates.npy", sessions.to_numpy(dtype="datetime64[D]"), allow_pickle=False)
    np.save(output / "asset_ids.npy", ids, allow_pickle=False)
    actual_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if actual_bytes > config["max_cache_bytes"]:
        raise OSError("Actual derived cache exceeds the configured cap")
    if disk_usage(output).free < config["min_free_disk_bytes"]:
        raise OSError("Free-space floor breached during preparation")
    manifest = {
        "schema_version": 1,
        "preparation_complete": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bundle": config["bundle"],
        "config": config,
        "bundle_reference": {
            "sha256": config["bundle_reference_hash"],
            "kind": "prior_audit_reference_not_recomputed",
        },
        "source_hashes": source_hashes or {},
        "scope": {
            "learning_only": True,
            "reserved_2011_raw_read": False,
            "first_session": str(sessions[0].date()),
            "last_session": str(sessions[-1].date()),
            "development_end_exclusive": config["development_end_exclusive"],
            "raw_exported": False,
            "row_level_derived_cache_platform_only": True,
            "model_fitted": False,
            "2011_performance_computed": False,
        },
        "dimensions": {"dates": n_dates, "assets": n_assets, "features": n_features, "rows": n_rows},
        "feature_names": list(core.FEATURE_NAMES),
        "label_status_names": list(core.LABEL_STATUS_NAMES),
        "persistent_sids": ids.tolist(),
        "files": {name: f"{name}.npy" for name in (*specs, *flat_specs, "dates", "asset_ids")},
        "universe": {
            "definition": "All bundle IDs; current close, trailing ADV20, valid past history; daily ADV cap",
            "static_classification_used": False,
            "future_lifetimes_used": False,
            "historical_common_stock_or_exchange_qualification": "not_claimed",
            "tie_break": "ADV descending then persistent SID ascending",
            "row_order": "date ascending then persistent SID ascending",
            "eligibility_observes_labels_or_status": False,
            "eligible_counts_per_date": eligible_counts.tolist(),
            "selected_counts_per_date": selected_counts.tolist(),
        },
        "feature_proof": {
            "availability": "close-of-session t; only observations <=t",
            "cross_sectional_transform": "selected-date average rank to [-0.5,0.5]",
            "targets_ranked": False,
            "missing_targets_retained": True,
            "right_boundary_predictions_retained": True,
            "source": "equity_alpha.wide_data; hashed source supplied at execution",
        },
        "label_coverage": {
            "selected_status_counts": dict(zip(core.LABEL_STATUS_NAMES, status_counts.tolist(), strict=True)),
            "selected_status_counts_per_date": per_date_status.tolist(),
        },
        "adjustment_audits": adjustment_audits,
        "resources": {
            "estimated_cache_bytes": estimate,
            "actual_array_bytes": actual_bytes,
            "free_bytes_before": free_before,
            "max_cache_bytes": config["max_cache_bytes"],
            "min_free_disk_bytes": config["min_free_disk_bytes"],
        },
        "limitations": [
            "Security type/exchange metadata is not verified point-in-time and is not selected on.",
            "Labels are an adjusted price proxy, not a certified complete terminal-wealth return.",
            "Missing exit/action labels remain explicit statuses; they are not silently filled.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > config["max_cache_bytes"]:
        raise OSError("Final derived cache including manifest exceeds the configured cap")
    if disk_usage(output).free < config["min_free_disk_bytes"]:
        raise OSError("Free-space floor breached after writing the manifest")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    output = args.output_dir
    if output.parent != PLATFORM_OUTPUT or not output.name.startswith("data-"):
        parser.error("Output must be a new /codeload/equity-alpha-wide/data-<timestamp> directory")
    if not PLATFORM_SOURCE.is_dir():
        parser.error("Run inside the platform with /codeload/equity-alpha-src mounted; do not pip-install")
    sys.path.insert(0, str(PLATFORM_SOURCE))
    from zipline.data import bundles
    from zipline.research import use_bundle

    from equity_alpha import wide_data

    source = Path(wide_data.__file__).resolve()
    if not source.is_relative_to(PLATFORM_SOURCE):
        raise RuntimeError("Core module must load from the mounted platform source")
    hashes = {
        "prepare_wide.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wide_data.py": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    use_bundle(AUTHORIZED_BUNDLE)
    manifest = prepare_wide(bundles.load(AUTHORIZED_BUNDLE), config, output, wide_data, hashes)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "dimensions": manifest["dimensions"],
                "scope": manifest["scope"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
