"""Read-only internal-consistency verification for a frozen wide ML V1 run.

No model, prediction archive, or NumPy array is deserialized. External cache and
registry paths are opened only when explicitly supplied, never discovered from
paths recorded in an untrusted manifest. Hashes establish consistency, not
authorship or immutability, and do not validate the scientific conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path, PurePosixPath

from .integrity import IntegrityError, canonical_digest, file_digest
from .registry import RegistryError, read_registry

SOURCE_FILES = frozenset(
    ("wide_experiment.py", "wide_data.py", "wide_models.py", "wide_stats.py", "registry.py", "integrity.py")
)
DATA_FILES = frozenset(
    (
        "X.npy",
        "y.npy",
        "status.npy",
        "date_index.npy",
        "asset_index.npy",
        "dates.npy",
        "asset_ids.npy",
        "manifest.json",
    )
)
MODEL_NAMES = ("ridge_10", "ridge_100", "hgb_7", "hgb_15", "null_hgb_191", "null_hgb_727", "null_hgb_2027")
EXPECTED_FITS = frozenset((year, name) for year in (2009, 2010) for name in MODEL_NAMES)
MODEL_FILES = frozenset(f"models/{year}_{name}.joblib" for year, name in EXPECTED_FITS)
REQUIRED_FILES = (
    MODEL_FILES
    | {f"source_snapshot/{name}" for name in SOURCE_FILES}
    | {
        "run_started.json",
        "config.json",
        "summary.json",
        "trial_ledger.json",
        "fit_lifecycle.jsonl",
        "training_split_2009.json",
        "training_split_2010.json",
        "private_predictions.npz",
        "daily_forecast_metrics.csv",
        "daily_score_decile_coverage.csv",
        "diagnostics.png",
        "report.html",
    }
)


class WideIntegrityError(ValueError):
    """A wide run is incomplete, inconsistent, or unsafe to inspect."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WideIntegrityError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(value, name):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise WideIntegrityError(f"Nonfinite JSON in {name}")
        return result

    try:
        return json.loads(
            value,
            object_pairs_hook=_object,
            parse_float=finite_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                WideIntegrityError(f"Nonfinite JSON in {name}: {value}")
            ),
        )
    except WideIntegrityError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise WideIntegrityError(f"Invalid JSON: {name}") from exc


def _json(path):
    return _decode(path.read_text(encoding="utf-8"), path.name)


def _safe_name(value):
    if not isinstance(value, str) or not value or any(char in value for char in ("\\", ":", "\x00")):
        raise WideIntegrityError(f"Unsafe artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise WideIntegrityError(f"Unsafe artifact path: {value!r}")
    return value


def _hash_map(value, name):
    if not isinstance(value, dict) or not value:
        raise WideIntegrityError(f"Missing or empty {name}")
    for relative, digest in value.items():
        _safe_name(relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise WideIntegrityError(f"Invalid SHA-256 in {name}: {relative}")
    return value


def _reject_symlink_components(path):
    for component in (path.absolute(), *path.absolute().parents):
        if component.is_symlink():
            raise WideIntegrityError("Symlink forbidden in explicitly supplied path")


def _inventory(root):
    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise WideIntegrityError("Run/cache directory is missing or is a symlink")
    files = set()

    def walk_error(error):
        raise error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        for name in dirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise WideIntegrityError(f"Symlink forbidden: {path.relative_to(root)}")
            if not path.is_dir() and not path.is_file():
                raise WideIntegrityError(f"Nonregular artifact: {path.relative_to(root)}")
        files.update((Path(directory) / name).relative_to(root).as_posix() for name in names)
    return files


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise WideIntegrityError(f"{name} must be a positive integer")
    return value


def _identity(value):
    if not isinstance(value, dict) or type(value.get("year")) is not int:
        raise WideIntegrityError("Invalid fit/trial identity")
    identity = (value["year"], value.get("model"))
    if not isinstance(identity[1], str) or identity not in EXPECTED_FITS:
        raise WideIntegrityError("Unexpected fit/trial identity")
    return identity


def _timestamp(value):
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
        return timestamp
    except (TypeError, ValueError, AttributeError) as exc:
        raise WideIntegrityError("Invalid lifecycle UTC timestamp") from exc


def _day(value):
    try:
        result = date.fromisoformat(value)
        if result.isoformat() != value:
            raise ValueError("date must use YYYY-MM-DD")
        return result
    except (TypeError, ValueError) as exc:
        raise WideIntegrityError("Invalid training split date") from exc


def _verify_trials(root):
    raw = (root / "fit_lifecycle.jsonl").read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise WideIntegrityError("Incomplete lifecycle ledger tail")
    events = [_decode(line, "fit_lifecycle.jsonl") for line in raw.splitlines()]
    if len(events) != 42:
        raise WideIntegrityError("Lifecycle must contain exactly 14 complete fit/forecast triples")
    fits, previous_timestamp = {}, None
    for position in range(0, len(events), 3):
        triple = events[position : position + 3]
        identity = _identity(triple[0])
        if identity in fits:
            raise WideIntegrityError("Duplicate lifecycle fit identity")
        rows = None
        for event, expected_event in zip(
            triple, ("fit_started", "fit_completed", "forecast_completed"), strict=True
        ):
            if _identity(event) != identity or event.get("event") != expected_event:
                raise WideIntegrityError("Invalid lifecycle ordering or failed fit/forecast")
            if any("error" in key.lower() or "fail" in key.lower() for key in event):
                raise WideIntegrityError("Lifecycle contains error/failure metadata")
            timestamp = _timestamp(event.get("timestamp_utc"))
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise WideIntegrityError("Lifecycle timestamps are not nondecreasing")
            previous_timestamp = timestamp
            counts = (
                _positive_int(event.get("train_rows"), "train_rows"),
                _positive_int(event.get("forecast_rows"), "forecast_rows"),
            )
            if rows is not None and rows != counts:
                raise WideIntegrityError("Lifecycle row counts differ within one fit")
            rows = counts
        fits[identity] = rows
    if fits.keys() != EXPECTED_FITS:
        raise WideIntegrityError("Lifecycle fit identities do not match the registered 14 fits")
    trials = _json(root / "trial_ledger.json")
    if not isinstance(trials, list) or len(trials) != 14:
        raise WideIntegrityError("Trial ledger must contain exactly 14 trials")
    seen = set()
    for trial in trials:
        identity = _identity(trial)
        if identity in seen:
            raise WideIntegrityError("Duplicate trial identity")
        seen.add(identity)
        counts = (
            _positive_int(trial.get("rows"), "trial rows"),
            _positive_int(trial.get("forecast_rows"), "trial forecast_rows"),
        )
        if counts != fits[identity]:
            raise WideIntegrityError("Trial/lifecycle row count mismatch")
        if trial.get("label_permutation") is not identity[1].startswith("null_hgb_"):
            raise WideIntegrityError("Trial permutation flag mismatch")
        seconds = trial.get("seconds")
        if (
            type(seconds) not in (int, float)
            or (type(seconds) is float and not math.isfinite(seconds))
            or seconds < 0
        ):
            raise WideIntegrityError("Invalid trial runtime")
    if seen != EXPECTED_FITS:
        raise WideIntegrityError("Trial identities do not match the registered 14 fits")
    for year in (2009, 2010):
        split = _json(root / f"training_split_{year}.json")
        if not isinstance(split, dict) or split.get("training_year") != year:
            raise WideIntegrityError("Invalid training split year")
        selected = _positive_int(split.get("selected_rows"), "training selected_rows")
        if any(fits[(year, name)][0] != selected for name in MODEL_NAMES):
            raise WideIntegrityError("Training split/lifecycle row count mismatch")
        boundary = date(year, 1, 1)
        if (
            split.get("training_boundary_exclusive") != boundary.isoformat()
            or split.get("label_exit_offset") != 6
            or not _day(split.get("min_row_date"))
            <= _day(split.get("max_row_date"))
            <= _day(split.get("max_label_exit_date"))
            < boundary
        ):
            raise WideIntegrityError("Training split maturity/boundary metadata mismatch")


def _verify_registry(path, root, started, inventory_digest):
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise WideIntegrityError("Explicit registry is missing, nonregular, or a symlink")
    events = read_registry(path)
    run = [event for event in events if event["run_id"] == started["run_id"]]
    if [event["event"] for event in run] != ["run_started", "run_completed"]:
        raise WideIntegrityError("Registry requires one started and one completed event for this run")
    if any(event["experiment_id"] != started["config"]["experiment_id"] for event in run):
        raise WideIntegrityError("Registry experiment binding mismatch")
    expected = {
        "config_sha256": started["config_sha256"],
        "data_sha256": started["data_digest"],
        "source_sha256": canonical_digest(started["source_sha256"]),
    }
    if any(run[0]["payload"].get(key) != digest for key, digest in expected.items()):
        raise WideIntegrityError("Registry start binding mismatch")
    if run[0]["payload"].get("output_dir") not in (str(root), str(root.absolute())):
        raise WideIntegrityError("Registry output directory binding mismatch")
    if run[1]["payload"].get("artifacts_sha256") != inventory_digest:
        raise WideIntegrityError("Registry completion/inventory binding mismatch")


def verify_wide_run(run_dir: Path, *, cache_path: Path | None = None, registry_path: Path | None = None):
    """Verify frozen bytes and lifecycle, optionally explicitly supplied externals.

    The result contains aggregate counts/digests only. Omitting ``cache_path`` or
    ``registry_path`` deliberately leaves those external checks unverified; paths
    in ``run_started.json`` are never followed. Models are hashed as opaque bytes.
    """
    try:
        root = Path(run_dir)
        actual = _inventory(root)
        if "FAILURE.json" in actual:
            raise WideIntegrityError("Failed run cannot verify as complete")
        hashes = _hash_map(_json(root / "artifact_hashes.json"), "artifact inventory")
        if "artifact_hashes.json" in hashes:
            raise WideIntegrityError("Artifact inventory cannot hash itself")
        if actual - {"artifact_hashes.json"} != hashes.keys():
            raise WideIntegrityError("Exact artifact inventory mismatch (missing or extra files)")
        if not REQUIRED_FILES.issubset(hashes):
            raise WideIntegrityError("Required wide run artifacts are missing")
        for relative, digest in hashes.items():
            if file_digest(root / relative) != digest:
                raise WideIntegrityError(f"Artifact checksum mismatch: {relative}")
        started, config = _json(root / "run_started.json"), _json(root / "config.json")
        if not isinstance(started, dict) or not isinstance(config, dict):
            raise WideIntegrityError("Started manifest/configuration must be objects")
        if not isinstance(started.get("run_id"), str) or not started["run_id"]:
            raise WideIntegrityError("Missing run identity")
        if config != started.get("config") or canonical_digest(config) != started.get("config_sha256"):
            raise WideIntegrityError("Frozen configuration checksum/binding mismatch")
        if (
            config.get("experiment_id") != "wide_ml_v1"
            or config.get("prediction_years") != [2009, 2010]
            or config.get("models") != list(MODEL_NAMES[:4])
            or config.get("permutation_seeds") != [191, 727, 2027]
            or config.get("max_total_model_fits") != 14
        ):
            raise WideIntegrityError("Configuration differs from the registered wide ML V1 fit grid")
        if (
            config.get("reserved_start") != "2011-01-01"
            or config.get("development_end_exclusive") != "2011-01-01"
            or config.get("reserved_performance_enabled") is not False
            or started.get("reserved_performance_enabled") is not False
            or config.get("early_stopping") is not False
            or config.get("primary_linear") != "ridge_100"
            or config.get("primary_nonlinear") != "hgb_7"
            or config.get("execution_delay_sessions") != 1
            or config.get("label_exit_offset") != 6
            or config.get("horizon_sessions") != 5
        ):
            raise WideIntegrityError("Frozen scope/primary comparison metadata contradicts wide ML V1")
        source = _hash_map(started.get("source_sha256"), "frozen source map")
        snapshots = {name for name in hashes if name.startswith("source_snapshot/")}
        if source.keys() != SOURCE_FILES or snapshots != {f"source_snapshot/{name}" for name in SOURCE_FILES}:
            raise WideIntegrityError("Frozen source inventory must contain the exact six source files")
        if any(hashes[f"source_snapshot/{name}"] != digest for name, digest in source.items()):
            raise WideIntegrityError("Frozen source checksum/binding mismatch")
        if {name for name in hashes if name.startswith("models/")} != MODEL_FILES:
            raise WideIntegrityError("Model file inventory must contain the exact 14 opaque model files")
        data = _hash_map(started.get("data_sha256"), "bound cache hashes")
        if data.keys() != DATA_FILES or canonical_digest(data) != started.get("data_digest"):
            raise WideIntegrityError("Bound cache inventory/digest mismatch")
        _verify_trials(root)
        summary = _json(root / "summary.json")
        if not isinstance(summary, dict) or type(summary.get("completed_model_fits")) is not int:
            raise WideIntegrityError("Missing summary fit count")
        if summary["completed_model_fits"] != 14:
            raise WideIntegrityError("Summary fit count differs from the verified lifecycle")
        for flag in (
            "model_or_trading_promotion_authorized",
            "net_sharpe_computed",
            "reserved_performance_computed",
        ):
            if summary.get(flag) is not False:
                raise WideIntegrityError(f"Summary must explicitly disallow {flag}")
        if cache_path is not None:
            cache = Path(cache_path)
            cache_files = _inventory(cache)
            if not DATA_FILES.issubset(cache_files):
                raise WideIntegrityError("Explicit cache is missing bound files")
            for relative, digest in data.items():
                if file_digest(cache / relative) != digest:
                    raise WideIntegrityError(f"Explicit cache checksum mismatch: {relative}")
        inventory_digest = canonical_digest(hashes)
        if registry_path is not None:
            _verify_registry(Path(registry_path), root, started, inventory_digest)
        return {
            "status": "verified",
            "run_id": started["run_id"],
            "experiment_id": config["experiment_id"],
            "artifact_count": len(hashes),
            "source_file_count": len(source),
            "model_file_count": len(MODEL_FILES),
            "completed_model_fits": 14,
            "bound_cache_file_count": len(data),
            "cache_verified": cache_path is not None,
            "registry_verified": registry_path is not None,
            "config_sha256": started["config_sha256"],
            "data_digest": started["data_digest"],
            "source_digest": canonical_digest(source),
            "artifact_inventory_digest": inventory_digest,
            "assurance": "internal_consistency_only_not_authorship_scientific_validation_or_promotion",
        }
    except (IntegrityError, RegistryError, OSError, UnicodeError) as exc:
        raise WideIntegrityError(str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_wide_run(args.run, cache_path=args.cache, registry_path=args.registry)
    except WideIntegrityError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, allow_nan=False))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
