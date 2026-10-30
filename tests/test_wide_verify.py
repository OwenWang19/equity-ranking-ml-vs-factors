"""Synthetic frozen wide-run checks; no real cache or executable model is read."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from equity_alpha.integrity import canonical_digest, file_digest
from equity_alpha.registry import complete_run, start_run
from equity_alpha.wide_verify import (
    DATA_FILES,
    EXPECTED_FITS,
    REQUIRED_FILES,
    SOURCE_FILES,
    WideIntegrityError,
    main,
    verify_wide_run,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh(run):
    hashes = {
        path.relative_to(run).as_posix(): file_digest(path)
        for path in run.rglob("*")
        if path.is_file() and path != run / "artifact_hashes.json"
    }
    _write_json(run / "artifact_hashes.json", hashes)
    return hashes


def _registry(path, run, *, start_changes=None, completion_digest=None):
    started = json.loads((run / "run_started.json").read_text())
    bindings = {
        "config_sha256": started["config_sha256"],
        "data_sha256": started["data_digest"],
        "source_sha256": canonical_digest(started["source_sha256"]),
        "output_dir": str(run),
    }
    bindings.update(start_changes or {})
    start_run(path, "wide_ml_v1", started["run_id"], **bindings)
    complete_run(
        path,
        "wide_ml_v1",
        started["run_id"],
        artifacts_sha256=completion_digest
        or canonical_digest(json.loads((run / "artifact_hashes.json").read_text())),
    )


@pytest.fixture
def wide_run(tmp_path):
    run, cache, registry = tmp_path / "run", tmp_path / "cache", tmp_path / "registry.jsonl"
    run.mkdir()
    cache.mkdir()
    # Invalid pickle/NPZ bytes deliberately prove these artifacts are never loaded.
    for name in REQUIRED_FILES:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"Opaque synthetic artifact: {name}\n".encode())
    for name in DATA_FILES | {"features_wide.npy", "y_wide.npy"}:
        (cache / name).write_bytes(f"Opaque synthetic cache bytes: {name}\n".encode())
    config = {
        "experiment_id": "wide_ml_v1",
        "prediction_years": [2009, 2010],
        "models": ["ridge_10", "ridge_100", "hgb_7", "hgb_15"],
        "permutation_seeds": [191, 727, 2027],
        "max_total_model_fits": 14,
        "reserved_start": "2011-01-01",
        "development_end_exclusive": "2011-01-01",
        "reserved_performance_enabled": False,
        "early_stopping": False,
        "primary_linear": "ridge_100",
        "primary_nonlinear": "hgb_7",
        "execution_delay_sessions": 1,
        "label_exit_offset": 6,
        "horizon_sessions": 5,
    }
    source = {name: file_digest(run / "source_snapshot" / name) for name in SOURCE_FILES}
    data = {name: file_digest(cache / name) for name in DATA_FILES}
    started = {
        "run_id": "wide_synthetic",
        "config": config,
        "config_sha256": canonical_digest(config),
        "source_sha256": source,
        "data_sha256": data,
        "data_digest": canonical_digest(data),
        "data_cache_path": "/this/recorded/path/must/not/be/opened",
        "reserved_performance_enabled": False,
    }
    _write_json(run / "run_started.json", started)
    _write_json(run / "config.json", config)
    _write_json(
        run / "summary.json",
        {
            "completed_model_fits": 14,
            "model_or_trading_promotion_authorized": False,
            "net_sharpe_computed": False,
            "reserved_performance_computed": False,
        },
    )
    events, trials = [], []
    timestamp = datetime(2026, 10, 30, tzinfo=UTC)
    for year, name in sorted(EXPECTED_FITS):
        for event in ("fit_started", "fit_completed", "forecast_completed"):
            events.append(
                {
                    "event": event,
                    "year": year,
                    "model": name,
                    "train_rows": 10000,
                    "forecast_rows": 1234,
                    "timestamp_utc": timestamp.isoformat(),
                }
            )
            timestamp += timedelta(seconds=1)
        trials.append(
            {
                "year": year,
                "model": name,
                "rows": 10000,
                "forecast_rows": 1234,
                "seconds": 0.5,
                "label_permutation": name.startswith("null_hgb_"),
            }
        )
    (run / "fit_lifecycle.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    _write_json(run / "trial_ledger.json", trials)
    for year in (2009, 2010):
        _write_json(
            run / f"training_split_{year}.json",
            {
                "training_year": year,
                "selected_rows": 10000,
                "training_boundary_exclusive": f"{year}-01-01",
                "label_exit_offset": 6,
                "min_row_date": "2007-07-05",
                "max_row_date": f"{year - 1}-12-22",
                "max_label_exit_date": f"{year - 1}-12-31",
            },
        )
    _refresh(run)
    _registry(registry, run)
    return run, cache, registry


def test_complete_read_only_verification_and_explicit_optional_paths(wide_run):
    run, cache, registry = wide_run
    before = {path: (path.stat().st_mtime_ns, file_digest(path)) for path in run.rglob("*") if path.is_file()}
    result = verify_wide_run(run)
    assert result["status"] == "verified"
    assert result["artifact_count"] == 32
    assert result["source_file_count"] == 6
    assert result["model_file_count"] == result["completed_model_fits"] == 14
    assert result["cache_verified"] is result["registry_verified"] is False
    result = verify_wide_run(run, cache_path=cache, registry_path=registry)
    assert result["cache_verified"] is result["registry_verified"] is True
    assert before == {
        path: (path.stat().st_mtime_ns, file_digest(path)) for path in run.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("action", ["tamper", "missing", "extra", "failure", "missing_hashes", "self_hash"])
def test_artifact_inventory_failures(wide_run, action):
    run, _, _ = wide_run
    if action == "tamper":
        (run / "models/2009_hgb_7.joblib").write_bytes(b"tampered opaque bytes")
    elif action == "missing":
        (run / "report.html").unlink()
    elif action == "extra":
        (run / "extra.txt").write_text("not inventoried")
    elif action == "failure":
        _write_json(run / "FAILURE.json", {})
        _refresh(run)
    elif action == "missing_hashes":
        (run / "artifact_hashes.json").unlink()
    else:
        hashes = json.loads((run / "artifact_hashes.json").read_text())
        hashes["artifact_hashes.json"] = "a" * 64
        _write_json(run / "artifact_hashes.json", hashes)
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run)


@pytest.mark.parametrize(
    "unsafe", ["../secret", "/etc/passwd", "./report.html", "x//y", "x\\y", "C:/x", "x\x00y"]
)
def test_unsafe_inventory_keys(wide_run, unsafe):
    run, _, _ = wide_run
    hashes = json.loads((run / "artifact_hashes.json").read_text())
    hashes[unsafe] = "a" * 64
    _write_json(run / "artifact_hashes.json", hashes)
    with pytest.raises(WideIntegrityError, match="Unsafe"):
        verify_wide_run(run)


@pytest.mark.parametrize("target", ["root", "file", "directory", "cache", "registry", "ancestor"])
def test_symlinks_are_rejected(wide_run, tmp_path, target):
    run, cache, registry = wide_run
    if target == "root":
        linked = tmp_path / "linked-run"
        linked.symlink_to(run, target_is_directory=True)
        run = linked
    elif target == "file":
        (run / "alias").symlink_to(run / "report.html")
    elif target == "directory":
        (run / "linked-dir").symlink_to(cache, target_is_directory=True)
    elif target == "cache":
        (cache / "unbound-alias").symlink_to(cache / "X.npy")
    elif target == "registry":
        linked = tmp_path / "linked-registry"
        linked.symlink_to(registry)
        registry = linked
    else:
        linked = tmp_path / "linked-parent"
        linked.symlink_to(tmp_path, target_is_directory=True)
        run = linked / "run"
    with pytest.raises(WideIntegrityError, match="[Ss]ymlink"):
        verify_wide_run(run, cache_path=cache, registry_path=registry)


@pytest.mark.parametrize("target", ["config", "source", "source_extra", "data", "model_missing", "summary"])
def test_independent_manifest_bindings_survive_rebuilt_artifact_inventory(wide_run, target):
    run, _, _ = wide_run
    if target == "config":
        _write_json(run / "config.json", {"changed": True})
    elif target == "source":
        (run / "source_snapshot/wide_data.py").write_text("revised after freezing")
    elif target == "source_extra":
        (run / "source_snapshot/extra.py").write_text("not one of the six files")
    elif target == "data":
        started = json.loads((run / "run_started.json").read_text())
        started["data_sha256"]["y.npy"] = "a" * 64
        _write_json(run / "run_started.json", started)
    elif target == "model_missing":
        (run / "models/2009_hgb_7.joblib").unlink()
    else:
        _write_json(run / "summary.json", {"completed_model_fits": 13})
    _refresh(run)
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run)


@pytest.mark.parametrize("target", ["count", "failed", "duplicate", "order", "rows", "bool", "time", "tail"])
def test_lifecycle_failures_even_when_hashes_are_refreshed(wide_run, target):
    run, _, _ = wide_run
    path = run / "fit_lifecycle.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if target == "count":
        events.pop()
    elif target == "failed":
        events[1]["event"] = "fit_or_forecast_failed"
    elif target == "duplicate":
        events[3:6] = events[:3]
    elif target == "order":
        events[0], events[1] = events[1], events[0]
    elif target == "rows":
        events[1]["forecast_rows"] += 1
    elif target == "bool":
        events[0]["train_rows"] = True
    elif target == "time":
        events[1]["timestamp_utc"] = "2000-01-01T00:00:00+00:00"
    raw = "".join(json.dumps(event) + "\n" for event in events)
    path.write_text(raw.rstrip("\n") if target == "tail" else raw)
    _refresh(run)
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run)


@pytest.mark.parametrize("target", ["count", "duplicate", "rows", "permutation", "split"])
def test_trial_ledger_and_training_split_consistency(wide_run, target):
    run, _, _ = wide_run
    path = run / "trial_ledger.json"
    trials = json.loads(path.read_text())
    if target == "count":
        trials.pop()
    elif target == "duplicate":
        trials[1] = trials[0]
    elif target == "rows":
        trials[0]["rows"] += 1
    elif target == "permutation":
        trials[0]["label_permutation"] = not trials[0]["label_permutation"]
    else:
        _write_json(run / "training_split_2009.json", {"training_year": 2009, "selected_rows": 9999})
    _write_json(path, trials)
    _refresh(run)
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run)


@pytest.mark.parametrize("target", ["modified", "missing", "missing_path"])
def test_only_explicit_cache_is_opened_and_must_match(wide_run, target):
    run, cache, _ = wide_run
    if target == "modified":
        (cache / "y.npy").write_bytes(b"revision")
    elif target == "missing":
        (cache / "manifest.json").unlink()
    else:
        cache = cache / "does-not-exist"
    assert verify_wide_run(run)["cache_verified"] is False
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run, cache_path=cache)


@pytest.mark.parametrize("target", ["start", "completion", "missing", "chain", "output"])
def test_explicit_registry_binding_failures(wide_run, tmp_path, target):
    run, _, registry = wide_run
    if target in {"start", "completion", "output"}:
        registry = tmp_path / "different.jsonl"
        changes = {"config_sha256": "a" * 64} if target == "start" else None
        if target == "output":
            changes = {"output_dir": "/not/the/explicit/run"}
        _registry(
            registry,
            run,
            start_changes=changes,
            completion_digest="a" * 64 if target == "completion" else None,
        )
    elif target == "missing":
        registry = tmp_path / "missing.jsonl"
    else:
        registry.write_text(registry.read_text().replace('"sequence":1', '"sequence":9', 1))
    assert verify_wide_run(run)["registry_verified"] is False
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run, registry_path=registry)


def test_duplicate_and_nonfinite_json_fail_closed(wide_run):
    run, _, _ = wide_run
    (run / "artifact_hashes.json").write_text('{"x":"a", "x":"b"}')
    with pytest.raises(WideIntegrityError, match="Duplicate"):
        verify_wide_run(run)
    (run / "artifact_hashes.json").write_text('{"x": NaN}')
    with pytest.raises(WideIntegrityError, match="Nonfinite"):
        verify_wide_run(run)
    (run / "artifact_hashes.json").write_text('{"x": 1e999}')
    with pytest.raises(WideIntegrityError, match="Nonfinite"):
        verify_wide_run(run)


@pytest.mark.parametrize(
    "flag", ["model_or_trading_promotion_authorized", "net_sharpe_computed", "reserved_performance_computed"]
)
def test_summary_cannot_claim_holdout_trading_or_sharpe_release(wide_run, flag):
    run, _, _ = wide_run
    summary = json.loads((run / "summary.json").read_text())
    summary[flag] = True
    _write_json(run / "summary.json", summary)
    _refresh(run)
    with pytest.raises(WideIntegrityError, match="Summary must explicitly disallow"):
        verify_wide_run(run)


@pytest.mark.parametrize("target", ["reserved", "primary", "started", "maturity", "boundary", "offset"])
def test_contradictory_frozen_scope_or_maturity_cannot_verify(wide_run, target):
    run, _, _ = wide_run
    started = json.loads((run / "run_started.json").read_text())
    if target in {"reserved", "primary"}:
        config = started["config"]
        config["reserved_performance_enabled" if target == "reserved" else "primary_nonlinear"] = (
            True if target == "reserved" else "hgb_15"
        )
        started["config_sha256"] = canonical_digest(config)
        _write_json(run / "config.json", config)
    elif target == "started":
        started["reserved_performance_enabled"] = True
    else:
        path = run / "training_split_2009.json"
        split = json.loads(path.read_text())
        if target == "maturity":
            split["max_label_exit_date"] = "2009-01-01"
        elif target == "boundary":
            split["training_boundary_exclusive"] = "2011-01-01"
        else:
            split["label_exit_offset"] = 5
        _write_json(path, split)
    _write_json(run / "run_started.json", started)
    _refresh(run)
    with pytest.raises(WideIntegrityError):
        verify_wide_run(run)


def test_cli_emits_aggregate_json_and_failure_exit(wide_run, capsys):
    run, cache, registry = wide_run
    assert main(["--run", str(run), "--cache", str(cache), "--registry", str(registry)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["registry_verified"] is True
    (run / "report.html").write_text("tampered")
    assert main(["--run", str(run)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
