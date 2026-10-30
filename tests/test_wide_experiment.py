"""Orchestration checks without private vendor data, model fitting or Docker."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_alpha import wide_data
from equity_alpha.integrity import file_digest
from equity_alpha.registry import fail_run, start_run
from equity_alpha.wide_data import FEATURE_NAMES, LABEL_STATUS_NAMES
from equity_alpha.wide_experiment import (
    append_fit_event,
    enforce_trial_budget,
    evaluate_predictions,
    finite_json,
    load_cache,
    summarize_results,
    validate_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/wide_ml_v1.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


@pytest.mark.parametrize(
    "change",
    [
        {"prediction_years": [2009, 2010, 2011]},
        {"reserved_performance_enabled": True},
        {"reserved_start": "2012-01-01"},
        {"early_stopping": True},
        {"primary_nonlinear": "hgb_15"},
        {"max_total_model_fits": 15},
        {"threads": 8},
        {"bootstrap_samples": 200},
    ],
)
def test_cannot_silently_expand_protocol(change):
    spec = config()
    spec.update(change)
    with pytest.raises(ValueError):
        validate_config(spec)


def test_actual_protocol_valid_and_finite_json_keeps_unknown_as_null():
    assert validate_config(config())["reserved_performance_enabled"] is False
    value = finite_json({"not_known": np.nan, "count": np.int64(3), "vector": np.array([np.inf, 1])})
    assert value == {"not_known": None, "count": 3, "vector": [None, 1.0]}
    json.dumps(value, allow_nan=False)


def test_prediction_cohort_never_shrinks_on_missing_future_outcomes():
    dates = pd.to_datetime(["2009-01-02", "2009-01-05", "2009-01-06"])
    row_dates = dates.repeat(4)
    scores = np.tile(np.arange(4, dtype=float), (3, 1)).ravel()[:, None]
    labels = np.tile(np.arange(4, dtype=float), 3)
    labels[[3, 8, 9, 10, 11]] = np.nan
    statuses = np.where(np.isfinite(labels), 0, 2)
    statuses[8:] = 4
    daily, deciles = evaluate_predictions(scores, row_dates, labels, statuses, dates, ["model"])
    assert daily.predicted_rows.sum() == 12
    assert daily.observed_rows.sum() == 7
    assert daily.right_boundary_rows.tolist() == [0, 0, 4]
    assert deciles.predicted_rows.sum() == 12
    assert deciles.boundary_rows.sum() == 4
    assert daily.iloc[0].kendall_tau_a_lower <= daily.iloc[0].kendall_tau_a_upper


def test_all_missing_nonboundary_date_is_not_hidden_from_kendall_bounds():
    spec = config()
    dates = pd.bdate_range("2009-01-01", periods=60)
    row_dates = dates.repeat(4)
    names = spec["models"] + [f"null_hgb_{s}" for s in spec["permutation_seeds"]] + spec["baselines"]
    scores = np.tile(np.arange(4, dtype=float), 60)[:, None] * np.ones((1, len(names)))
    labels = np.tile(np.arange(4, dtype=float), 60)
    labels[4:8] = np.nan
    statuses = np.where(np.isfinite(labels), 0, 2)
    daily, deciles = evaluate_predictions(scores, row_dates, labels, statuses, dates, names)
    summary = summarize_results(daily, deciles, dates, spec)
    stats = summary["models"]["hgb_7"]
    assert stats["forecast_rows"] == 240
    assert stats["mean_full_cohort_kendall_lower_nonboundary"] == pytest.approx(58 / 60)
    assert stats["mean_full_cohort_kendall_upper_nonboundary"] == pytest.approx(1)
    assert summary["primary_paired_ic_difference"]["calendar_dates"] == 60
    assert summary["primary_paired_ic_difference"]["missing_pair_dates"] == 1
    assert not summary["reserved_performance_computed"]
    assert not summary["model_or_trading_promotion_authorized"]


def write_cache(tmp_path, spec):
    n = 6
    values = {
        "X": np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32),
        "y": np.array([1, 2, np.nan, 1, 2, 3], dtype=np.float32),
        "status": np.array([0, 0, 2, 0, 0, 0], dtype=np.uint8),
        "date_index": np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        "asset_index": np.tile(np.arange(3, dtype=np.int32), 2),
        "dates": np.array(["2010-12-29", "2010-12-30"], dtype="datetime64[D]"),
        "asset_ids": np.array(["ID_A", "ID_B", "ID_C"]),
    }
    for name, array in values.items():
        np.save(tmp_path / f"{name}.npy", array, allow_pickle=False)
    preparation = CONFIG_PATH.parent.parent / "infra/quantrocket/research/prepare_wide.py"
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "preparation_complete": True,
                "config": spec,
                "feature_names": list(FEATURE_NAMES),
                "label_status_names": list(LABEL_STATUS_NAMES),
                "source_hashes": {
                    "wide_data.py": file_digest(Path(wide_data.__file__)),
                    "prepare_wide.py": file_digest(preparation),
                },
                "scope": {"reserved_2011_raw_read": False},
            }
        )
    )
    return values


def test_cache_retains_missing_outcomes_and_hashes_every_model_input(tmp_path):
    spec = config()
    write_cache(tmp_path, spec)
    arrays, dates, _, hashes = load_cache(tmp_path, spec)
    assert len(arrays["y"]) == 6
    assert np.isnan(arrays["y"][2])
    assert len(hashes) == 8
    assert dates.max() < pd.Timestamp("2011-01-01")


def test_reserved_cache_or_status_mismatch_is_rejected(tmp_path):
    spec = config()
    values = write_cache(tmp_path, spec)
    values["dates"][1] = np.datetime64("2011-01-03")
    np.save(tmp_path / "dates.npy", values["dates"], allow_pickle=False)
    with pytest.raises(ValueError, match="reserved"):
        load_cache(tmp_path, spec)
    write_cache(tmp_path, spec)
    values["status"][2] = 0
    np.save(tmp_path / "status.npy", values["status"], allow_pickle=False)
    with pytest.raises(ValueError, match="missingness"):
        load_cache(tmp_path, spec)


def test_cache_config_binding_prevents_posthoc_change(tmp_path):
    spec = config()
    write_cache(tmp_path, spec)
    changed = copy.deepcopy(spec)
    changed["min_price"] = 10
    with pytest.raises(ValueError, match="configuration"):
        load_cache(tmp_path, changed)


def test_stale_source_and_duplicate_forecast_rows_are_rejected(tmp_path):
    spec = config()
    write_cache(tmp_path, spec)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_hashes"]["wide_data.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source differs"):
        load_cache(tmp_path, spec)
    values = write_cache(tmp_path, spec)
    values["asset_index"][5] = values["asset_index"][4]
    np.save(tmp_path / "asset_index.npy", values["asset_index"], allow_pickle=False)
    with pytest.raises(ValueError, match="Duplicate decision-date"):
        load_cache(tmp_path, spec)


def test_fit_attempts_are_persisted_before_completion(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_fit_event(path, {"event": "fit_started", "model": "hgb_7"})
    assert json.loads(path.read_text())["event"] == "fit_started"
    append_fit_event(path, {"event": "fit_completed", "model": "hgb_7"})
    assert [json.loads(x)["event"] for x in path.read_text().splitlines()] == ["fit_started", "fit_completed"]


def test_missing_prior_fit_ledger_does_not_become_zero_consumed_trials(tmp_path):
    registry = tmp_path / "registry.jsonl"
    output = tmp_path / "attempt"
    output.mkdir()
    start_run(
        registry,
        "wide_ml_v1",
        "attempt1",
        config_sha256="a" * 64,
        data_sha256="b" * 64,
        source_sha256="c" * 64,
        output_dir=str(output),
    )
    fail_run(registry, "wide_ml_v1", "attempt1", error="engineering failure")
    with pytest.raises(RuntimeError, match="lacks its fit lifecycle"):
        enforce_trial_budget(registry, "wide_ml_v1", 14)
    (output / "fit_lifecycle.jsonl").touch()
    assert enforce_trial_budget(registry, "wide_ml_v1", 14) == 0
    append_fit_event(output / "fit_lifecycle.jsonl", {"event": "fit_started", "model": "hgb_7"})
    with pytest.raises(RuntimeError, match="already consumed"):
        enforce_trial_budget(registry, "wide_ml_v1", 14)
