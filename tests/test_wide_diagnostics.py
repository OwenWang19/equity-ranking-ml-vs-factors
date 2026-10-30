"""Invented-array diagnostics tests; no vendor data, models or Docker needed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_alpha import wide_diagnostics
from equity_alpha.integrity import canonical_digest
from equity_alpha.wide_data import FEATURE_NAMES
from equity_alpha.wide_diagnostics import (
    MODEL_NAMES,
    SCOPES,
    authorize_analysis_attempt,
    diagnostic_daily,
    summarize_diagnostics,
    validate_frozen_predictions,
    validate_protocol,
)
from equity_alpha.wide_stats import daily_summary


def protocol():
    path = Path(__file__).resolve().parents[1] / "configs/wide_diagnostics_v1.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "change",
    [
        {"model": "hgb_15"},
        {"new_model_fits": 1},
        {"new_model_fits": False},
        {"reserved_performance_enabled": True},
        {"reserved_performance_enabled": 0},
        {"prediction_years": [2009, 2010, 2011]},
        {"seed": 42},
        {"bootstrap_sensitivity_blocks": [5, 20, 60]},
        {"promotion_authorized": True},
        {"analysis_status": "Confirmatory study"},
        {"source_run_id": "another"},
    ],
)
def test_protocol_cannot_be_silently_changed(change):
    config = protocol()
    config.update(change)
    with pytest.raises(ValueError):
        validate_protocol(config)


def test_frozen_protocol_valid():
    assert validate_protocol(protocol())["new_model_fits"] == 0


def prediction_fixture():
    dates = pd.to_datetime(["2008-12-30", "2009-01-02", "2010-01-04"])
    arrays = {
        "date_index": np.repeat(np.arange(3), 3),
        "asset_index": np.tile(np.arange(3), 3),
        "asset_ids": np.array(["a", "b", "c"]),
        "X": np.zeros((9, len(FEATURE_NAMES))),
        "y": np.ones(9),
        "status": np.zeros(9, dtype=np.int8),
    }
    frozen = {
        "date_index": arrays["date_index"][3:].copy(),
        "asset_index": arrays["asset_index"][3:].copy(),
        "model_names": np.array(MODEL_NAMES),
        "predictions": np.ones((6, len(MODEL_NAMES))),
    }
    return dates, arrays, frozen


def test_exact_frozen_cohort_rows_returned():
    dates, arrays, frozen = prediction_fixture()
    np.testing.assert_array_equal(validate_frozen_predictions(frozen, arrays, dates), np.arange(3, 9))


@pytest.mark.parametrize(
    "problem",
    [
        "row_reorder",
        "missing_row",
        "extra_model",
        "nonfinite",
        "reserved",
        "duplicate",
        "wrong_schema",
        "float_index",
        "security_reference",
        "dictionary",
        "timestamp",
    ],
)
def test_frozen_prediction_binding_fails_closed(problem):
    dates, arrays, frozen = prediction_fixture()
    if problem == "row_reorder":
        frozen["asset_index"] = frozen["asset_index"][::-1]
    elif problem == "missing_row":
        frozen["date_index"] = frozen["date_index"][:-1]
    elif problem == "extra_model":
        frozen["model_names"] = np.append(frozen["model_names"], "new_model")
    elif problem == "nonfinite":
        frozen["predictions"][0, 0] = np.nan
    elif problem == "reserved":
        dates = pd.to_datetime(["2008-12-30", "2009-01-02", "2011-01-04"])
    elif problem == "duplicate":
        arrays["asset_index"][-1] = arrays["asset_index"][-2]
        frozen["asset_index"][-1] = frozen["asset_index"][-2]
    elif problem == "wrong_schema":
        frozen["labels"] = np.ones(6)
    elif problem == "float_index":
        frozen["date_index"] = frozen["date_index"].astype(float)
    elif problem == "security_reference":
        arrays["asset_index"][0] = 99
    elif problem == "dictionary":
        arrays["asset_ids"][0] = "b"
    elif problem == "timestamp":
        dates = dates + pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        validate_frozen_predictions(frozen, arrays, dates)


def daily_fixture(sessions=80, names=24):
    rng = np.random.default_rng(2026)
    calendar = pd.bdate_range("2009-01-02", periods=sessions)
    scores = rng.normal(size=(sessions * names, len(MODEL_NAMES)))
    features = rng.normal(size=(sessions * names, len(FEATURE_NAMES)))
    labels = rng.normal(size=sessions * names)
    statuses = np.zeros(len(labels), dtype=np.int8)
    labels[-6 * names :] = np.nan
    statuses[-6 * names :] = 4
    labels[0] = np.nan
    statuses[0] = 2
    return scores, labels, statuses, calendar.repeat(names), calendar, features


def test_daily_scopes_keep_missing_and_boundary_rows_and_original_ic():
    args = daily_fixture(sessions=12, names=50)
    scores, labels, statuses, row_dates, calendar, _ = args
    daily = diagnostic_daily(*args)
    assert daily.scope.drop_duplicates().tolist() == list(SCOPES)
    all_days = daily.loc[daily.scope == "all"]
    assert all_days.predicted_rows.sum() == len(labels)
    assert all_days.nonboundary_missing_rows.sum() == 1
    assert all_days.hgb_7_ic.isna().sum() == 6
    assert all_days.hgb_7_partial_ic.isna().sum() == 6
    for date in calendar:
        mask = row_dates == date
        expected = daily_summary(scores[mask, MODEL_NAMES.index("hgb_7")], labels[mask], statuses[mask])
        got = all_days.loc[all_days.date == date].iloc[0]
        if np.isfinite(expected["observed_spearman_ic"]):
            assert got.hgb_7_ic == pytest.approx(expected["observed_spearman_ic"])
        else:
            assert np.isnan(got.hgb_7_ic)
        for family in ("liquidity", "volatility"):
            scopes = daily.loc[(daily.date == date) & daily.scope.isin([f"low_{family}", f"high_{family}"])]
            assert scopes.predicted_rows.sum() == mask.sum()
            assert scopes.observed_rows.sum() == np.isfinite(labels[mask]).sum()


def test_outcomes_cannot_change_group_forecast_counts():
    args = list(daily_fixture(sessions=2, names=50))
    first = diagnostic_daily(*args)
    args[1] = np.full_like(args[1], np.nan)
    args[2] = np.ones_like(args[2])
    changed = diagnostic_daily(*args)
    np.testing.assert_array_equal(first.predicted_rows, changed.predicted_rows)
    assert changed.observed_rows.sum() == 0


def test_full_calendar_and_every_fixed_sensitivity_are_retained():
    args = daily_fixture(sessions=80, names=24)
    daily = diagnostic_daily(*args)
    summary = summarize_diagnostics(daily, args[4], protocol())
    assert summary["new_model_fits"] == 0
    assert summary["reserved_performance_computed"] is False
    assert set(summary["scopes"]) == set(SCOPES)
    metric = summary["scopes"]["all"]["metrics"]["hgb7_minus_reversal5"]
    assert set(metric["block_sensitivity"]) == {"10", "20", "40"}
    assert metric["primary_interval"]["calendar_dates"] == 80
    assert metric["primary_interval"]["missing_pair_dates"] == 6
    assert metric["primary_interval"]["paired_dates"] == 74
    broken = daily.drop(daily.index[0])
    with pytest.raises(ValueError, match="exact full calendar"):
        summarize_diagnostics(broken, args[4], protocol())


def test_second_resolution_cache_calendar_matches_nanosecond_daily_table():
    args = list(daily_fixture(sessions=80, names=24))
    args[4] = pd.DatetimeIndex(args[4].to_numpy().astype("datetime64[D]"))
    args[3] = args[4].repeat(24)
    daily = diagnostic_daily(*args)
    result = summarize_diagnostics(daily, args[4], protocol())
    primary = result["scopes"]["all"]["metrics"]["hgb7_minus_reversal5"]["primary_interval"]
    assert primary["calendar_dates"] == 80
    assert primary["missing_pair_dates"] == 6


def test_new_analysis_cannot_claim_recovery_without_a_failed_attempt(tmp_path):
    assert authorize_analysis_attempt(tmp_path / "registry.jsonl", protocol()) is None
    with pytest.raises(RuntimeError, match="nonexistent"):
        authorize_analysis_attempt(tmp_path / "registry.jsonl", protocol(), "made_up")


@pytest.fixture
def recoverable_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(wide_diagnostics, "PLATFORM_ROOT", tmp_path)
    folder = tmp_path / "failed"
    folder.mkdir()
    (folder / "FAILURE.json").write_text(
        json.dumps({"type": "ValueError", "error": "Each scope must retain the exact full calendar"})
    )
    (folder / "config.json").write_text(json.dumps(protocol()))
    events = [
        {
            "experiment_id": protocol()["experiment_id"],
            "run_id": "failed1",
            "event": "run_started",
            "payload": {"output_dir": str(folder), "config_sha256": canonical_digest(protocol())},
        },
        {"experiment_id": protocol()["experiment_id"], "run_id": "failed1", "event": "run_failed"},
    ]
    monkeypatch.setattr(wide_diagnostics, "read_registry", lambda _: events)
    return folder, events


def test_one_explicit_pre_summary_engineering_recovery_is_logged(recoverable_attempt):
    folder, _ = recoverable_attempt
    got = authorize_analysis_attempt(Path("not_opened"), protocol(), "failed1")
    assert got["failed_attempt_retained"] is True
    assert got["failed_before_statistical_summary"] is True
    assert got["failed_run_id"] == "failed1"
    assert (folder / "FAILURE.json").is_file()
    with pytest.raises(RuntimeError, match="no automatic repeat"):
        authorize_analysis_attempt(Path("not_opened"), protocol())


@pytest.mark.parametrize(
    "problem", ["summary", "changed_config", "different_failure", "completed", "second_attempt"]
)
def test_recovery_cannot_become_a_result_or_protocol_search(recoverable_attempt, problem):
    folder, events = recoverable_attempt
    if problem == "summary":
        (folder / "summary.json").write_text("{}")
    elif problem == "changed_config":
        (folder / "config.json").write_text("{}")
    elif problem == "different_failure":
        (folder / "FAILURE.json").write_text("{}")
    elif problem == "completed":
        events[-1]["event"] = "run_completed"
    elif problem == "second_attempt":
        events.append(dict(events[0], run_id="failed2"))
    with pytest.raises(RuntimeError):
        authorize_analysis_attempt(Path("not_opened"), protocol(), "failed1")


@pytest.mark.parametrize("problem", ["labels_shape", "bad_status", "reserved", "nonfinite_features"])
def test_daily_rejects_invalid_input(problem):
    args = list(daily_fixture(sessions=2, names=50))
    if problem == "labels_shape":
        args[1] = args[1][:, None]
    elif problem == "bad_status":
        args[2][0] = -1
    elif problem == "reserved":
        args[4] = args[4] + pd.DateOffset(years=2)
    elif problem == "nonfinite_features":
        args[5][0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnostic_daily(*args)
