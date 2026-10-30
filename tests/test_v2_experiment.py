"""Actual small walk-forward runs test V2's research and frozen-model semantics."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from threadpoolctl import threadpool_limits

from equity_alpha.alignment import residualize_scores
from equity_alpha.config import ResearchConfig
from equity_alpha.experiment import _atomic_joblib_dump, run_forecasts
from equity_alpha.features import build_features
from equity_alpha.inference import predict_asof


@pytest.fixture(scope="module")
def miniature_run(tmp_path_factory):
    dates = pd.bdate_range("2016-01-04", "2021-12-31", name="date")
    names = pd.Index(["A", "B", "C", "D", "E", "F"], name="ticker")
    rng = np.random.default_rng(911)
    # Independent innovations deliberately provide no designed trading edge.
    # Shared shocks give the sector projection a substantive common component.
    sector_shocks = rng.normal(0, 0.004, size=(len(dates), 2))
    innovations = rng.normal(0, 0.012, size=(len(dates), len(names)))
    innovations += sector_shocks[:, [0, 0, 0, 1, 1, 1]]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(innovations, axis=0)), index=dates, columns=names)
    sectors = pd.Series(["Tech"] * 3 + ["Finance"] * 3, index=names, name="sector")
    config = ResearchConfig(
        experiment_id="synthetic_v2_integration",
        test_years=(2019, 2020),
        holdout_start="2021-01-01",
        min_train_dates=200,
        min_validation_dates=200,
        models=("ridge_1", "hgb_shallow"),
        target_mode="sector_relative",
        selection_metric="sector_mse",
        cash_hurdle=True,
    )
    bundle_dir = tmp_path_factory.mktemp("v2_models")
    with threadpool_limits(limits=2):
        output = run_forecasts(prices, sectors, config, controls=False, model_dir=bundle_dir)
    return prices, sectors, config, bundle_dir, output


def test_actual_v2_forecasts_project_names_within_the_same_sector(miniature_run):
    _, sectors, _, _, (forecasts, _, _, _) = miniature_run
    for family in ("ridge", "hgb", "selected"):
        ungated = forecasts[f"{family}_ungated"]
        labels = sectors.reindex(ungated.index.get_level_values("ticker"))
        totals = ungated.groupby([ungated.index.get_level_values("date"), labels.to_numpy()]).sum()
        np.testing.assert_allclose(totals, 0.0, atol=1e-15)
        assert np.isfinite(ungated).all()
        # An accidentally absent model cannot satisfy the neutrality check.
        assert ungated.abs().max() > 0


def test_actual_selection_minimizes_validation_mse_and_enforces_both_cash_gates(miniature_run):
    _, _, config, _, (forecasts, trials, _, _) = miniature_run
    failed = 0
    for year in config.test_years:
        candidates = trials.loc[(trials.fold == year) & (trials.kind == "selection_candidate")]
        candidates = candidates.set_index("candidate").reindex(config.models)
        expected_winner = candidates.validation_mse.idxmin()
        chosen = trials.loc[(trials.fold == year) & (trials.kind == "selected_selected")].iloc[0]
        assert chosen.candidate == expected_winner
        for family in ("ridge", "hgb", "selected"):
            decision = trials.loc[(trials.fold == year) & (trials.kind == f"selected_{family}")].iloc[0]
            row = candidates.loc[decision.candidate]
            expected_active = row.validation_mse < row.validation_zero_mse and row.validation_rank_ic > 0
            assert bool(decision.active) == bool(expected_active)
            fold_forecasts = forecasts.loc[forecasts.fold == year]
            if expected_active:
                np.testing.assert_array_equal(fold_forecasts[family], fold_forecasts[f"{family}_ungated"])
            else:
                failed += 1
                assert (fold_forecasts[family] == 0.0).all()
                assert decision.gate_reason == "validation_cash_hurdle_failed"
                assert fold_forecasts[f"{family}_ungated"].abs().max() > 0
    assert failed > 0, "This no-designed-edge fixture must exercise actual cash deployment"


def test_actual_model_bundles_retain_label_maturity_and_deployment_decisions(miniature_run):
    _, _, config, bundle_dir, (forecasts, trials, splits, _) = miniature_run
    for row in splits.itertuples(index=False):
        assert pd.Timestamp(row.train_max_label_end) < pd.Timestamp(f"{row.year - 1}-01-01")
        assert pd.Timestamp(row.validation_max_label_end) < pd.Timestamp(f"{row.year}-01-01")
        assert pd.Timestamp(row.refit_max_label_end) < pd.Timestamp(row.test_first_decision)
    assert forecasts.index.get_level_values("date").max() < pd.Timestamp(config.holdout_start)
    assert forecasts.label_end.max() < pd.Timestamp(config.holdout_start)
    for year in config.test_years:
        for family in ("ridge", "hgb", "selected"):
            bundle = joblib.load(bundle_dir / f"fold_{year}_{family}.joblib")
            decision = trials.loc[(trials.fold == year) & (trials.kind == f"selected_{family}")].iloc[0]
            assert bundle["candidate"] == decision.candidate
            assert bundle["deployment_active"] == bool(decision.active)
            assert bundle["deployment_gate_reason"] == decision.gate_reason
            assert bundle["target_mode"] == "sector_relative"
            assert bundle["config_sha256"] == config.digest()
            assert pd.Timestamp(bundle["fit_max_label_end"]) < pd.Timestamp(bundle["training_cutoff"])


def test_frozen_inference_reproduces_gated_scores_and_retains_ungated_estimator(miniature_run):
    prices, sectors, config, bundle_dir, (forecasts, _, _, _) = miniature_run
    for year in config.test_years:
        decision = forecasts.loc[forecasts.fold == year].index.get_level_values("date").unique()[100]
        for family in ("ridge", "hgb", "selected"):
            path = bundle_dir / f"fold_{year}_{family}.joblib"
            actual = predict_asof(prices.loc[:decision], sectors, path, decision)
            expected = forecasts.xs(decision, level="date")[family]
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-14, rtol=1e-13)
            # The saved estimator itself retains scientific forecasts even if
            # its deployment gate is false; inference correctly returns cash.
            bundle = joblib.load(path)
            history = build_features(prices.loc[:decision], sectors)
            features = history.loc[[decision], bundle["feature_names"]]
            with threadpool_limits(limits=2):
                raw = pd.Series(bundle["estimator"].predict(features), index=features.index)
            ungated = residualize_scores(raw, sectors).xs(decision, level="date")
            expected_ungated = forecasts.xs(decision, level="date")[f"{family}_ungated"]
            np.testing.assert_allclose(ungated, expected_ungated, atol=1e-14, rtol=1e-13)


def test_mid_test_future_price_changes_cannot_change_prior_trials_or_mature_predictions(miniature_run):
    prices, sectors, config, _, original = miniature_run
    cutoff = pd.Timestamp("2020-07-01")
    altered = prices.copy()
    after = altered.index > cutoff
    altered.loc[after] *= np.linspace(0.5, 4.0, int(after.sum()))[:, None]
    altered.loc[after, "A"] *= 10.0
    altered.loc[after, "F"] *= 0.1
    with threadpool_limits(limits=2):
        changed = run_forecasts(altered, sectors, config, controls=False)
    forecasts, trials, splits, importance = original
    altered_forecasts, altered_trials, altered_splits, altered_importance = changed
    pd.testing.assert_frame_equal(trials, altered_trials)
    pd.testing.assert_frame_equal(splits, altered_splits)
    pd.testing.assert_frame_equal(importance, altered_importance)
    # Restrict outcome columns to labels fully realized before perturbation.
    mature = forecasts.label_end <= cutoff
    pd.testing.assert_frame_equal(forecasts.loc[mature], altered_forecasts.loc[mature])
    # Model forecasts depend only on t features, even when their future labels
    # mature after the cutoff and are deliberately excluded from the check above.
    earlier = forecasts.index.get_level_values("date") <= cutoff
    score_columns = [name for name in forecasts if name not in ("target", "raw_forward_return")]
    pd.testing.assert_frame_equal(
        forecasts.loc[earlier, score_columns], altered_forecasts.loc[earlier, score_columns]
    )
    assert not forecasts["target"].equals(altered_forecasts["target"])


def test_exact_mse_ties_follow_declared_order_and_equal_zero_does_not_pass_gate(miniature_run, monkeypatch):
    prices, sectors, config, _, _ = miniature_run
    from equity_alpha import experiment

    monkeypatch.setattr(
        experiment,
        "make_model",
        lambda name, seed: DummyRegressor(strategy="constant", constant=0.0),
    )
    declared = replace(config, test_years=(2019,), models=("hgb_shallow", "ridge_1"))
    forecasts, trials, _, _ = run_forecasts(prices, sectors, declared, controls=False)
    chosen = trials.loc[trials.kind == "selected_selected"].iloc[0]
    assert chosen.candidate == "hgb_shallow"
    assert chosen.validation_mse == chosen.validation_zero_mse
    assert chosen.validation_rank_ic == 0.0
    assert bool(chosen.active) is False
    assert (forecasts[["ridge", "hgb", "selected"]] == 0.0).all().all()


def test_frozen_bundle_gate_changes_only_deployed_output_not_stored_model(miniature_run, tmp_path):
    prices, sectors, _, bundle_dir, (forecasts, trials, _, _) = miniature_run
    failed = trials.loc[trials.kind.str.startswith("selected_") & trials.active.eq(False)].iloc[0]
    family = failed.kind.removeprefix("selected_")
    year = int(failed.fold)
    original_path = bundle_dir / f"fold_{year}_{family}.joblib"
    bundle = joblib.load(original_path)
    decision = forecasts.loc[forecasts.fold == year].index.get_level_values("date").unique()[80]
    assert (predict_asof(prices, sectors, original_path, decision) == 0.0).all()
    # A diagnostic copy makes the pre-gate score observable through the same
    # inference implementation, while the real frozen deployment bundle stays
    # unchanged on disk and retains its failed-validation decision.
    bundle["deployment_active"] = True
    diagnostic_path = tmp_path / "diagnostic_ungated.joblib"
    _atomic_joblib_dump(bundle, diagnostic_path)
    diagnostic = predict_asof(prices, sectors, diagnostic_path, decision)
    expected = forecasts.xs(decision, level="date")[f"{family}_ungated"]
    np.testing.assert_allclose(diagnostic, expected, atol=1e-14, rtol=1e-13)
    assert joblib.load(original_path)["deployment_active"] is False
