"""Artificial controls test engineering recovery, never financial performance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from equity_alpha.alignment import daily_cross_sectional_mse, residualize_scores
from equity_alpha.features import build_panel, feature_columns
from equity_alpha.synthetic_control import generate_control_prices


def test_longer_synthetic_paths_cannot_change_an_observed_prefix():
    short, sectors = generate_control_prices(sessions=350)
    long, long_sectors = generate_control_prices(sessions=700)
    pd.testing.assert_frame_equal(short, long.iloc[:350])
    pd.testing.assert_series_equal(sectors, long_sectors)
    assert np.isfinite(long).all().all()
    assert (long > 0).all().all()


def test_seed_and_universe_are_explicit_and_reproducible():
    original, sectors = generate_control_prices(seed=1729, sessions=50, n_names=5)
    same, _ = generate_control_prices(seed=1729, sessions=50, n_names=5)
    changed, _ = generate_control_prices(seed=1730, sessions=50, n_names=5)
    pd.testing.assert_frame_equal(original, same)
    assert not original.equals(changed)
    assert original.index.names == ["date"]
    assert original.columns.name == "ticker"
    assert sectors.index.equals(original.columns)
    assert sorted(sectors.value_counts()) == [2, 3]


def test_null_log_returns_equal_direct_independent_shocks_without_history_state():
    # At rho=0, the stationary initial state drops out exactly. Every log-return
    # vector is a function only of that date's independent innovations. Future
    # innovations are independent of the generated past by this construction.
    seed, count, names = 1729, 800, 8
    prices, _ = generate_control_prices(seed=seed, rho=0.0, sessions=count, n_names=names)
    streams = np.random.SeedSequence(seed).spawn(4)
    market = np.random.default_rng(streams[1]).normal(0, 0.0015, size=count)
    sector = np.random.default_rng(streams[2]).normal(0, 0.0015, size=(count, 2))
    idiosyncratic = np.random.default_rng(streams[3]).normal(0, 0.0025, size=(count, names))
    independent_returns = market[:, None] + sector[:, np.arange(names) % 2] + idiosyncratic
    observed = np.log(prices).diff().iloc[1:].to_numpy()
    np.testing.assert_allclose(observed, independent_returns[1:], atol=2e-15, rtol=0)
    # Identical marginal variances imply identical expected future simple returns
    # within each sector, hence zero conditional expected sector-relative target.
    marginal_variance = 0.0015**2 + 0.0015**2 + 0.0025**2
    expected_five_session_return = np.exp(5 * marginal_variance / 2) - 1
    expected_sector_relative_return = expected_five_session_return - expected_five_session_return
    assert expected_sector_relative_return == 0.0


def test_planted_persistence_survives_removal_of_common_sector_shocks():
    prices, sectors = generate_control_prices(sessions=1500)
    log_returns = np.log(prices).diff().dropna()
    relative = log_returns - log_returns.T.groupby(sectors).transform("mean").T
    lagged = relative.shift(1).iloc[1:].to_numpy().ravel()
    current = relative.iloc[1:].to_numpy().ravel()
    coefficient = np.dot(lagged, current) / np.dot(lagged, lagged)
    # Independently estimated known AR coefficient, not a fitted alpha threshold.
    assert abs(coefficient - 0.85) < 0.04


def test_price_features_recover_the_fixed_planted_signal_on_later_dates():
    prices, sectors = generate_control_prices()
    # No synthetic holdout outcome is needed for this engineering check.
    history = prices.loc[prices.index < pd.Timestamp("2021-01-01")]
    panel = build_panel(history, sectors, target_mode="sector_relative")
    columns = feature_columns()
    panel = panel.dropna(subset=columns + ["target", "label_end"])
    train = panel.loc[panel.label_end < pd.Timestamp("2020-01-01")]
    test = panel.loc[panel.index.get_level_values("date") >= pd.Timestamp("2020-01-01")]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    with threadpool_limits(limits=2):
        model.fit(train[columns], train.target)
        scores = pd.Series(model.predict(test[columns]), index=test.index)
    scores = residualize_scores(scores, sectors)
    error = daily_cross_sectional_mse(scores, test.target).mean()
    zero_error = daily_cross_sectional_mse(scores * 0, test.target).mean()
    assert error < zero_error


@pytest.mark.parametrize(
    "changes", [{"rho": 1.0}, {"rho": np.nan}, {"sessions": 1}, {"n_names": 3}, {"seed": -1}]
)
def test_invalid_generator_specifications_are_rejected(changes):
    with pytest.raises(ValueError):
        generate_control_prices(**changes)
