import numpy as np
import pandas as pd
import pytest

from equity_alpha.portfolio import construct_weights, simulate_portfolio


def _prices(values, dates=None):
    if dates is None:
        dates = pd.date_range("2024-01-02", periods=len(values), freq="B")
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates), columns=["A", "B"], dtype=float)


def _targets(prices, rows, date_indices=None):
    if date_indices is None:
        date_indices = range(len(rows))
    return pd.DataFrame(rows, index=prices.index[list(date_indices)], columns=prices.columns, dtype=float)


def test_sector_neutral_weights_global_cap_and_single_name_sector():
    scores = pd.Series([4.0, 2.0, 6.0, 0.0, 99.0], index=list("ABCDE"))
    sectors = pd.Series(["x", "x", "y", "y", "z"], index=scores.index)
    weights = construct_weights(scores, sectors, max_weight=0.1)
    assert weights.groupby(sectors).sum().abs().max() < 1e-14
    assert weights.abs().max() == pytest.approx(0.1)
    assert weights.abs().sum() < 1
    assert weights["E"] == 0
    assert weights["A"] / weights["C"] == pytest.approx(1 / 3)
    pd.testing.assert_series_equal(weights, construct_weights(scores, sectors, max_weight=0.1))


def test_constant_and_zero_exposure_scores_are_cash():
    scores = pd.Series([2.0, 2.0], index=["A", "B"])
    sectors = pd.Series("x", index=scores.index)
    assert construct_weights(scores, sectors).eq(0).all()
    assert construct_weights(scores * [1, 2], sectors, gross_exposure=0).eq(0).all()


def test_decimal_constant_scores_do_not_normalize_roundoff_into_exposure():
    scores = pd.Series([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    sectors = pd.Series(["a", "a", "a", "b", "b", "b"])
    assert construct_weights(scores, sectors).eq(0).all()


def test_extreme_finite_scores_do_not_overflow_weight_construction():
    scores = pd.Series([1e308, -1e308, 1e307, -1e307])
    sectors = pd.Series(["a", "a", "b", "b"])
    weights = construct_weights(scores, sectors)
    assert np.isfinite(weights).all()
    assert weights.groupby(sectors).sum().abs().max() < 1e-14


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_scores_rejected(invalid):
    with pytest.raises(ValueError, match="complete and finite"):
        construct_weights(pd.Series([1, invalid]), pd.Series(["x", "x"]))


def test_sector_assignment_must_exist():
    with pytest.raises(ValueError, match="sector assignment"):
        construct_weights(pd.Series([1, 2], index=["A", "B"]), pd.Series({"A": "x"}))


def test_weights_do_not_force_requested_gross_when_capped():
    weights = construct_weights(pd.Series([1.0, -1.0]), pd.Series(["x", "x"]), max_weight=0.08)
    assert weights.abs().sum() == pytest.approx(0.16)
    assert weights.sum() == pytest.approx(0)


def test_flat_prices_charge_exact_entry_and_single_exit_costs():
    prices = _prices([[100, 100], [100, 100]])
    targets = _targets(prices, [[0.5, -0.5]])
    result = simulate_portfolio(prices, targets, one_way_cost_bps=100, annual_short_borrow_bps=0)
    entry_nav = 1 / 1.01
    final_nav = entry_nav * 0.99
    assert result.iloc[0]["equity"] == pytest.approx(entry_nav)
    assert result.iloc[-1]["equity"] == pytest.approx(final_nav)
    assert result["transaction_cost"].sum() == pytest.approx(1 - final_nav)
    assert result.iloc[0]["turnover"] == pytest.approx(entry_nav)
    assert result.iloc[-1]["turnover"] == pytest.approx(1)
    assert result.iloc[-1]["cash"] == pytest.approx(final_nav)
    assert result.iloc[-1]["gross_exposure"] == 0


def test_long_short_price_pnl_is_earned_before_execution():
    prices = _prices([[100, 100], [110, 90]])
    targets = _targets(prices, [[0.5, -0.5]])
    result = simulate_portfolio(prices, targets, 0, 0, liquidate=False)
    assert result.iloc[0]["equity"] == 1
    assert result.iloc[-1]["price_pnl"] == pytest.approx(0.1)
    assert result.iloc[-1]["return"] == pytest.approx(0.1)
    assert result.iloc[-1]["equity"] == pytest.approx(1.1)
    assert result.iloc[-1]["weight::A"] == pytest.approx(0.55 / 1.1)
    assert result.iloc[-1]["weight::B"] == pytest.approx(-0.45 / 1.1)
    assert result.iloc[-1]["net_exposure"] == pytest.approx(0.1 / 1.1)


def test_holdings_drift_and_rebalance_has_actual_not_target_weight_turnover():
    prices = _prices([[100, 100], [120, 100], [120, 100]])
    targets = _targets(prices, [[0.5, -0.5], [0.5, -0.5]], date_indices=[0, 2])
    result = simulate_portfolio(prices, targets, 0, 0, liquidate=False)
    assert result.iloc[1]["weight::A"] == pytest.approx(0.6 / 1.1)
    assert result.iloc[1]["turnover"] == 0
    assert result.iloc[2]["traded_notional"] == pytest.approx(0.1)
    assert result.iloc[2]["turnover"] == pytest.approx(0.1 / 1.1)
    assert result.iloc[2]["weight::A"] == pytest.approx(0.5)
    assert result.iloc[2]["weight::B"] == pytest.approx(-0.5)


def test_weekend_borrow_is_three_calendar_days_on_previous_short_notional():
    prices = _prices([[100, 100], [100, 100]], dates=["2024-01-05", "2024-01-08"])
    targets = _targets(prices, [[0.5, -0.5]])
    result = simulate_portfolio(prices, targets, 0, 3650, liquidate=False)
    assert result.iloc[0]["borrow_cost"] == 0
    assert result.iloc[1]["borrow_cost"] == pytest.approx(0.5 * 0.365 * 3 / 365)
    assert result.iloc[1]["equity"] == pytest.approx(0.9985)


def test_delayed_target_cannot_earn_pre_execution_return():
    prices = _prices([[100, 100], [200, 100], [220, 90]])
    targets = _targets(prices, [[0.5, -0.5]], date_indices=[1])
    result = simulate_portfolio(prices, targets, 0, 0, liquidate=False)
    assert result.iloc[1]["return"] == 0
    assert result.iloc[2]["return"] == pytest.approx(0.1)


def test_target_on_final_date_does_not_open_a_position_before_liquidation():
    prices = _prices([[100, 100], [100, 100]])
    targets = _targets(prices, [[0.5, -0.5]], date_indices=[1])
    result = simulate_portfolio(prices, targets, 25, 100)
    assert result["turnover"].sum() == 0
    assert result["equity"].eq(1).all()


def test_single_date_liquidating_simulation_remains_cash():
    prices = _prices([[100, 100]])
    result = simulate_portfolio(prices, _targets(prices, [[0.5, -0.5]]))
    assert result.iloc[0]["equity"] == 1
    assert result.iloc[0]["transaction_cost"] == 0


def test_empty_targets_stays_cash():
    prices = _prices([[100, 100], [110, 90]])
    targets = pd.DataFrame(index=pd.DatetimeIndex([]), columns=prices.columns, dtype=float)
    result = simulate_portfolio(prices, targets)
    assert result["equity"].eq(1).all()
    assert result["return"].eq(0).all()


def test_sector_drift_is_measured_only_with_metadata():
    prices = _prices([[100, 100], [110, 90]])
    targets = _targets(prices, [[0.5, -0.5]])
    assert simulate_portfolio(prices, targets, 0, 0, False)["sector_max_abs"].isna().all()
    targets.attrs["sectors"] = {"A": "same", "B": "same"}
    result = simulate_portfolio(prices, targets, 0, 0, False)
    assert result.iloc[0]["sector_max_abs"] == pytest.approx(0)
    assert result.iloc[1]["sector_max_abs"] == pytest.approx(0.1 / 1.1)
    direct = simulate_portfolio(prices, targets, 0, 0, False, sectors=pd.Series({"A": "x", "B": "x"}))
    np.testing.assert_allclose(result["sector_max_abs"], direct["sector_max_abs"])


def test_attribution_is_precost_and_sums_to_actual_price_return():
    prices = _prices([[100, 100], [110, 90], [108, 92]])
    result = simulate_portfolio(prices, _targets(prices, [[0.5, -0.5]]), 10, 100)
    contributions = result.attrs["contributions"]
    np.testing.assert_allclose(contributions.sum(axis=1), result["gross_return"])
    assert contributions.iloc[1]["A"] == pytest.approx(0.05)
    assert contributions.iloc[1]["B"] == pytest.approx(0.05)
    np.testing.assert_allclose(result.attrs["weights"]["A"], result["weight::A"])
    assert result.attrs["weights"].iloc[-1].eq(0).all()


def test_full_cash_and_return_ledger_reconciles_after_drift_costs_and_borrow():
    prices = _prices([[100, 100], [108, 97], [105, 99], [111, 96], [107, 102]])
    targets = _targets(prices, [[0.4, -0.4], [-0.3, 0.3]], date_indices=[0, 2])
    result = simulate_portfolio(prices, targets, 23, 275)
    asset_value = result["units::A"] * prices["A"] + result["units::B"] * prices["B"]
    np.testing.assert_allclose(result["cash"] + asset_value, result["equity"], atol=1e-14)
    np.testing.assert_allclose(
        result["return"],
        result["gross_return"] - result["transaction_cost_fraction"] - result["borrow_cost_fraction"],
        atol=1e-14,
    )
    np.testing.assert_allclose(result["turnover"] * 23 / 10_000, result["transaction_cost_fraction"])
    assert result.iloc[-1]["equity"] == pytest.approx(
        1 + result["price_pnl"].sum() - result["transaction_cost"].sum() - result["borrow_cost"].sum()
    )
    assert np.prod(1 + result["return"]) == pytest.approx(result.iloc[-1]["equity"])


def test_cost_solver_attains_post_fee_weights_on_rebalance():
    prices = _prices([[100, 100], [115, 90]])
    targets = _targets(prices, [[0.4, -0.4], [-0.25, 0.25]])
    result = simulate_portfolio(prices, targets, 100, 0, liquidate=False)
    assert result.iloc[-1]["weight::A"] == pytest.approx(-0.25)
    assert result.iloc[-1]["weight::B"] == pytest.approx(0.25)
    assert result.iloc[-1]["gross_exposure"] == pytest.approx(0.5)


def test_one_sided_trade_cash_ledger_and_full_sign_flip():
    prices = _prices([[100, 100], [110, 100], [100, 100]])
    targets = _targets(prices, [[1.0, 0.0], [-1.0, 0.0]], date_indices=[0, 1])
    result = simulate_portfolio(prices, targets, 100, 0)
    assert result.iloc[0]["cash"] == pytest.approx(0)
    assert result.iloc[1]["cash"] == pytest.approx(2 * result.iloc[1]["equity"])
    assert result.iloc[1]["weight::A"] == pytest.approx(-1)
    np.testing.assert_allclose(
        result["equity"],
        result["cash"] + result["units::A"] * prices["A"],
        atol=1e-14,
    )
    assert result.iloc[-1]["cash"] == pytest.approx(result.iloc[-1]["equity"])


@pytest.mark.parametrize("bad_price", [0, -1, np.nan, np.inf])
def test_invalid_prices_fail_instead_of_becoming_fills(bad_price):
    prices = _prices([[100, 100], [100, bad_price]])
    with pytest.raises(ValueError, match="positive, and finite"):
        simulate_portfolio(prices, _targets(prices, [[0.5, -0.5]]))


def test_non_execution_date_target_rejected():
    prices = _prices([[100, 100], [100, 100]])
    targets = _targets(prices, [[0.5, -0.5]])
    targets.index = pd.DatetimeIndex(["2024-01-06"])
    with pytest.raises(ValueError, match="execution date"):
        simulate_portfolio(prices, targets)


def test_incomplete_target_and_reordered_columns_rejected():
    prices = _prices([[100, 100], [100, 100]])
    targets = _targets(prices, [[0.5, np.nan]])
    with pytest.raises(ValueError, match="complete and finite"):
        simulate_portfolio(prices, targets)
    with pytest.raises(ValueError, match="ordered asset columns"):
        simulate_portfolio(prices, targets[["B", "A"]])


def test_overleveraged_target_and_bankruptcy_are_explicit_errors():
    prices = _prices([[100, 100], [100, 1000]])
    with pytest.raises(ValueError, match="gross exposure"):
        simulate_portfolio(prices, _targets(prices, [[1.0, -1.0]]))
    with pytest.raises(ValueError, match="equity is non-positive"):
        simulate_portfolio(prices, _targets(prices, [[0.5, -0.5]]), 0, 0, liquidate=False)
