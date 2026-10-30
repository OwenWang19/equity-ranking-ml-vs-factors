import numpy as np
import pandas as pd
import pytest

from equity_alpha.analytics import daily_rank_ic, paired_sharpe_interval
from equity_alpha.config import ResearchConfig
from equity_alpha.evaluation import backtest_scores, make_schedule


def test_schedule_uses_signal_then_delayed_execution_and_complete_horizon():
    dates = pd.bdate_range("2022-01-03", periods=25)
    names = ["A", "B", "C", "D"]
    prices = pd.DataFrame(100.0, index=dates, columns=names)
    index = pd.MultiIndex.from_product([dates[:18], names], names=["date", "ticker"])
    predictions = pd.DataFrame({"score": np.tile([1.0, -1.0, 2.0, -2.0], 18)}, index=index)
    sectors = pd.Series(["x", "x", "y", "y"], index=names)
    config = ResearchConfig()
    schedule = make_schedule(predictions, prices, config)
    assert schedule.entry_date.iloc[0] == dates[1]
    assert schedule.exit_date.iloc[0] == dates[6]
    assert (schedule.entry_date.iloc[1:].to_numpy() == schedule.exit_date.iloc[:-1].to_numpy()).all()
    daily, targets, _ = backtest_scores(predictions, prices, sectors, config, "score")
    assert daily.index[0] == dates[0]
    assert daily["turnover"].iloc[0] == 0
    assert targets.index[0] == dates[1]
    assert daily.equity.iloc[-1] < 1  # flat prices: only costs and borrow can lose money
    assert daily.gross_exposure.iloc[-1] == 0


def test_rank_ic_handworked_by_date_and_constant_baseline():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-01", periods=2), ["A", "B", "C"]], names=["date", "ticker"]
    )
    y = pd.Series([1, 2, 3, 1, 2, 3], index=index)
    score = pd.Series([3, 2, 1, 1, 2, 3], index=index)
    np.testing.assert_allclose(daily_rank_ic(score, y), [-1, 1])
    assert daily_rank_ic(score * 0, y).eq(0).all()


def test_paired_interval_identical_returns_is_exact_zero():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 300))
    interval = paired_sharpe_interval(r, r, samples=100, block=20)
    assert interval["estimate"] == interval["lower_95"] == interval["upper_95"] == 0


def test_paired_interval_ignores_non_numeric_pandas_attribution_metadata():
    rng = np.random.default_rng(9)
    a = pd.Series(rng.normal(0, 0.01, 100))
    b = pd.Series(rng.normal(0, 0.01, 100))
    a.attrs["contributions"] = pd.DataFrame({"a": [1, 2]})
    b.attrs["contributions"] = pd.DataFrame({"a": [3, 4]})
    interval = paired_sharpe_interval(a, b, samples=100, block=10)
    assert np.isfinite(interval["estimate"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"models": ("hgb_shallow",)},
        {"models": ("unknown",)},
        {"one_way_cost_bps": float("nan")},
        {"holdout_start": "wrong"},
        {"test_years": (2024, 2023)},
        {"horizon": 1},
    ],
)
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        ResearchConfig(**kwargs)
