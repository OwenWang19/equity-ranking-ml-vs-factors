"""Hand-worked checks of forecast/portfolio subspace alignment."""

import numpy as np
import pandas as pd
import pytest

from equity_alpha.alignment import (
    daily_cross_sectional_mse,
    residualize_scores,
    sector_rank_ic,
    sector_relative_target,
)


@pytest.fixture
def sample():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    names = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, names], names=["date", "ticker"])
    sectors = pd.Series(["Tech", "Tech", "Bank", "Bank", "Single"], index=names)
    raw = pd.Series([0.10, 0.04, -0.01, 0.03, 0.50, 0.02, 0.08, 0.09, 0.01, -0.20], index=index)
    return raw, sectors


def test_hand_worked_labels_and_sector_neutral_pnl_identity(sample):
    raw, sectors = sample
    actual = sector_relative_target(raw, sectors)
    expected = [0.03, -0.03, -0.02, 0.02, 0.0, -0.03, 0.03, 0.04, -0.04, 0.0]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    # An independently specified sector-neutral portfolio earns exactly the
    # same gross return against raw returns and sector-relative labels.
    weights = np.array([0.12, -0.12, -0.08, 0.08, 0.0])
    for date in raw.index.get_level_values("date").unique():
        assert np.isclose(weights @ raw.loc[date], weights @ actual.loc[date], atol=1e-15)


def test_other_dates_cannot_change_target_or_score_projection(sample):
    raw, sectors = sample
    altered = raw.copy()
    altered.loc[pd.Timestamp("2020-01-03")] = [25.0, -7.0, 4.0, 10.0, 5.0]
    for function in (sector_relative_target, residualize_scores):
        pd.testing.assert_series_equal(
            function(raw, sectors).loc["2020-01-02"],
            function(altered, sectors).loc["2020-01-02"],
        )


def test_singleton_and_constant_decimal_forecasts_are_exactly_zero(sample):
    raw, sectors = sample
    scores = pd.Series(0.1, index=raw.index)
    projected = residualize_scores(scores, sectors)
    assert (projected == 0.0).all()
    assert (sector_relative_target(raw, sectors).xs("E", level="ticker") == 0).all()
    # The singleton's arbitrary values cannot affect a relative forecast metric.
    ic = sector_rank_ic(raw, raw, sectors)
    assert (ic == 1.0).all()
    assert (sector_rank_ic(scores, raw, sectors) == 0.0).all()


def test_sector_common_moves_do_not_change_rank_ic(sample):
    raw, sectors = sample
    shifted = raw.copy()
    shifted.loc[(slice(None), ["A", "B"])] += 2.5
    shifted.loc[(slice(None), ["C", "D"])] -= 4.0
    np.testing.assert_allclose(sector_rank_ic(shifted, raw, sectors), 1.0)
    np.testing.assert_allclose(sector_rank_ic(-shifted, raw, sectors), -1.0)


def test_missing_outcome_invalidates_group_not_its_peers(sample):
    raw, sectors = sample
    raw.loc[(pd.Timestamp("2020-01-02"), "A")] = np.nan
    target = sector_relative_target(raw, sectors)
    assert target.loc["2020-01-02", ["A", "B"]].isna().all()
    assert target.loc["2020-01-02", ["C", "D", "E"]].notna().all()
    assert target.loc["2020-01-03"].notna().all()


def test_daily_mse_uses_same_names_and_is_not_rank_invariant(sample):
    raw, sectors = sample
    target = sector_relative_target(raw, sectors)
    scores = target + 0.01
    actual = daily_cross_sectional_mse(scores, target.iloc[::-1])
    np.testing.assert_allclose(actual, [0.0001, 0.0001], atol=1e-15)
    assert (daily_cross_sectional_mse(target, target) == 0.0).all()
    assert (daily_cross_sectional_mse(target * 10, target) > actual).all()


def test_all_missing_targets_remain_undefined(sample):
    raw, sectors = sample
    missing = raw * np.nan
    assert sector_relative_target(missing, sectors).isna().all()
    assert sector_rank_ic(raw, missing, sectors).isna().all()
    assert daily_cross_sectional_mse(raw, missing).isna().all()


def test_diagnostics_reject_unmatched_names_and_invalid_sector_labels(sample):
    raw, sectors = sample
    with pytest.raises(ValueError, match="same date/ticker"):
        daily_cross_sectional_mse(raw, raw.iloc[:-1])
    with pytest.raises(ValueError, match="sector label"):
        sector_relative_target(raw, sectors.drop("A"))
    with pytest.raises(ValueError, match="unique"):
        residualize_scores(raw, pd.concat([sectors, sectors.iloc[:1]]))
    with pytest.raises(ValueError, match="unique"):
        sector_relative_target(pd.concat([raw, raw.iloc[:1]]), sectors)
    with pytest.raises(ValueError, match="finite"):
        residualize_scores(raw.mask(raw.index.get_level_values("ticker") == "A"), sectors)


def test_diagnostics_require_named_panel_and_finite_values(sample):
    raw, sectors = sample
    with pytest.raises(ValueError, match="MultiIndex"):
        residualize_scores(raw.reset_index(drop=True), sectors)
    with pytest.raises(ValueError, match="finite"):
        sector_relative_target(raw * np.inf, sectors)


def test_no_identifiable_cross_section_has_undefined_ic():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-02"]), ["A", "B", "C"]], names=["date", "ticker"]
    )
    values = pd.Series([0.1, 0.2, 0.3], index=index)
    sectors = pd.Series(["OnlyA", "OnlyB", "OnlyC"], index=["A", "B", "C"])
    assert sector_rank_ic(values, values, sectors).isna().all()
