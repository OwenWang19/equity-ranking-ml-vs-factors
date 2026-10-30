"""Independent tests of the information clock and chronological research splits."""

import numpy as np
import pandas as pd
import pytest

from equity_alpha.config import ResearchConfig
from equity_alpha.features import build_features, build_panel, feature_columns
from equity_alpha.splits import walk_forward_folds


@pytest.fixture(scope="module")
def source():
    dates = pd.bdate_range("2016-01-04", periods=1100, name="date")
    rng = np.random.default_rng(927)
    shocks = rng.normal(0.0002, 0.01, (len(dates), 4))
    shocks += np.sin(np.arange(len(dates))[:, None] / 29) * np.array([0.002, -0.001, 0.001, -0.002])
    prices = pd.DataFrame(
        np.exp(np.cumsum(shocks, axis=0)) * [100, 55, 130, 80],
        index=dates,
        columns=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"),
    )
    sectors = pd.Series(["Tech", "Tech", "Utilities", "Utilities"], index=prices.columns, name="sector")
    return prices, sectors


def test_future_price_perturbations_cannot_change_past_features(source):
    prices, sectors = source
    cut = prices.index[750]
    altered = prices.copy()
    altered.loc[altered.index > cut, "AAA"] *= 7.0
    altered.loc[altered.index > cut, "CCC"] *= np.linspace(0.1, 3, (altered.index > cut).sum())
    original_features = build_features(prices, sectors)
    altered_features = build_features(altered, sectors)
    original_past = original_features.loc[original_features.index.get_level_values("date") <= cut]
    altered_past = altered_features.loc[altered_features.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(original_past, altered_past)
    # The perturbation is substantial: this test must actually exercise changed data.
    assert not original_features.equals(altered_features)


def test_feature_ranks_are_computed_within_each_date(source):
    prices, sectors = source
    features = build_features(prices, sectors)
    date = prices.index[600]
    expected = prices.pct_change(fill_method=None).loc[date].rank(pct=True)
    pd.testing.assert_series_equal(features.loc[date, "return_1"], expected, check_names=False)
    assert features.index.names == ["date", "ticker"]
    assert set(features.columns) == set(feature_columns())
    values = features.to_numpy()
    finite = values[np.isfinite(values)]
    assert np.all((finite > 0) & (finite <= 1))


def test_label_uses_close_t_plus_1_to_close_t_plus_6(source):
    prices, sectors = source
    panel = build_panel(prices, sectors, horizon=5, execution_delay=1)
    decision_position = 600
    decision = prices.index[decision_position]
    entry = prices.index[decision_position + 1]
    exit_date = prices.index[decision_position + 6]
    expected_raw = prices.loc[exit_date] / prices.loc[entry] - 1
    actual = panel.loc[decision]
    np.testing.assert_allclose(actual.raw_forward_return.to_numpy(), expected_raw.to_numpy(), atol=1e-14)
    np.testing.assert_allclose(
        actual.target.to_numpy(), (expected_raw - expected_raw.mean()).to_numpy(), atol=1e-14
    )
    assert (actual.entry_date == entry).all()
    assert (actual.label_end == exit_date).all()
    assert abs(actual.target.sum()) < 1e-14
    # A tempting same-close label is measurably different from the requested one.
    same_close = prices.loc[prices.index[decision_position + 5]] / prices.loc[decision] - 1
    assert not np.allclose(expected_raw, same_close)


def test_unknown_future_labels_remain_missing(source):
    prices, sectors = source
    panel = build_panel(prices, sectors, horizon=5, execution_delay=1)
    unavailable = panel.loc[prices.index[-6:]]
    assert unavailable.raw_forward_return.isna().all()
    assert unavailable.target.isna().all()
    assert unavailable.label_end.isna().all()
    assert panel.loc[prices.index[-7], "label_end"].notna().all()
    assert panel.loc[prices.index[-1], "entry_date"].isna().all()


def test_date_groups_and_label_maturity_are_purged_at_boundaries(source):
    prices, sectors = source
    panel = build_panel(prices, sectors).dropna(subset=feature_columns() + ["target", "label_end"])
    config = ResearchConfig(
        test_years=(2019, 2020), holdout_start="2021-01-01", min_train_dates=100, min_validation_dates=100
    )
    folds = walk_forward_folds(panel, config)
    assert [fold.year for fold in folds] == [2019, 2020]
    for fold in folds:
        validation_start = pd.Timestamp(f"{fold.year - 1}-01-01")
        test_start = pd.Timestamp(f"{fold.year}-01-01")
        train = panel.loc[fold.train]
        validation = panel.loc[fold.validation]
        refit = panel.loc[fold.refit]
        test = panel.loc[fold.test]
        assert train.label_end.max() < validation_start
        assert validation.label_end.max() < test_start
        assert refit.label_end.max() < test_start
        assert train.index.get_level_values("date").max() < validation_start
        assert validation.index.get_level_values("date").min() >= validation_start
        assert test.index.get_level_values("date").min() >= test_start
        assert test.index.get_level_values("date").max() < pd.Timestamp(f"{fold.year + 1}-01-01")
        assert set(fold.train).isdisjoint(fold.validation)
        assert set(fold.train).isdisjoint(fold.test)
        assert set(fold.validation).isdisjoint(fold.test)
        assert set(fold.refit).isdisjoint(fold.test)
        for partition in (train, validation, refit, test):
            assert (partition.groupby(level="date").size() == len(prices.columns)).all()
        # Explicitly prove rows with immature labels were removed, rather than
        # merely asserting chronological order after a convenient pre-filter.
        all_dates = panel.index.get_level_values("date")
        immature_train = panel.index[(all_dates < validation_start) & (panel.label_end >= validation_start)]
        immature_refit = panel.index[(all_dates < test_start) & (panel.label_end >= test_start)]
        assert len(immature_train) > 0
        assert len(immature_refit) > 0
        assert set(immature_train).isdisjoint(fold.train)
        assert set(immature_refit).isdisjoint(fold.refit)


def test_split_rejects_insufficient_history(source):
    prices, sectors = source
    panel = build_panel(prices, sectors).dropna(subset=feature_columns() + ["target", "label_end"])
    config = ResearchConfig(
        test_years=(2019,), holdout_start="2020-01-01", min_train_dates=10000, min_validation_dates=100
    )
    with pytest.raises(ValueError, match="train has .* requires 10000"):
        walk_forward_folds(panel, config)
