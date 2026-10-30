"""Offline learner/cohort checks on invented targets and security identities."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from equity_alpha.wide_models import (
    make_wide_model,
    permute_training_targets,
    train_rows_for_year,
)


@pytest.mark.parametrize("name, alpha", [("ridge_10", 10), ("ridge_100", 100)])
def test_ridge_is_fixed_penalty_without_an_extra_scaler(name, alpha):
    model = make_wide_model(name, {"seed": 1729})
    assert isinstance(model, Ridge)
    assert model.alpha == alpha
    assert model.random_state == 1729
    assert not hasattr(model, "steps")


@pytest.mark.parametrize("name, leaves", [("hgb_7", 7), ("hgb_15", 15)])
def test_nonlinear_model_is_fixed_and_never_randomly_early_stops(name, leaves):
    model = make_wide_model(name, {"seed": 42})
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.max_leaf_nodes == leaves
    assert model.max_iter == 80
    assert model.learning_rate == 0.05
    assert model.l2_regularization == 10
    assert model.min_samples_leaf == 150
    assert model.early_stopping is False
    assert model.random_state == 42


def test_model_config_is_honored_and_sample_weights_are_supported():
    config = {
        "hgb_max_iter": 2,
        "hgb_learning_rate": 0.1,
        "hgb_l2_regularization": 4,
        "hgb_min_samples_leaf": 1,
    }
    model = make_wide_model("hgb_7", config)
    assert model.max_iter == 2 and model.learning_rate == 0.1
    assert model.l2_regularization == 4 and model.min_samples_leaf == 1
    x = np.arange(40).reshape(20, 2) / 40
    for candidate in (model, make_wide_model("ridge_100", {})):
        candidate.fit(x, np.sin(np.arange(20)), sample_weight=np.ones(20))
        assert np.isfinite(candidate.predict(x)).all()


@pytest.mark.parametrize(
    "name, config",
    [
        ("unknown", {}),
        ("hgb_7", {"early_stopping": True}),
        ("ridge_100", {"early_stopping": "auto"}),
        ("hgb_7", {"hgb_max_iter": 0}),
        ("hgb_7", {"hgb_min_samples_leaf": -1}),
        ("hgb_7", {"hgb_learning_rate": np.nan}),
        ("hgb_7", {"hgb_l2_regularization": -1}),
        ("ridge_10", {"seed": -1}),
    ],
)
def test_invalid_model_specification_fails_closed(name, config):
    with pytest.raises(ValueError):
        make_wide_model(name, config)


def panel():
    # Deliberately irregular observed-session spacing. The seventh following
    # calendar day is NOT the six-session exit used for the purge.
    dates = pd.DatetimeIndex(
        [
            "2008-12-15",
            "2008-12-18",
            "2008-12-19",
            "2008-12-22",
            "2008-12-23",
            "2008-12-24",
            "2008-12-26",
            "2008-12-29",
            "2008-12-30",
            "2008-12-31",
            "2009-01-02",
            "2009-01-05",
            "2009-01-06",
            "2009-01-07",
            "2009-01-08",
        ]
    )
    identities = np.array(["PERSIST_Z", "PERSIST_A", "PERSIST_K", "PERSIST_B", "PERSIST_F"])
    date_index = np.repeat(np.arange(12), len(identities))
    asset_index = np.tile(np.arange(len(identities)), 12)
    labels = (date_index + asset_index / 10).astype(float)
    statuses = np.zeros(len(labels), dtype=np.uint8)
    return date_index, dates, labels, statuses, asset_index, identities


def run_panel(values, config=None, year=2009):
    return train_rows_for_year(*values, year, {} if config is None else config)


def test_actual_session_label_maturity_purge_and_separate_missingness_audit():
    values = list(panel())
    values[3][0] = 2
    values[2][6] = np.nan
    selected, targets, weights, audit = run_panel(values)
    assert values[0][selected].max() == 3  # Dec22 + six actual sessions = Dec31.
    assert audit["rows_before_maturity_purge"] == 50
    assert audit["rows_after_maturity_purge"] == 20
    assert audit["maturity_purged_rows"] == 30
    assert audit["missing_label_rows"] == 2
    assert audit["noncomplete_status_rows_after_maturity_purge"] == 1
    assert audit["nonfinite_label_rows_after_maturity_purge"] == 1
    assert audit["observed_rows_before_sampling"] == audit["selected_rows"] == 18
    assert audit["max_row_date"] == "2008-12-22"
    assert audit["max_label_exit_date"] == "2008-12-31"
    assert all(row["selected_rows"] == 0 for row in audit["per_date"] if row["date"] >= "2008-12-23")
    assert len(targets) == len(weights) == len(selected)
    assert targets.dtype == np.float32 and weights.dtype == np.float64


def test_unavailable_exit_calendar_cannot_be_treated_as_mature():
    values = list(panel())
    values[1] = values[1][:12]
    selected, _, _, audit = run_panel(values, year=2010)
    assert values[0][selected].max() == 5
    assert audit["maturity_purged_rows"] == 30
    assert audit["max_label_exit_date"] == "2009-01-05"


def test_rank_targets_use_all_observed_day_rows_before_sampling():
    values = panel()
    selected, targets, _, _ = run_panel(values, {"max_training_names_per_date": 2})
    for day in np.unique(values[0][selected]):
        all_rows = np.flatnonzero(values[0] == day)
        expected = (rankdata(values[2][all_rows]) - 0.5) / len(all_rows) - 0.5
        lookup = dict(zip(all_rows, expected, strict=True))
        for row, actual in zip(
            selected[values[0][selected] == day], targets[values[0][selected] == day], strict=True
        ):
            assert actual == pytest.approx(lookup[row])
    # A two-name sample reranked after selection would only contain +/-0.25.
    assert not np.isin(np.abs(targets), [0.25]).any()


def test_tied_labels_receive_average_rank_before_sampling():
    values = list(panel())
    values[2][:5] = [0, 0, 2, 3, 3]
    selected, targets, _, _ = run_panel(values)
    lookup = dict(zip(selected, targets, strict=True))
    assert lookup[0] == lookup[1] == pytest.approx(-0.3)
    assert lookup[3] == lookup[4] == pytest.approx(0.3)
    assert lookup[2] == 0


def test_sampling_is_deterministic_capped_and_independent_of_target_values():
    values = panel()
    config = {"max_training_names_per_date": 2, "seed": 7}
    selected, _, _, audit = run_panel(values, config)
    repeated, _, _, _ = run_panel(values, config)
    np.testing.assert_array_equal(selected, repeated)
    assert max(row["selected_rows"] for row in audit["per_date"]) == 2
    altered = list(values)
    altered[2] = values[2] * -1000 + 21
    changed, _, _, _ = run_panel(altered, config)
    np.testing.assert_array_equal(selected, changed)
    other_seed, _, _, _ = run_panel(values, {**config, "seed": 8})
    assert not np.array_equal(selected, other_seed)


def test_future_or_unmatured_labels_cannot_influence_training_targets_or_sampling():
    values = panel()
    original = run_panel(values, {"max_training_names_per_date": 2})
    altered = list(values)
    altered[2] = values[2].copy()
    altered[3] = values[3].copy()
    future_or_unmatured = values[0] >= 4
    altered[2][future_or_unmatured] = np.nan
    altered[3][future_or_unmatured] = 2
    changed = run_panel(altered, {"max_training_names_per_date": 2})
    for first, second in zip(original[:3], changed[:3], strict=True):
        np.testing.assert_array_equal(first, second)
    assert original[3] == changed[3]


def test_sampling_and_order_are_invariant_to_input_rows_and_numeric_identity_dictionary():
    values = panel()
    config = {"max_training_names_per_date": 3, "seed": 55}
    original = run_panel(values, config)
    rng = np.random.default_rng(4)
    row_order = rng.permutation(len(values[0]))
    dictionary_order = np.array([3, 0, 4, 2, 1])
    inverse = np.argsort(dictionary_order)
    reordered = (
        values[0][row_order],
        values[1],
        values[2][row_order],
        values[3][row_order],
        inverse[values[4][row_order]],
        values[5][dictionary_order],
    )
    changed = run_panel(reordered, config)
    original_keys = [(int(values[0][row]), values[5][values[4][row]]) for row in original[0]]
    changed_keys = [(int(reordered[0][row]), reordered[5][reordered[4][row]]) for row in changed[0]]
    assert original_keys == changed_keys
    np.testing.assert_array_equal(original[1], changed[1])
    np.testing.assert_array_equal(original[2], changed[2])
    assert original[3] == changed[3]


def test_weights_balance_dates_and_are_rescaled_to_overall_mean_one():
    values = list(panel())
    values[3][:3] = 2  # First selected date has two names; the others have four.
    selected, _, weights, _ = run_panel(values, {"max_training_names_per_date": 4})
    sums = [weights[values[0][selected] == day].sum() for day in np.unique(values[0][selected])]
    np.testing.assert_allclose(sums, np.full(4, len(selected) / 4))
    assert weights.mean() == pytest.approx(1)
    assert np.isfinite(weights).all() and (weights > 0).all()


def test_nonfinite_or_noncomplete_training_outcomes_are_not_fitted():
    values = list(panel())
    values[2][:5] = [np.nan, np.inf, -np.inf, 3, 4]
    values[3][4] = 5
    selected, _, _, audit = run_panel(values)
    assert not {0, 1, 2, 4} & set(selected)
    assert audit["missing_label_rows"] == 4
    assert audit["nonfinite_label_rows_after_maturity_purge"] == 3


@pytest.mark.parametrize("year", [2008, 2011, 2012, 2009.0, True])
def test_no_reserved_2011_training_or_unapproved_prediction_year(year):
    with pytest.raises(ValueError, match="2009 and 2010"):
        run_panel(panel(), year=year)


def test_2011_rows_are_never_in_2010_training():
    values = list(panel())
    values[1] = pd.DatetimeIndex(
        [
            "2009-12-01",
            "2009-12-02",
            "2009-12-03",
            "2009-12-04",
            "2009-12-07",
            "2009-12-08",
            "2009-12-09",
            "2009-12-10",
            "2011-01-03",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
            "2011-01-07",
            "2011-01-10",
            "2011-01-11",
        ]
    )
    selected, _, _, audit = run_panel(values, year=2010)
    assert values[0][selected].max() == 1
    assert audit["max_row_date"] == "2009-12-02"
    assert audit["max_label_exit_date"] == "2009-12-10"


def test_all_missing_training_labels_return_empty_arrays_with_visible_audit():
    values = list(panel())
    values[2][:] = np.nan
    selected, targets, weights, audit = run_panel(values)
    assert selected.size == targets.size == weights.size == 0
    assert selected.dtype == np.int64 and targets.dtype == np.float32 and weights.dtype == np.float64
    assert audit["missing_label_rows"] == 20
    assert audit["selected_rows"] == audit["training_dates"] == 0
    assert audit["max_row_date"] is audit["max_label_exit_date"] is None


def test_zero_input_rows_return_empty_without_an_artificial_date():
    values = list(panel())
    for position in (0, 2, 3, 4):
        values[position] = values[position][:0]
    selected, _, _, audit = run_panel(values)
    assert selected.size == 0 and audit["per_date"] == []


def test_duplicate_security_dates_or_id_dictionary_are_rejected():
    values = list(panel())
    values[4][1] = values[4][0]
    with pytest.raises(ValueError, match="Duplicate"):
        run_panel(values)
    values = list(panel())
    values[5][1] = values[5][0]
    with pytest.raises(ValueError, match="unique"):
        run_panel(values)


@pytest.mark.parametrize("config", [{"max_training_names_per_date": 0}, {"label_exit_offset": 5}])
def test_invalid_training_policy_rejected(config):
    with pytest.raises(ValueError):
        run_panel(panel(), config)


def test_invalid_row_shapes_positions_and_calendar_fail_closed():
    for position, replacement in [(0, [-1]), (4, [100]), (3, [0]), (2, [[0]])]:
        values = list(panel())
        values[position] = replacement
        with pytest.raises(ValueError):
            run_panel(values)
    values = list(panel())
    values[1] = values[1][::-1]
    with pytest.raises(ValueError, match="increasing"):
        run_panel(values)


def test_permutation_preserves_selected_date_distributions_without_mutating_input():
    targets = np.arange(18, dtype=np.float32)
    dates = np.tile([2, 7, 9], 6)  # Date groups need not be adjacent.
    original = targets.copy()
    permuted = permute_training_targets(targets, dates, 191)
    np.testing.assert_array_equal(targets, original)
    np.testing.assert_array_equal(permuted, permute_training_targets(targets, dates, 191))
    assert permuted.dtype == np.float32 and not np.array_equal(permuted, targets)
    for day in np.unique(dates):
        np.testing.assert_array_equal(np.sort(permuted[dates == day]), np.sort(targets[dates == day]))


def test_permutation_handles_empty_singleton_and_rejects_invalid_targets():
    assert permute_training_targets(np.array([], dtype=np.float32), np.array([], dtype=int), 2).size == 0
    np.testing.assert_array_equal(permute_training_targets([0.1], [2], 2), [0.1])
    for targets, dates in [([np.nan], [1]), ([1], [1, 2]), ([[1]], [1])]:
        with pytest.raises(ValueError):
            permute_training_targets(targets, dates, 2)
