"""Invented-array checks for descriptive, observed-case rank projections."""

import numpy as np
import pytest
from scipy.stats import rankdata

from equity_alpha.incremental_stats import cross_sectional_groups, partial_rank_summary


def invented(n=80):
    rng = np.random.default_rng(191)
    controls = rng.normal(size=(n, 2))
    scores = 0.5 * controls[:, 0] + rng.normal(size=n)
    labels = 0.3 * controls[:, 1] + 0.4 * scores + rng.normal(size=n)
    return scores, labels, controls


def test_hand_calculated_single_control_fwl_example():
    scores = np.array([1, 4, 2, 6, 3, 5])
    labels = np.array([2, 1, 5, 3, 6, 4])
    control = np.arange(1, 7)[:, None]
    result = partial_rank_summary(scores, labels, control, min_observed=3)
    # Centered rank sums of squares are 17.5. The two control slopes are
    # 10.5 / 17.5 and 11.5 / 17.5. Residual covariance sum is -1.5 - 6.9.
    expected = -8.4 / np.sqrt(11.2 * (348 / 35))
    assert result["partial_rank_correlation"] == pytest.approx(expected)
    assert result["score_residual_norm_ratio"] == pytest.approx(0.8)
    assert result["outcome_residual_norm_ratio"] == pytest.approx(np.sqrt((348 / 35) / 17.5))
    assert result["design_rank"] == 2
    assert result["residual_dimension"] == 4
    assert result["status"] == "ok"


def test_matches_independent_normal_equation_projection():
    scores, labels, controls = invented()
    score_rank = rankdata(scores)
    label_rank = rankdata(labels)
    design = np.column_stack((np.ones(len(scores)), rankdata(controls[:, 0]), rankdata(controls[:, 1])))
    score_residual = score_rank - design @ np.linalg.solve(design.T @ design, design.T @ score_rank)
    label_residual = label_rank - design @ np.linalg.solve(design.T @ design, design.T @ label_rank)
    expected = np.corrcoef(score_residual, label_residual)[0, 1]
    result = partial_rank_summary(scores, labels, controls)
    assert result["partial_rank_correlation"] == pytest.approx(expected, abs=1e-12)
    assert result["design_rank"] == 3


def test_row_permutation_and_strictly_increasing_transforms_are_invariant():
    scores, labels, controls = invented()
    original = partial_rank_summary(scores, labels, controls)
    order = np.random.default_rng(727).permutation(len(scores))
    changed = partial_rank_summary(scores[order] ** 3, np.exp(labels[order]), controls[order] ** 3)
    assert changed["partial_rank_correlation"] == pytest.approx(original["partial_rank_correlation"])
    assert changed["design_rank"] == original["design_rank"]
    assert changed["score_residual_norm_ratio"] == pytest.approx(original["score_residual_norm_ratio"])


def test_collinear_and_constant_controls_preserve_projection_space():
    scores, labels, controls = invented()
    original = partial_rank_summary(scores, labels, controls)
    redundant = np.column_stack((controls, controls[:, 0] ** 3, np.ones(len(scores))))
    result = partial_rank_summary(scores, labels, redundant)
    assert result["design_rank"] == original["design_rank"] == 3
    assert result["partial_rank_correlation"] == pytest.approx(original["partial_rank_correlation"])


def test_zero_controls_reduce_to_observed_spearman_correlation():
    scores, labels, _ = invented()
    expected = np.corrcoef(rankdata(scores), rankdata(labels))[0, 1]
    result = partial_rank_summary(scores, labels, np.empty((len(scores), 0)))
    assert result["design_rank"] == 1
    assert result["partial_rank_correlation"] == pytest.approx(expected)
    assert result["score_residual_norm_ratio"] == pytest.approx(1)
    assert result["outcome_residual_norm_ratio"] == pytest.approx(1)


def test_all_constant_controls_are_equivalent_to_intercept_only():
    scores, labels, _ = invented()
    expected = partial_rank_summary(scores, labels, np.empty((len(scores), 0)))
    result = partial_rank_summary(scores, labels, np.ones((len(scores), 3)))
    assert result["design_rank"] == 1
    assert result["partial_rank_correlation"] == pytest.approx(expected["partial_rank_correlation"])


def test_single_residual_dimension_is_not_reported_as_mechanical_perfect_correlation():
    scores, labels, _ = invented(20)
    controls = np.eye(20)[:, :18]
    result = partial_rank_summary(scores, labels, controls)
    assert result["design_rank"] == 19
    assert result["residual_dimension"] == 1
    assert result["status"] == "insufficient_residual_dimension"
    assert np.isnan(result["partial_rank_correlation"])


def test_tied_ranks_match_independent_projection_without_input_mutation():
    scores, labels, controls = invented()
    scores, labels, controls = np.round(scores), np.round(labels), np.round(controls)
    before = [item.copy() for item in (scores, labels, controls)]
    design = np.column_stack((np.ones(len(scores)), rankdata(controls[:, 0]), rankdata(controls[:, 1])))
    residuals = []
    for values in (scores, labels):
        ranks = rankdata(values)
        residuals.append(ranks - design @ np.linalg.solve(design.T @ design, design.T @ ranks))
    result = partial_rank_summary(scores, labels, controls)
    assert result["partial_rank_correlation"] == pytest.approx(np.corrcoef(*residuals)[0, 1])
    assert all(
        np.array_equal(previous, current) for previous, current in zip(before, (scores, labels, controls))
    )


@pytest.mark.parametrize("which", ["scores", "labels"])
def test_score_or_outcome_equal_to_a_control_has_undefined_residual_correlation(which):
    scores, labels, controls = invented()
    if which == "scores":
        scores = controls[:, 0].copy()
    else:
        labels = controls[:, 0].copy()
    result = partial_rank_summary(scores, labels, controls)
    assert result["status"] == f"degenerate_{'score' if which == 'scores' else 'outcome'}_residual"
    assert np.isnan(result["partial_rank_correlation"])
    assert result[f"{'score' if which == 'scores' else 'outcome'}_residual_norm_ratio"] <= 1e-10


@pytest.mark.parametrize("which", ["scores", "labels"])
def test_constant_observed_scores_or_outcomes_are_explicitly_undefined(which):
    scores, labels, controls = invented()
    if which == "scores":
        scores[:] = 1
    else:
        labels[:] = 1
    result = partial_rank_summary(scores, labels, controls)
    assert result["status"] == f"constant_observed_{'scores' if which == 'scores' else 'outcomes'}"
    assert np.isnan(result["partial_rank_correlation"])


def test_unknown_outcomes_keep_coverage_but_ranks_are_on_observed_subset():
    scores, labels, controls = invented()
    labels[[0, 4, 9, 12, 13]] = np.nan
    observed = np.isfinite(labels)
    result = partial_rank_summary(scores, labels, controls)
    complete = partial_rank_summary(scores[observed], labels[observed], controls[observed])
    assert result["predicted_rows"] == 80
    assert result["observed_rows"] == 75
    assert result["missing_rows"] == 5
    assert result["coverage"] == 75 / 80
    assert result["partial_rank_correlation"] == pytest.approx(complete["partial_rank_correlation"])


def test_default_minimum_twenty_observed_rows_and_empty_cohort():
    scores, labels, controls = invented(20)
    assert partial_rank_summary(scores, labels, controls)["status"] == "ok"
    labels[0] = np.nan
    result = partial_rank_summary(scores, labels, controls)
    assert result["status"] == "insufficient_observed"
    assert result["observed_rows"] == 19
    assert np.isnan(result["partial_rank_correlation"])
    empty = partial_rank_summary([], [], np.empty((0, 2)))
    assert empty["status"] == "insufficient_observed"
    assert empty["predicted_rows"] == 0
    assert np.isnan(empty["coverage"])


def test_all_missing_outcomes_do_not_acquire_zero_correlation():
    scores, labels, controls = invented()
    labels[:] = np.nan
    result = partial_rank_summary(scores, labels, controls)
    assert result["observed_rows"] == 0
    assert result["missing_rows"] == result["predicted_rows"] == 80
    assert result["coverage"] == 0
    assert np.isnan(result["partial_rank_correlation"])


@pytest.mark.parametrize("minimum", [0, 2, True, 20.0, np.nan])
def test_invalid_minimum_is_rejected(minimum):
    scores, labels, controls = invented()
    with pytest.raises(ValueError, match="min_observed"):
        partial_rank_summary(scores, labels, controls, min_observed=minimum)


@pytest.mark.parametrize("where", ["scores", "labels", "controls"])
def test_nonfinite_scores_or_controls_rejected_even_at_missing_label_rows(where):
    scores, labels, controls = invented()
    labels[0] = np.nan
    if where == "scores":
        scores[0] = np.nan
    elif where == "controls":
        controls[0, 0] = np.inf
    else:
        labels[1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        partial_rank_summary(scores, labels, controls)


@pytest.mark.parametrize(
    "scores,labels,controls",
    [
        ([[1, 2]], [1, 2], [[1], [2]]),
        ([1, 2], [[1, 2]], [[1], [2]]),
        ([1, 2], [1], [[1], [2]]),
        ([1, 2], [1, 2], [1, 2]),
        ([1, 2], [1, 2], [[1, 2]]),
    ],
)
def test_invalid_parallel_vector_and_matrix_shapes_are_rejected(scores, labels, controls):
    with pytest.raises(ValueError):
        partial_rank_summary(scores, labels, controls)


def test_median_groups_keep_ties_together_in_lower_group():
    result = cross_sectional_groups([1, 2, 2, 3], [3, 2, 2, 1])
    assert result["all"].tolist() == [True] * 4
    assert result["low_liquidity"].tolist() == [True, True, True, False]
    assert result["low_volatility"].tolist() == [False, True, True, True]
    for variable in ("liquidity", "volatility"):
        assert np.array_equal(result[f"high_{variable}"], ~result[f"low_{variable}"])


def test_all_ties_empty_and_odd_groups_have_declared_rules():
    tied = cross_sectional_groups([1] * 4, [2] * 4)
    assert tied["low_liquidity"].all() and not tied["high_liquidity"].any()
    odd = cross_sectional_groups([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert odd["low_liquidity"].sum() == 3
    empty = cross_sectional_groups([], [])
    assert set(empty) == {"all", "low_liquidity", "high_liquidity", "low_volatility", "high_volatility"}
    assert all(value.dtype == bool and value.size == 0 for value in empty.values())


def test_grouping_is_row_order_transform_and_outcome_independent():
    scores, labels, controls = invented()
    groups_before = cross_sectional_groups(controls[:, 0], controls[:, 1])
    order = np.random.default_rng(2027).permutation(len(scores))
    groups_permuted = cross_sectional_groups(controls[order, 0] ** 3, np.exp(controls[order, 1]))
    for key in groups_before:
        assert np.array_equal(groups_before[key][order], groups_permuted[key])
    labels[::2] = np.nan
    groups_after = cross_sectional_groups(controls[:, 0], controls[:, 1])
    for key in groups_before:
        assert np.array_equal(groups_before[key], groups_after[key])


@pytest.mark.parametrize(
    "liquidity,volatility", [([1, np.nan], [1, 2]), ([1, 2], [1, np.inf]), ([[1, 2]], [1, 2]), ([1], [1, 2])]
)
def test_invalid_groups_fail_instead_of_silently_shrinking_cohort(liquidity, volatility):
    with pytest.raises(ValueError):
        cross_sectional_groups(liquidity, volatility)
