"""Offline checks using invented arrays; no provider data or security export."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from equity_alpha.wide_stats import daily_summary, paired_calendar_bootstrap


def direct_tau_a(scores, labels):
    terms = [
        np.sign(scores[j] - scores[i]) * np.sign(labels[j] - labels[i])
        for i, j in combinations(range(len(scores)), 2)
    ]
    return float(np.mean(terms))


def test_observed_ic_and_completion_bounds_are_hand_calculated():
    result = daily_summary([1, 2, 3, 4], [1, 3, 2, np.nan], ["ok", "ok", "ok", "terminal_missing"])
    assert result["predicted_rows"] == 4
    assert result["observed_rows"] == 3
    assert result["coverage"] == 0.75
    assert result["observed_spearman_ic"] == pytest.approx(0.5)
    assert result["known_concordance_sum"] == 1
    assert result["observed_kendall_tau_a"] == pytest.approx(1 / 3)
    assert result["unknown_untied_pairs"] == 3
    assert result["kendall_tau_a_lower"] == pytest.approx(-2 / 6)
    assert result["kendall_tau_a_upper"] == pytest.approx(4 / 6)
    assert result["label_status_counts"] == {"ok": 3, "terminal_missing": 1}


def test_kendall_tie_b_recovery_and_score_ties_in_missing_pairs():
    result = daily_summary([1, 1, 2, 2], [1, 2, 2, np.nan])
    assert result["score_tie_pairs"] == 2
    assert result["known_label_tie_pairs"] == 1
    assert result["known_concordance_sum"] == 1
    assert result["unknown_untied_pairs"] == 2
    assert result["kendall_tau_a_lower"] == pytest.approx(-1 / 6)
    assert result["kendall_tau_a_upper"] == pytest.approx(3 / 6)


@pytest.mark.parametrize("seed", range(5))
def test_random_full_completions_always_lie_within_bounds(seed):
    rng = np.random.default_rng(seed)
    scores = rng.integers(-2, 3, size=15)
    labels = rng.integers(-3, 4, size=15).astype(float)
    missing = rng.random(15) < 0.4
    result = daily_summary(scores, np.where(missing, np.nan, labels))
    for _ in range(100):
        completed = labels.copy()
        completed[missing] = rng.integers(-5, 6, size=int(missing.sum()))
        value = direct_tau_a(scores, completed)
        assert result["kendall_tau_a_lower"] - 1e-12 <= value
        assert value <= result["kendall_tau_a_upper"] + 1e-12


def test_complete_cohort_bounds_equal_tau_a_even_with_both_kinds_of_tie():
    scores, labels = np.array([0, 0, 2, 3, 3]), np.array([4, 4, 3, 3, 0])
    result = daily_summary(scores, labels)
    assert result["kendall_tau_a_lower"] == pytest.approx(direct_tau_a(scores, labels))
    assert result["kendall_tau_a_upper"] == result["kendall_tau_a_lower"]


def test_empty_cohort_and_constant_scores_have_explicit_conventions():
    empty = daily_summary([], [])
    assert empty["predicted_rows"] == empty["observed_rows"] == 0
    assert np.isnan(empty["coverage"])
    assert np.isnan(empty["observed_spearman_ic"])
    assert np.isnan(empty["kendall_tau_a_lower"])
    assert len(empty["score_deciles"]) == 10
    assert all(row["predicted_rows"] == 0 for row in empty["score_deciles"])
    constant = daily_summary([2] * 5, [1, 2, 3, np.nan, np.nan])
    assert constant["constant_scores"] and constant["constant_observed_scores"]
    assert constant["observed_spearman_ic"] == 0
    assert constant["kendall_tau_a_lower"] == constant["kendall_tau_a_upper"] == 0
    assert sum(row["predicted_rows"] > 0 for row in constant["score_deciles"]) == 1


def test_constant_labels_and_insufficient_observed_labels_have_undefined_ic():
    assert np.isnan(daily_summary([1, 2, 3], [4, 4, 4])["observed_spearman_ic"])
    assert np.isnan(daily_summary([1, 2, 3], [4, np.nan, 5])["observed_spearman_ic"])
    missing = daily_summary([1, 2, 3], [np.nan] * 3)
    assert missing["kendall_tau_a_lower"] == -1
    assert missing["kendall_tau_a_upper"] == 1


def test_deciles_keep_equal_scores_together_and_report_missing_rates():
    result = daily_summary([1, 1, 2, 3], [10, np.nan, 20, 30], [0, 2, 0, 0])
    assert sum(row["predicted_rows"] for row in result["score_deciles"]) == 4
    tied = next(row for row in result["score_deciles"] if row["predicted_rows"] == 2)
    assert tied["missing_rate"] == 0.5
    assert tied["label_status_counts"] == {"0": 1, "2": 1}
    assert sum(row["missing_rows"] for row in result["score_deciles"]) == 1


@pytest.mark.parametrize(
    "scores, labels, statuses",
    [
        ([1, np.nan], [1, 2], None),
        ([1, np.inf], [1, 2], None),
        ([1, 2], [1, np.inf], None),
        ([1, 2], [1], None),
        ([[1, 2]], [1, 2], None),
        ([1, 2], [1, 2], ["ok"]),
    ],
)
def test_invalid_vectors_are_not_silently_filtered(scores, labels, statuses):
    with pytest.raises(ValueError):
        daily_summary(scores, labels, statuses)


def manual_calendar_statistics(a, b, *, seed, samples, block):
    values = np.where(np.isfinite(a) & np.isfinite(b), a - b, np.nan)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(samples):
        starts = rng.integers(0, len(a), size=(len(a) + block - 1) // block)
        positions = np.concatenate([(start + np.arange(block)) % len(a) for start in starts])[: len(a)]
        drawn = values[positions]
        finite = drawn[np.isfinite(drawn)]
        results.append(float(finite.mean()) if finite.size else np.nan)
    return np.array(results)


def test_bootstrap_is_fixed_paired_mean_with_joint_date_resampling():
    index = pd.bdate_range("2009-01-02", periods=80)
    a = pd.Series(np.sin(np.arange(80)) + np.arange(80) / 10, index=index)
    b = a - 0.125
    result = paired_calendar_bootstrap(a, b, samples=100, block=10)
    assert result["estimate"] == pytest.approx(0.125)
    assert result["lower_95"] == pytest.approx(0.125)
    assert result["upper_95"] == pytest.approx(0.125)
    assert result["calendar_dates"] == result["paired_dates"] == 80
    assert result["valid_resamples"] == 100
    assert result["status"] == "ok"


def test_calendar_gaps_are_retained_not_compressed_before_block_sampling():
    index = pd.bdate_range("2009-01-02", periods=24)
    a = pd.Series(np.arange(24, dtype=float), index=index)
    b = pd.Series(np.zeros(24), index=index)
    a.iloc[[3, 4, 5, 6, 7, 8]] = np.nan
    b.iloc[[16, 17, 18]] = np.nan
    result = paired_calendar_bootstrap(a, b, seed=19, samples=200, block=3)
    manual = manual_calendar_statistics(a.to_numpy(), b.to_numpy(), seed=19, samples=200, block=3)
    assert result["calendar_dates"] == 24
    assert result["paired_dates"] == 15
    assert result["estimate"] == pytest.approx((a - b).mean())
    assert result["lower_95"] == pytest.approx(np.quantile(manual, 0.025))
    assert result["upper_95"] == pytest.approx(np.quantile(manual, 0.975))
    compressed = (a - b).dropna().to_numpy()
    wrong = manual_calendar_statistics(compressed, compressed * 0, seed=19, samples=200, block=3)
    assert not np.allclose(manual, wrong)


def test_missing_only_resamples_are_visible_not_removed_from_ci():
    index = pd.bdate_range("2009-01-02", periods=20)
    a = pd.Series(np.nan, index=index)
    b = pd.Series(0.0, index=index)
    a.iloc[2] = 1
    result = paired_calendar_bootstrap(a, b, samples=100, block=2)
    assert result["estimate"] == 1
    assert 0 < result["valid_resamples"] < 100
    assert np.isnan(result["lower_95"]) and np.isnan(result["upper_95"])
    assert result["status"] == "undefined_resamples_due_to_missing_pairs"


def test_all_missing_pairs_have_undefined_estimate_and_bounds():
    index = pd.bdate_range("2009-01-02", periods=40)
    result = paired_calendar_bootstrap(pd.Series(np.nan, index=index), pd.Series(0.0, index=index))
    assert result["paired_dates"] == result["valid_resamples"] == 0
    assert np.isnan(result["estimate"])
    assert np.isnan(result["lower_95"])


def test_unaligned_or_nonchronological_calendars_are_rejected():
    index = pd.bdate_range("2009-01-02", periods=40)
    a = pd.Series(np.arange(40), index=index)
    for b in (a.iloc[1:], a.iloc[::-1], a.set_axis(index + pd.Timedelta(days=1))):
        with pytest.raises(ValueError, match="calendar"):
            paired_calendar_bootstrap(a, b)
    duplicate = a.set_axis([index[0]] * 40)
    with pytest.raises(ValueError, match="calendar"):
        paired_calendar_bootstrap(duplicate, duplicate)


@pytest.mark.parametrize("kwargs", [{"samples": 0}, {"block": 0}, {"block": True}, {"samples": 1.5}])
def test_invalid_bootstrap_configuration_is_rejected(kwargs):
    a = pd.Series(np.arange(40), index=pd.bdate_range("2009-01-02", periods=40))
    with pytest.raises(ValueError):
        paired_calendar_bootstrap(a, a, **kwargs)


def test_infinite_daily_metrics_and_too_short_calendar_are_rejected():
    a = pd.Series(np.arange(40, dtype=float), index=pd.bdate_range("2009-01-02", periods=40))
    b = a.copy()
    b.iloc[0] = np.inf
    with pytest.raises(ValueError, match="daily_b"):
        paired_calendar_bootstrap(a, b)
    with pytest.raises(ValueError, match="two full"):
        paired_calendar_bootstrap(a.iloc[:10], a.iloc[:10])
