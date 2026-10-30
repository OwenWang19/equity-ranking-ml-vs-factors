"""Derived-only prediction diagnostics for the separate wide-security study.

The score cohort is fixed before labels are inspected. Observed-case Spearman
IC does not identify full-cohort IC when outcomes are missing. The completion
bounds below concern Kendall tau-a, not Spearman, and assume a latent outcome
exists for each missing label. They do not reconstruct a delisting return.
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata


def _numeric_vector(values, name, *, missing=False):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isinf(array).any() or (not missing and not np.isfinite(array).all()):
        allowed = "finite values or NaN" if missing else "finite values"
        raise ValueError(f"{name} must contain only {allowed}")
    return array


def _pairs(count):
    return count * (count - 1) // 2


def _tie_pairs(values):
    _, counts = np.unique(values, return_counts=True)
    return sum(_pairs(int(count)) for count in counts)


def _status_counts(statuses):
    return dict(sorted(Counter(str(value) for value in statuses).items()))


def daily_summary(scores, labels, label_status=None):
    """Summarize one model/date without returning security-level observations.

    All scores must be finite: invalid predictions are a visible error, not an
    opportunity to silently shrink the forecast cohort. NaN labels are retained
    as missing outcomes; infinite labels are rejected. ``label_status`` is an
    optional parallel vector of diagnostic categories, not an eligibility mask.

    IC uses at least three observed labels and nonconstant observed outcomes.
    A constant observed score has IC zero by declared convention, with explicit
    flags; an insufficient sample or constant outcome has undefined (NaN) IC.
    Score deciles use average ranks, keeping ties together and empty bins visible.
    """
    scores = _numeric_vector(scores, "scores")
    labels = _numeric_vector(labels, "labels", missing=True)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    observed = np.isfinite(labels)
    if label_status is None:
        statuses = np.where(observed, "observed", "missing")
    else:
        statuses = np.asarray(label_status)
        if statuses.ndim != 1 or len(statuses) != len(scores):
            raise ValueError("label_status must be a one-dimensional parallel vector")

    n, k = len(scores), int(observed.sum())
    observed_scores, observed_labels = scores[observed], labels[observed]
    constant_scores = bool(n and np.unique(scores).size == 1)
    constant_observed_scores = bool(k and np.unique(observed_scores).size == 1)
    constant_observed_labels = bool(k and np.unique(observed_labels).size == 1)
    ic = float("nan")
    if k >= 3 and not constant_observed_labels:
        if constant_observed_scores:
            ic = 0.0
        else:
            ic = float(np.corrcoef(rankdata(observed_scores), rankdata(observed_labels))[0, 1])

    total_pairs, known_pairs = _pairs(n), _pairs(k)
    score_ties, known_score_ties = _tie_pairs(scores), _tie_pairs(observed_scores)
    known_label_ties = _tie_pairs(observed_labels)
    concordance_sum = 0
    # tau-b = S / sqrt((P - score ties) * (P - outcome ties)). A zero
    # denominator means every known pair has a tie in at least one variable.
    denominator = np.sqrt(float(known_pairs - known_score_ties) * (known_pairs - known_label_ties))
    if denominator > 0:
        tau_b = float(kendalltau(observed_scores, observed_labels, variant="b").statistic)
        if not np.isfinite(tau_b):
            raise ValueError("Kendall calculation returned a non-finite statistic")
        concordance_sum = int(np.rint(tau_b * denominator))
    # Pairs with a tied prediction always contribute zero, even if their outcome
    # is missing. Removing these gives sharper, still conservative bounds.
    unknown_untied_pairs = (total_pairs - score_ties) - (known_pairs - known_score_ties)
    lower = (concordance_sum - unknown_untied_pairs) / total_pairs if total_pairs else float("nan")
    upper = (concordance_sum + unknown_untied_pairs) / total_pairs if total_pairs else float("nan")

    deciles = np.ceil(rankdata(scores, method="average") * 10 / n).astype(int) if n else np.array([])
    decile_rows = []
    for decile in range(1, 11):
        mask = deciles == decile
        count, known = int(mask.sum()), int(observed[mask].sum())
        decile_rows.append(
            {
                "decile": decile,
                "predicted_rows": count,
                "observed_rows": known,
                "missing_rows": count - known,
                "coverage": known / count if count else float("nan"),
                "missing_rate": (count - known) / count if count else float("nan"),
                "label_status_counts": _status_counts(statuses[mask]),
            }
        )
    return {
        "predicted_rows": n,
        "observed_rows": k,
        "missing_rows": n - k,
        "coverage": k / n if n else float("nan"),
        "missing_rate": (n - k) / n if n else float("nan"),
        "observed_spearman_ic": ic,
        "constant_scores": constant_scores,
        "constant_observed_scores": constant_observed_scores,
        "constant_observed_labels": constant_observed_labels,
        "score_tie_pairs": score_ties,
        "known_label_tie_pairs": known_label_ties,
        "known_concordance_sum": concordance_sum,
        "unknown_untied_pairs": unknown_untied_pairs,
        "observed_kendall_tau_a": concordance_sum / known_pairs if known_pairs else float("nan"),
        "kendall_tau_a_lower": float(lower),
        "kendall_tau_a_upper": float(upper),
        "label_status_counts": _status_counts(statuses),
        "score_deciles": decile_rows,
    }


def paired_calendar_bootstrap(daily_a, daily_b, *, seed=1729, samples=1000, block=20):
    """CI for paired mean daily ``a - b`` using whole-calendar circular blocks.

    Callers must provide Series on the same complete session calendar, including
    NaNs. Missing session rows cannot be inferred from calendar-day spacing, so
    unequal indexes are rejected instead of implicitly inner-joining them. Both
    models receive identical sampled date positions; NaNs remain at their original
    calendar positions until each resampled paired mean is calculated.

    If any bootstrap replicate has no observed pair, the interval is undefined
    and its bounds are NaN, rather than silently discarding those replicates.
    """
    if not isinstance(daily_a, pd.Series) or not isinstance(daily_b, pd.Series):
        raise TypeError("daily_a and daily_b must be pandas Series")
    index = daily_a.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or not isinstance(daily_b.index, pd.DatetimeIndex)
        or not index.equals(daily_b.index)
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Both inputs require the same unique, increasing, complete datetime calendar")
    for name, value in (("samples", samples), ("block", block)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len(index) < 2 * block:
        raise ValueError("Require at least two full calendar blocks")
    a = _numeric_vector(daily_a.to_numpy(), "daily_a", missing=True)
    b = _numeric_vector(daily_b.to_numpy(), "daily_b", missing=True)
    paired = np.isfinite(a) & np.isfinite(b)
    differences = np.full(len(index), np.nan)
    differences[paired] = a[paired] - b[paired]
    if not np.isfinite(differences[paired]).all():
        raise ValueError("Paired differences overflowed")
    estimate = float(differences[paired].mean()) if paired.any() else float("nan")
    rng = np.random.default_rng(seed)
    statistics = np.full(samples, np.nan)
    for sample in range(samples):
        starts = rng.integers(0, len(index), size=(len(index) + block - 1) // block)
        positions = ((starts[:, None] + np.arange(block)) % len(index)).ravel()[: len(index)]
        values = differences[positions]
        valid = np.isfinite(values)
        if valid.any():
            statistics[sample] = values[valid].mean()
    valid_resamples = int(np.isfinite(statistics).sum())
    defined = valid_resamples == samples
    return {
        "metric": "mean_daily_a_minus_b",
        "method": "paired_circular_calendar_block_bootstrap",
        "estimate": estimate,
        "lower_95": float(np.quantile(statistics, 0.025)) if defined else float("nan"),
        "upper_95": float(np.quantile(statistics, 0.975)) if defined else float("nan"),
        "calendar_dates": len(index),
        "paired_dates": int(paired.sum()),
        "missing_pair_dates": int((~paired).sum()),
        "block_sessions": int(block),
        "resamples": int(samples),
        "valid_resamples": valid_resamples,
        "status": "ok" if defined else "undefined_resamples_due_to_missing_pairs",
    }
