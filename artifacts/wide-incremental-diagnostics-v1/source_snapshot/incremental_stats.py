"""Descriptive incremental-information diagnostics; no model fitting or trading.

Partial rank correlation here means a linear projection of contemporaneous
cross-sectional ranks, evaluated only where outcomes are observed. It is not a
tradable residual forecast, a general conditional-independence test, or a claim
that missing outcomes are ignorable. Group membership never accepts outcomes.
"""

import numpy as np
from scipy.stats import rankdata

RESIDUAL_ZERO_RATIO = 1e-10


def _vector(values, name, *, missing=False):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if np.isinf(values).any() or (not missing and not np.isfinite(values).all()):
        allowed = "finite values or NaN" if missing else "finite values"
        raise ValueError(f"{name} must contain only {allowed}")
    return values


def partial_rank_summary(scores, labels, controls, *, min_observed=20):
    """Correlate score/outcome rank residuals after the same SVD projection.

    ``controls`` is a finite [forecast rows, controls] matrix; zero controls are
    supported. Scores and controls must be finite on the entire forecast cohort,
    including rows with unknown labels. Average-tie ranks of all three variables
    are recomputed on the common observed-label subset, as required by this
    observed-case partial-rank estimand.

    Centered control rank columns are scaled to unit norm, preserving their
    projection space while avoiding scale-induced numerical rank changes.
    The normalized intercept is included in ``design_rank``. Rank-deficient and
    constant controls are allowed. A residual at most 1e-10 of its original
    centered-rank norm is numerically zero and produces an undefined statistic,
    never a conventionally zero or floating-noise correlation.

    The returned coverage retains all forecast rows. No coefficients, residual
    vectors, per-security values or newly fitted prediction model are returned.
    """
    scores = _vector(scores, "scores")
    labels = _vector(labels, "labels", missing=True)
    controls = np.asarray(controls, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("scores and labels must have the same length")
    if controls.ndim != 2 or controls.shape[0] != len(scores):
        raise ValueError("controls must be a [forecast rows, controls] matrix")
    if not np.isfinite(controls).all():
        raise ValueError("controls must contain only finite values on the full forecast cohort")
    if (
        isinstance(min_observed, (bool, np.bool_))
        or not isinstance(min_observed, (int, np.integer))
        or min_observed < 3
    ):
        raise ValueError("min_observed must be an integer at least three")

    observed = np.isfinite(labels)
    n, k = len(scores), int(observed.sum())
    result = {
        "predicted_rows": n,
        "observed_rows": k,
        "missing_rows": n - k,
        "coverage": k / n if n else float("nan"),
        "partial_rank_correlation": float("nan"),
        "design_rank": 0,
        "residual_dimension": 0,
        "score_residual_norm_ratio": float("nan"),
        "outcome_residual_norm_ratio": float("nan"),
        "status": "insufficient_observed",
    }
    if k < min_observed:
        return result

    score_ranks = rankdata(scores[observed], method="average")
    outcome_ranks = rankdata(labels[observed], method="average")
    score_centered = score_ranks - score_ranks.mean()
    outcome_centered = outcome_ranks - outcome_ranks.mean()
    score_norm, outcome_norm = np.linalg.norm(score_centered), np.linalg.norm(outcome_centered)

    ranked_controls = np.empty((k, controls.shape[1]), dtype=np.float64)
    for column in range(controls.shape[1]):
        ranked_controls[:, column] = rankdata(controls[observed, column], method="average")
    if controls.shape[1]:
        ranked_controls -= ranked_controls.mean(axis=0)
        norms = np.linalg.norm(ranked_controls, axis=0)
        np.divide(ranked_controls, norms, out=ranked_controls, where=norms > 0)
    design = np.column_stack((np.full(k, 1 / np.sqrt(k)), ranked_controls))
    left, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(np.float64).eps * singular_values[0]
    design_rank = int(np.count_nonzero(singular_values > tolerance))
    basis = left[:, :design_rank]
    score_residual = score_centered - basis @ (basis.T @ score_centered)
    outcome_residual = outcome_centered - basis @ (basis.T @ outcome_centered)
    score_residual_norm = np.linalg.norm(score_residual)
    outcome_residual_norm = np.linalg.norm(outcome_residual)
    score_ratio = float(score_residual_norm / score_norm) if score_norm else 0.0
    outcome_ratio = float(outcome_residual_norm / outcome_norm) if outcome_norm else 0.0
    result.update(
        design_rank=design_rank,
        residual_dimension=k - design_rank,
        score_residual_norm_ratio=float(np.clip(score_ratio, 0, 1)),
        outcome_residual_norm_ratio=float(np.clip(outcome_ratio, 0, 1)),
    )
    if score_norm == 0:
        result["status"] = "constant_observed_scores"
    elif outcome_norm == 0:
        result["status"] = "constant_observed_outcomes"
    elif score_ratio <= RESIDUAL_ZERO_RATIO:
        result["status"] = "degenerate_score_residual"
    elif outcome_ratio <= RESIDUAL_ZERO_RATIO:
        result["status"] = "degenerate_outcome_residual"
    elif k - design_rank < 2:
        result["status"] = "insufficient_residual_dimension"
    else:
        correlation = np.dot(score_residual, outcome_residual) / (score_residual_norm * outcome_residual_norm)
        result["partial_rank_correlation"] = float(np.clip(correlation, -1, 1))
        result["status"] = "ok"
    return result


def cross_sectional_groups(liquidity, volatility):
    """Median-split conditions using every eligible row, never future labels.

    Inputs may already be cross-section ranked; re-ranking makes the median rule
    explicit and invariant to strictly increasing transforms. Average ranks at
    or below (n + 1) / 2 belong to the lower group. Ties are never split, so group
    sizes may differ and an all-tied cohort belongs entirely to the lower group.
    These are within-date cross-sectional conditions, not market regimes.
    """
    liquidity = _vector(liquidity, "liquidity")
    volatility = _vector(volatility, "volatility")
    if liquidity.shape != volatility.shape:
        raise ValueError("liquidity and volatility must have the same length")
    n = len(liquidity)
    low_liquidity = rankdata(liquidity, method="average") <= (n + 1) / 2
    low_volatility = rankdata(volatility, method="average") <= (n + 1) / 2
    return {
        "all": np.ones(n, dtype=bool),
        "low_liquidity": low_liquidity,
        "high_liquidity": ~low_liquidity,
        "low_volatility": low_volatility,
        "high_volatility": ~low_volatility,
    }
