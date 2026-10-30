"""Fixed learners and past-only training cohorts for the wide prediction study.

Inputs are local derived row arrays. Selection is by a seeded hash of persistent
security identity and decision date, never by a label, feature, or transient
numeric security index. This module does not open data or release a holdout.
"""

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed(config):
    value = config.get("seed", 1729)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def make_wide_model(name, config):
    """Construct one of four fixed candidates; X is already cross-section ranked.

    In particular, Ridge receives no additional scaler. Tree early stopping is
    disabled rather than using an internal randomly held-out training subset.
    Model choice or holdout release is the caller's separately governed action.
    """
    if name not in {"ridge_10", "ridge_100", "hgb_7", "hgb_15"}:
        raise ValueError(f"Unknown wide model {name}")
    seed = _seed(config)
    if config.get("early_stopping", False) is not False:
        raise ValueError("The wide protocol requires early_stopping=False")
    if name.startswith("ridge_"):
        return Ridge(alpha=float(name.split("_")[1]), random_state=seed)
    iterations = _positive_integer(config.get("hgb_max_iter", 80), "hgb_max_iter")
    leaves = _positive_integer(config.get("hgb_min_samples_leaf", 150), "hgb_min_samples_leaf")
    rate = float(config.get("hgb_learning_rate", 0.05))
    regularization = float(config.get("hgb_l2_regularization", 10.0))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("hgb_learning_rate must be finite and positive")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("hgb_l2_regularization must be finite and nonnegative")
    return HistGradientBoostingRegressor(
        max_leaf_nodes=int(name.split("_")[1]),
        max_iter=iterations,
        learning_rate=rate,
        l2_regularization=regularization,
        min_samples_leaf=leaves,
        early_stopping=False,
        random_state=seed,
    )


def _index_vector(values, name, limit=None):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    if (array < 0).any() or (limit is not None and (array >= limit).any()):
        raise ValueError(f"{name} contains out-of-range positions")
    if len(array) and array.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds supported integer positions")
    return array.astype(np.int64, copy=False)


def _calendar(values):
    dates = pd.DatetimeIndex(values)
    if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("dates must be nonempty, unique, increasing session labels")
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    if not dates.equals(dates.normalize()):
        raise ValueError("dates must be UTC-midnight session labels")
    return dates


def _persistent_ids(values):
    supplied = np.asarray(values)
    if supplied.ndim != 1 or not len(supplied):
        raise ValueError("asset_ids must be a nonempty one-dimensional collection")
    if pd.isna(supplied).any():
        raise ValueError("asset_ids must not be missing")
    identities = supplied.astype(str)
    if any(not value.strip() for value in identities) or len(np.unique(identities)) != len(identities):
        raise ValueError("asset_ids must contain unique nonempty persistent identities")
    return identities


def _hash_order(identities, date, seed):
    """Stable hash ordering with identity itself as a deterministic collision tie."""
    prefix = f"wide_ml_v1\0{seed}\0{date}\0".encode()
    hashes = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(prefix + identity.encode(), digest_size=8).digest(), "big")
            for identity in identities
        ),
        dtype=np.uint64,
        count=len(identities),
    )
    return np.lexsort((identities, hashes))


def _counts(values):
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def train_rows_for_year(date_index, dates, y, status, asset_index, asset_ids, year, config):
    """Return original row positions, full-cohort rank targets, weights, and audit.

    ``date_index`` and ``asset_index`` map each row to the session calendar and
    persistent-ID list. Only 2009/2010 development fits are authorized. A row is
    considered for supervised fitting only if decision < Jan 1 of ``year`` AND
    its actual session-offset label exit < that boundary, status == 0, and y is
    finite. A missing exit-calendar position cannot count as a mature outcome.

    Targets are computed among ALL complete observed training rows on each date
    before sampling: (average_rank - 0.5) / cohort_size - 0.5. Sampling then caps
    each date independently. Output order is canonical date/persistent-ID order,
    so permuting input rows or the numeric ID dictionary does not change it.
    Weights give every selected date equal total weight and have overall mean 1.
    """
    if (
        isinstance(year, (bool, np.bool_))
        or not isinstance(year, (int, np.integer))
        or year not in (2009, 2010)
    ):
        raise ValueError("Only 2009 and 2010 development training years are authorized; never 2011")
    dates = _calendar(dates)
    identities = _persistent_ids(asset_ids)
    date_index = _index_vector(date_index, "date_index", len(dates))
    asset_index = _index_vector(asset_index, "asset_index", len(identities))
    status = _index_vector(status, "status")
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or any(len(values) != len(y) for values in (date_index, asset_index, status)):
        raise ValueError("Row vectors must be one-dimensional and have the same length")
    offset = _positive_integer(config.get("label_exit_offset", 6), "label_exit_offset")
    if offset != 6:
        raise ValueError("The fixed delayed five-session label requires label_exit_offset=6")
    cap = _positive_integer(config.get("max_training_names_per_date", 1500), "max_training_names_per_date")
    seed = _seed(config)
    boundary = pd.Timestamp(f"{int(year)}-01-01")
    before = dates.to_numpy()[date_index] < boundary.to_datetime64()
    exit_positions = date_index + offset
    has_exit = exit_positions < len(dates)
    mature = before & has_exit
    mature[has_exit] &= dates.to_numpy()[exit_positions[has_exit]] < boundary.to_datetime64()
    finite = np.isfinite(y)
    complete = mature & (status == 0) & finite
    missing = mature & ~((status == 0) & finite)

    selected_rows, selected_targets, selected_weights, per_date = [], [], [], []
    past_rows = np.flatnonzero(before)
    past_rows = past_rows[np.argsort(date_index[past_rows], kind="stable")]
    date_values = date_index[past_rows]
    starts = np.r_[0, np.flatnonzero(np.diff(date_values)) + 1]
    ends = np.r_[starts[1:], len(past_rows)]
    for start, end in zip(starts, ends, strict=True):
        if start == end:
            continue
        rows = past_rows[start:end]
        position = int(date_index[rows[0]])
        if len(np.unique(asset_index[rows])) != len(rows):
            raise ValueError("Duplicate decision-date/persistent-security rows")
        day = dates[position].date().isoformat()
        day_rows = rows[complete[rows]]
        day_ids = identities[asset_index[day_rows]]
        targets = (rankdata(y[day_rows], method="average") - 0.5) / max(1, len(day_rows)) - 0.5
        chosen = _hash_order(day_ids, day, seed)[:cap]
        chosen = chosen[np.argsort(day_ids[chosen], kind="stable")]
        if len(chosen):
            selected_rows.append(day_rows[chosen])
            selected_targets.append(targets[chosen].astype(np.float32))
            selected_weights.append(np.full(len(chosen), 1.0 / len(chosen), dtype=np.float64))
        exit_position = position + offset
        per_date.append(
            {
                "date": day,
                "label_exit_date": dates[exit_position].date().isoformat()
                if exit_position < len(dates)
                else None,
                "rows_before_maturity_purge": len(rows),
                "maturity_purged_rows": int((~mature[rows]).sum()),
                "rows_after_maturity_purge": int(mature[rows].sum()),
                "missing_label_rows": int(missing[rows].sum()),
                "observed_rows_before_sampling": len(day_rows),
                "selected_rows": len(chosen),
            }
        )
    selected = np.concatenate(selected_rows) if selected_rows else np.empty(0, dtype=np.int64)
    targets = np.concatenate(selected_targets) if selected_targets else np.empty(0, dtype=np.float32)
    weights = np.concatenate(selected_weights) if selected_weights else np.empty(0, dtype=np.float64)
    if len(weights):
        weights *= len(weights) / len(selected_weights)
    audit = {
        "training_year": int(year),
        "training_boundary_exclusive": boundary.date().isoformat(),
        "label_exit_offset": offset,
        "rows_before_maturity_purge": int(before.sum()),
        "rows_after_maturity_purge": int(mature.sum()),
        "maturity_purged_rows": int((before & ~mature).sum()),
        "missing_label_rows": int(missing.sum()),
        "noncomplete_status_rows_after_maturity_purge": int((mature & (status != 0)).sum()),
        "nonfinite_label_rows_after_maturity_purge": int((mature & ~finite).sum()),
        "label_status_counts_after_maturity_purge": _counts(status[mature]),
        "observed_rows_before_sampling": int(complete.sum()),
        "selected_rows": len(selected),
        "sampling_removed_rows": int(complete.sum()) - len(selected),
        "training_dates": len(selected_weights),
        "max_training_names_per_date": cap,
        "min_row_date": dates[int(date_index[selected].min())].date().isoformat() if len(selected) else None,
        "max_row_date": dates[int(date_index[selected].max())].date().isoformat() if len(selected) else None,
        "max_label_exit_date": dates[int(exit_positions[selected].max())].date().isoformat()
        if len(selected)
        else None,
        "target_rank_population": "All complete observed mature rows per date, before training sampling",
        "sampling": "blake2b(seed, decision_date, persistent_security_id); identity collision tie-break",
        "seed": seed,
        "per_date": per_date,
    }
    return selected, targets, weights, audit


def permute_training_targets(y_train, train_date_index, seed):
    """Break name/outcome alignment within selected dates; never shuffle dates."""
    targets = np.asarray(y_train)
    date_index = _index_vector(train_date_index, "train_date_index")
    if targets.ndim != 1 or len(targets) != len(date_index) or not np.isfinite(targets).all():
        raise ValueError("Targets must be a finite one-dimensional parallel vector")
    rng = np.random.default_rng(seed)
    result = targets.copy()
    for date in np.unique(date_index):
        rows = np.flatnonzero(date_index == date)
        result[rows] = rng.permutation(targets[rows])
    return result
