"""Past-only wide-panel transforms for the local Zipline learning exercise.

These are adjustment-ratio price-return proxies, not qualified cash-inclusive
total returns. Terminal payouts, corporate-action completeness, and historical
vendor revisions are not established by this module. No missing bar is filled.
The functions work on security chunks; nothing selects names using future data.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "rev1",
    "rev5",
    "mom21",
    "mom63",
    "mom126",
    "vol21",
    "vol63",
    "range20",
    "log_adv20",
    "volume_surprise",
    "illiq21",
    "close_location20",
)
LABEL_STATUS_NAMES = (
    "complete",
    "missing_entry",
    "missing_exit",
    "internal_gap",
    "right_boundary",
    "unresolved_action",
    "invalid_proxy",
)


def _session_seconds(sessions):
    index = pd.DatetimeIndex(sessions)
    if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("sessions must be nonempty, unique, increasing, and nonmissing")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if not index.equals(index.normalize()):
        raise ValueError("sessions must be UTC-midnight daily labels")
    return index.to_numpy(dtype="datetime64[s]").astype(np.int64)


def _event_table(frame, date_column):
    """Accept reader tables with epoch seconds or explicitly converted dates."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=["sid", "seconds", "ratio"])
    if not isinstance(frame, pd.DataFrame) or not {"sid", date_column}.issubset(frame):
        raise ValueError(f"action table requires sid and {date_column}")
    sid = pd.to_numeric(frame["sid"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(sid).all() or not (sid == np.floor(sid)).all():
        raise ValueError("action security IDs must be finite integers")
    values = frame[date_column]
    if pd.api.types.is_numeric_dtype(values.dtype):
        seconds = values.to_numpy(dtype=float)
        if not np.isfinite(seconds).all() or not (seconds == np.floor(seconds)).all():
            raise ValueError("action dates must be finite integer epoch seconds")
        seconds = seconds.astype(np.int64)
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
        if dates.hasnans:
            raise ValueError("action dates must not be missing")
        seconds = dates.to_numpy(dtype="datetime64[s]").astype(np.int64)
    result = pd.DataFrame({"sid": sid.astype(np.int64), "seconds": seconds})
    if "ratio" in frame:
        result["ratio"] = pd.to_numeric(frame["ratio"], errors="raise").to_numpy(dtype=float)
    return result


def adjustment_factors(sessions, sids, splits, dividends, mergers, payouts):
    """Return price/split factors, unresolved intervals, and aggregate counts.

    An event is assigned to the first session on/after its effective date, so
    factors at k represent (session[k-1], session[k]]. Events at/before the
    initial session, after the final session, or outside these sids are ignored.
    Split factors affect prices and inverse historical volume units; dividend
    and merger ratios affect prices only. Payouts lacking a finite positive
    matching dividend ratio invalidate the corresponding interval.
    """
    seconds = _session_seconds(sessions)
    ids = np.asarray(sids)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("sids must be a nonempty one-dimensional collection")
    numeric = ids.astype(float)
    if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
        raise ValueError("sids must be finite integers")
    ids = numeric.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("sids must be unique")
    positions = {int(sid): j for j, sid in enumerate(ids)}
    shape = (len(seconds), len(ids))
    qprice = np.ones(shape, dtype=np.float64)
    qsplit = np.ones(shape, dtype=np.float64)
    bad = np.zeros(shape, dtype=bool)
    tables = {
        "splits": _event_table(splits, "effective_date"),
        "dividends": _event_table(dividends, "effective_date"),
        "mergers": _event_table(mergers, "effective_date"),
    }
    audit = {"table_counts": {}, "interval_convention": "(previous_session, current_session]"}

    def locations(table):
        rows = np.searchsorted(seconds, table["seconds"].to_numpy(dtype=np.int64), side="left")
        columns = table["sid"].map(positions).fillna(-1).to_numpy(dtype=np.int64)
        scoped = (rows > 0) & (rows < len(seconds)) & (columns >= 0)
        return rows, columns, scoped

    for name, table in tables.items():
        if len(table) and "ratio" not in table:
            raise ValueError(f"{name} requires ratio")
        rows, columns, scoped = locations(table)
        ratios = table["ratio"].to_numpy(dtype=float)
        valid = np.isfinite(ratios) & (ratios > 0)
        selected = scoped & valid
        with np.errstate(over="ignore", invalid="ignore"):
            np.multiply.at(qprice, (rows[selected], columns[selected]), ratios[selected])
            if name == "splits":
                np.multiply.at(qsplit, (rows[selected], columns[selected]), ratios[selected])
        invalid = scoped & ~valid
        bad[rows[invalid], columns[invalid]] = True
        qprice[rows[invalid], columns[invalid]] = np.nan
        if name == "splits":
            qsplit[rows[invalid], columns[invalid]] = np.nan
        audit["table_counts"][name] = {
            "records": len(table),
            "applied": int(selected.sum()),
            "invalid_in_scope": int(invalid.sum()),
            "ignored_outside_interval_or_sids": int((~scoped).sum()),
        }

    cash = _event_table(payouts, "ex_date")
    rows, columns, scoped = locations(cash)
    div = tables["dividends"]
    valid_div = np.isfinite(div["ratio"].to_numpy(dtype=float)) & (div["ratio"].to_numpy(dtype=float) > 0)
    valid_keys = set(zip(div.loc[valid_div, "sid"], div.loc[valid_div, "seconds"], strict=True))
    matched = np.fromiter(
        ((sid, date) in valid_keys for sid, date in zip(cash["sid"], cash["seconds"], strict=True)),
        dtype=bool,
        count=len(cash),
    )
    unresolved = scoped & ~matched
    bad[rows[unresolved], columns[unresolved]] = True
    bad |= ~np.isfinite(qprice) | (qprice <= 0) | ~np.isfinite(qsplit) | (qsplit <= 0)
    audit["table_counts"]["payouts"] = {
        "records": len(cash),
        "in_scope": int(scoped.sum()),
        "without_valid_dividend_ratio_in_scope": int(unresolved.sum()),
        "ignored_outside_interval_or_sids": int((~scoped).sum()),
    }
    audit["unresolved_action_intervals"] = int(bad.sum())
    return qprice, qsplit, bad, audit


def _rolling(values, window, operation="mean"):
    rolling = pd.DataFrame(values).rolling(window, min_periods=window)
    return getattr(rolling, operation)().to_numpy(dtype=np.float64)


def features_and_labels(raw, sessions, qprice, qsplit, bad_action):
    """Compute 12 causal features and strict delayed five-session proxy labels.

    Momentum/reversal compound contiguous adjusted daily gross returns; a price
    gap or unresolved action therefore invalidates the entire affected window.
    Volatility is sample standard deviation of log gross returns, annualized by
    sqrt(252). Raw dollar volume is never dividend/split-adjusted. Volume-unit
    surprise uses V * cumulative_product(qsplit), a forward-causal unit basis.

    At decision close t, entry is t+1 and exit t+6; label returns end t+2..t+6.
    Status precedence is right boundary, missing entry, missing exit, internal
    close gap, unresolved action, invalid proxy, then complete. History validity
    depends only on contemporaneous/past observations, never future labels.
    """
    seconds = _session_seconds(sessions)
    fields = ("open", "high", "low", "close", "volume")
    if not set(fields).issubset(raw):
        raise ValueError("raw requires lowercase open/high/low/close/volume arrays")
    arrays = [np.asarray(raw[field], dtype=np.float64) for field in fields]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != len(seconds) or shape[1] == 0:
        raise ValueError("raw arrays must have shape (sessions, securities)")
    if any(array.shape != shape for array in arrays):
        raise ValueError("raw arrays must share the same shape")
    qprice, qsplit = np.asarray(qprice, dtype=float), np.asarray(qsplit, dtype=float)
    bad = np.asarray(bad_action, dtype=bool)
    if qprice.shape != shape or qsplit.shape != shape or bad.shape != shape:
        raise ValueError("adjustment factors and action mask must match raw arrays")
    opens, highs, lows, closes, volumes = arrays
    valid_close = np.isfinite(closes) & (closes > 0)
    valid_volume = np.isfinite(volumes) & (volumes >= 0)
    valid_ohlc = (
        np.isfinite(opens)
        & (opens > 0)
        & valid_close
        & np.isfinite(highs)
        & (highs > 0)
        & np.isfinite(lows)
        & (lows > 0)
        & (highs >= lows)
        & (highs >= opens)
        & (highs >= closes)
        & (lows <= opens)
        & (lows <= closes)
    )
    valid_factor = np.isfinite(qprice) & (qprice > 0)
    gross = np.full(shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gross[1:] = closes[1:] / (closes[:-1] * qprice[1:])
    valid_gross = np.zeros(shape, dtype=bool)
    valid_gross[1:] = (
        valid_close[1:]
        & valid_close[:-1]
        & valid_factor[1:]
        & ~bad[1:]
        & np.isfinite(gross[1:])
        & (gross[1:] > 0)
    )
    gross[~valid_gross] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gross = np.log(gross)
    features = np.full((*shape, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    features[:, :, 0] = -(gross - 1)
    for column, window, sign in ((1, 5, -1), (2, 21, 1), (3, 63, 1), (4, 126, 1)):
        with np.errstate(over="ignore", invalid="ignore"):
            features[:, :, column] = sign * np.expm1(_rolling(log_gross, window, "sum"))
    features[:, :, 5] = _rolling(log_gross, 21, "std") * np.sqrt(252)
    features[:, :, 6] = _rolling(log_gross, 63, "std") * np.sqrt(252)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        daily_range = np.where(valid_ohlc, (highs - lows) / closes, np.nan)
        dollars = np.where(valid_close & valid_volume, closes * volumes, np.nan)
    features[:, :, 7] = _rolling(daily_range, 20)
    adv20 = _rolling(dollars, 20)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        features[:, :, 8] = np.log(np.where(adv20 > 0, adv20, np.nan))
        split_ok = np.isfinite(qsplit) & (qsplit > 0)
        volume_scale = np.cumprod(np.where(split_ok, qsplit, np.nan), axis=0)
        causal_volume = np.where(valid_volume, volumes * volume_scale, np.nan)
        volume_mean = _rolling(causal_volume, 20)
        features[:, :, 9] = causal_volume / np.where(volume_mean > 0, volume_mean, np.nan) - 1
        illiq = np.abs(gross - 1) / np.where(dollars > 0, dollars, np.nan)
        location = np.where(highs == lows, 0.0, (2 * closes - highs - lows) / (highs - lows))
    features[:, :, 10] = _rolling(illiq, 21)
    features[:, :, 11] = _rolling(np.where(valid_ohlc, location, np.nan), 20)
    features[~np.isfinite(features)] = np.nan
    history_valid = np.isfinite(features).all(axis=2) & valid_ohlc & valid_volume

    label = np.full(shape, np.nan, dtype=np.float32)
    status = np.full(shape, 4, dtype=np.uint8)
    count = max(0, shape[0] - 6)
    if count:
        entry = valid_close[1 : count + 1]
        exit_ok = valid_close[6 : count + 6]
        interior = np.ones((count, shape[1]), dtype=bool)
        action = np.zeros_like(interior)
        proxy_ok = np.ones_like(interior)
        product = np.ones((count, shape[1]), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for offset in range(2, 7):
                if offset < 6:
                    interior &= valid_close[offset : count + offset]
                action |= bad[offset : count + offset]
                proxy_ok &= valid_gross[offset : count + offset]
                product *= gross[offset : count + offset]
            candidate = (product - 1).astype(np.float32)
        proxy_ok &= np.isfinite(candidate) & np.isfinite(product) & (product > 0)
        reason = np.select(
            [~entry, ~exit_ok, ~interior, action, ~proxy_ok],
            [1, 2, 3, 5, 6],
            default=0,
        ).astype(np.uint8)
        status[:count] = reason
        label[:count] = np.where(reason == 0, candidate, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        adv_out = adv20.astype(np.float32)
    adv_out[~np.isfinite(adv_out)] = np.nan
    return {
        "features": features,
        "adv20": adv_out,
        "history_valid": history_valid,
        "label_proxy": label,
        "label_status": status,
    }
