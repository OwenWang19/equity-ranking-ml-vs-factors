"""Synthetic corporate actions and missing observations; no platform access."""

import numpy as np
import pandas as pd
import pytest

from equity_alpha.wide_data import (
    FEATURE_NAMES,
    LABEL_STATUS_NAMES,
    adjustment_factors,
    features_and_labels,
)


def action(sid, date, ratio):
    return {"sid": sid, "effective_date": pd.Timestamp(date).value // 10**9, "ratio": ratio}


def raw_panel(length=180, count=2):
    sessions = pd.bdate_range("2008-01-02", periods=length)
    trend = np.arange(length)[:, None] * np.linspace(0.0005, 0.001, count)[None, :]
    close = 100 * np.exp(trend + 0.003 * np.sin(np.arange(length)[:, None] / 9))
    return sessions, {
        "open": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close.copy(),
        "volume": np.full(close.shape, 10000.0),
    }


def transform(raw, sessions, qprice=None, qsplit=None, bad=None):
    shape = raw["close"].shape
    return features_and_labels(
        raw,
        sessions,
        np.ones(shape) if qprice is None else qprice,
        np.ones(shape) if qsplit is None else qsplit,
        np.zeros(shape, dtype=bool) if bad is None else bad,
    )


def test_factors_map_non_session_overlap_and_ignore_scope():
    sessions = pd.DatetimeIndex(["2008-01-04", "2008-01-07", "2008-01-08"])
    split = pd.DataFrame(
        [
            action(7, "2008-01-04", 0.1),
            action(7, "2008-01-05", 0.5),
            action(7, "2008-01-07", 0.5),
            action(7, "2008-01-09", 0.2),
            action(999, "2008-01-08", 0.1),
        ]
    )
    div = pd.DataFrame([action(7, "2008-01-07", 0.98)])
    merger = pd.DataFrame([action(7, "2008-01-08", 0.9)])
    payout = pd.DataFrame({"sid": [7], "ex_date": [pd.Timestamp("2008-01-07")]})
    price, split_factor, bad, audit = adjustment_factors(sessions, [7], split, div, merger, payout)
    np.testing.assert_allclose(price[:, 0], [1, 0.25 * 0.98, 0.9])
    np.testing.assert_allclose(split_factor[:, 0], [1, 0.25, 1])
    assert not bad.any()
    assert price.dtype == split_factor.dtype == np.float64
    assert audit["table_counts"]["splits"]["applied"] == 2
    assert audit["table_counts"]["splits"]["ignored_outside_interval_or_sids"] == 3


def test_unmatched_payout_or_invalid_ratio_is_not_no_action():
    sessions = pd.bdate_range("2008-01-02", periods=8)
    div = pd.DataFrame([action(7, sessions[2], 0.98), action(7, sessions[3], 0)])
    payouts = pd.DataFrame({"sid": [7, 7, 7], "ex_date": sessions[[2, 3, 4]]})
    price, _, bad, audit = adjustment_factors(sessions, [7], None, div, None, payouts)
    assert price[2, 0] == 0.98
    assert np.isnan(price[3, 0])
    np.testing.assert_array_equal(bad[:, 0], [False, False, False, True, True, False, False, False])
    assert audit["table_counts"]["payouts"]["without_valid_dividend_ratio_in_scope"] == 2
    assert audit["unresolved_action_intervals"] == 2


def test_labels_exclude_entry_action_and_future_action():
    sessions, raw = raw_panel(12, 1)
    price = np.ones((12, 1))
    price[[1, 3, 7], 0] = [0.5, 0.98, 0.2]
    result = transform(raw, sessions, qprice=price)
    expected = raw["close"][6, 0] / (raw["close"][1, 0] * 0.98) - 1
    assert result["label_proxy"][0, 0] == pytest.approx(expected, rel=1e-6)
    assert result["label_status"][0, 0] == 0
    assert np.isnan(result["label_proxy"][-6:]).all()
    assert (result["label_status"][-6:] == 4).all()
    price[1, 0], price[7, 0] = 0.001, 1000
    assert transform(raw, sessions, qprice=price)["label_proxy"][0, 0] == result["label_proxy"][0, 0]


@pytest.mark.parametrize("position, expected", [(1, 1), (6, 2), (3, 3)])
@pytest.mark.parametrize("missing", [np.nan, 0.0])
def test_label_missing_entry_exit_or_internal_is_never_filled(position, expected, missing):
    sessions, raw = raw_panel(15, 1)
    raw["close"][position, 0] = missing
    result = transform(raw, sessions)
    assert result["label_status"][0, 0] == expected
    assert np.isnan(result["label_proxy"][0, 0])


def test_label_status_bad_action_invalid_proxy_and_boundary():
    sessions, raw = raw_panel(12, 1)
    bad = np.zeros((12, 1), dtype=bool)
    bad[3] = True
    result = transform(raw, sessions, bad=bad)
    assert result["label_status"][0, 0] == 5
    assert np.isnan(result["label_proxy"][0, 0])
    bad[3], bad[1] = False, True
    assert transform(raw, sessions, bad=bad)["label_status"][0, 0] == 0
    factor = np.ones((12, 1))
    factor[3] = 0
    assert transform(raw, sessions, qprice=factor)["label_status"][0, 0] == 6
    assert (transform(raw, sessions)["label_status"][-6:] == 4).all()


def test_features_future_tail_invariant_and_history_not_label_filtered():
    sessions, raw = raw_panel()
    original = transform(raw, sessions)
    altered = {name: values.copy() for name, values in raw.items()}
    for values in altered.values():
        values[151:] *= 20
    factor = np.ones(raw["close"].shape)
    factor[160] = 0.01
    changed = transform(altered, sessions, qprice=factor, qsplit=factor)
    np.testing.assert_array_equal(original["features"][:151], changed["features"][:151])
    np.testing.assert_array_equal(original["adv20"][:151], changed["adv20"][:151])
    np.testing.assert_array_equal(original["history_valid"][:151], changed["history_valid"][:151])
    assert original["history_valid"][-1].all()
    assert np.isnan(original["label_proxy"][-1]).all()
    assert original["features"].dtype == original["adv20"].dtype == np.float32
    assert original["label_proxy"].dtype == np.float32
    assert original["label_status"].dtype == np.uint8
    assert original["history_valid"].dtype == bool
    assert original["features"].shape == (180, 2, 12)
    assert len(FEATURE_NAMES) == 12
    assert LABEL_STATUS_NAMES[4] == "right_boundary"


def test_split_causal_volume_removes_unit_jump_but_dividend_does_not_change_volume():
    sessions, raw = raw_panel(180, 1)
    raw["volume"][:] = 100
    raw["volume"][140:] = 200
    split = np.ones((180, 1))
    split[140] = 0.5
    price = split.copy()
    price[150] = 0.98
    result = transform(raw, sessions, qprice=price, qsplit=split)
    surprise = FEATURE_NAMES.index("volume_surprise")
    np.testing.assert_allclose(result["features"][139:, 0, surprise], 0, atol=1e-7)
    no_split = transform(raw, sessions, qprice=price)
    assert no_split["features"][140, 0, surprise] > 0.8
    expected_adv = np.mean(raw["close"][131:151, 0] * raw["volume"][131:151, 0])
    assert result["adv20"][150, 0] == pytest.approx(expected_adv, rel=1e-6)


def test_split_and_dividend_price_jumps_are_removed_from_return_features():
    sessions, raw = raw_panel(180, 1)
    close = np.full((180, 1), 100.0)
    close[140:] = 50
    close[150:] = 49
    for field, multiplier in (("open", 1), ("high", 1.01), ("low", 0.99), ("close", 1)):
        raw[field] = close * multiplier
    raw["volume"][:] = 100
    raw["volume"][140:] = 200
    split = np.ones((180, 1))
    split[140] = 0.5
    price = split.copy()
    price[150] = 0.98
    result = transform(raw, sessions, qprice=price, qsplit=split)
    np.testing.assert_allclose(result["features"][126:, 0, :7], 0, atol=1e-10)
    np.testing.assert_allclose(result["features"][126:, 0, 7], 0.02, atol=1e-7)
    np.testing.assert_allclose(result["features"][126:, 0, 9:], 0, atol=1e-10)
    assert not result["history_valid"][:126].any()
    assert result["history_valid"][126:].all()
    np.testing.assert_allclose(result["label_proxy"][:-6], 0, atol=1e-10)


def test_feature_values_match_independent_contiguous_window_calculations():
    sessions, raw = raw_panel(180, 1)
    result = transform(raw, sessions)
    close = raw["close"][:, 0]
    logs = np.log(close[1:] / close[:-1])
    row = result["features"][160, 0]
    assert row[0] == pytest.approx(-(close[160] / close[159] - 1), rel=1e-6)
    for column, window, sign in ((1, 5, -1), (2, 21, 1), (3, 63, 1), (4, 126, 1)):
        assert row[column] == pytest.approx(sign * (close[160] / close[160 - window] - 1), rel=1e-6)
    for column, window in ((5, 21), (6, 63)):
        expected = np.std(logs[160 - window : 160], ddof=1) * np.sqrt(252)
        assert row[column] == pytest.approx(expected, rel=1e-6)
    expected_illiq = np.mean(np.abs(np.expm1(logs[139:160])) / (close[140:161] * 10000))
    assert row[10] == pytest.approx(expected_illiq, rel=1e-6)


def test_rolling_returns_cannot_bridge_close_gap_and_raw_nan_is_not_zero():
    sessions, raw = raw_panel(180, 1)
    raw["close"][140] = np.nan
    raw["volume"][145] = np.nan
    result = transform(raw, sessions)
    rev5 = FEATURE_NAMES.index("rev5")
    assert np.isnan(result["features"][140:146, 0, rev5]).all()
    assert np.isnan(result["adv20"][145:165]).all()
    assert not result["history_valid"][140:180].any()
    assert np.isfinite(result["features"][146, 0, rev5])


def test_flat_bar_location_zero_but_invalid_zero_price_and_zero_volume_not_filled():
    sessions, raw = raw_panel(180, 1)
    for field in ("open", "high", "low", "close"):
        raw[field][:] = 100
    result = transform(raw, sessions)
    location = FEATURE_NAMES.index("close_location20")
    assert result["features"][140, 0, location] == 0
    assert result["history_valid"][140, 0]
    raw["volume"][145] = 0
    zero_volume = transform(raw, sessions)
    assert np.isnan(zero_volume["features"][145, 0, FEATURE_NAMES.index("illiq21")])
    assert np.isfinite(zero_volume["adv20"][145, 0])
    raw["close"][150] = 0
    zero_price = transform(raw, sessions)
    assert not zero_price["history_valid"][150, 0]
    assert np.isnan(zero_price["features"][150, 0, 0])


def test_invalid_inputs_fail_closed():
    sessions, raw = raw_panel(20, 1)
    with pytest.raises(ValueError, match="unique"):
        adjustment_factors(sessions, [1, 1], None, None, None, None)
    with pytest.raises(ValueError, match="increasing"):
        adjustment_factors(sessions[::-1], [1], None, None, None, None)
    with pytest.raises(ValueError, match="same shape"):
        transform({**raw, "volume": np.ones((1, 1))}, sessions)
