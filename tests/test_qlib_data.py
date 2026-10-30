"""Independent, hand-written binary fixtures for archive structure and missingness."""

import hashlib
import json
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from equity_alpha.qlib_data import classify_membership_quality, load_qlib_archive


def payload(start, values):
    return struct.pack("<" + "f" * (len(values) + 1), start, *values)


def archive(tmp_path, overrides=None):
    members = {
        "calendars/day.txt": b"2020-01-02\n2020-01-03\n2020-01-06\n2020-01-07\n",
        "instruments/sp500.txt": (
            b"AAA\t2020-01-02\t2020-01-03\n"
            b"AAA\t2020-01-07\t2099-12-31\n"
            b"BBB\t2020-01-03\t2020-01-06\n"
            b"DEAD\t1999-01-01\t2020-01-07\n"
        ),
    }
    for ticker, start, values in (("aaa", 0, [1, 1.1, np.nan, 1.2]), ("bbb", 1, [2, 1.8, 1.9])):
        for field in ("open", "high", "low", "close"):
            members[f"features/{ticker}/{field}.day.bin"] = payload(start, values)
        members[f"features/{ticker}/volume.day.bin"] = payload(
            start, [100, 0, 120][: len(values)] if ticker == "bbb" else [100, 101, np.nan, 102]
        )
        members[f"features/{ticker}/factor.day.bin"] = payload(start, [0.01] * len(values))
    members.update(overrides or {})
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, value in members.items():
            output.writestr(name, value)
    return path


def test_binary_offset_membership_missingness_and_exit_audit(tmp_path):
    source = archive(tmp_path)
    before = source.read_bytes()
    prices, volume, factor, membership, audit = load_qlib_archive(source)
    assert prices.columns.tolist() == ["AAA", "BBB", "DEAD"]
    np.testing.assert_allclose(prices["AAA"], [1, 1.1, np.nan, 1.2], equal_nan=True)
    np.testing.assert_allclose(prices["BBB"], [np.nan, 2, 1.8, 1.9], equal_nan=True)
    assert membership["AAA"].tolist() == [True, True, False, True]
    assert membership["BBB"].tolist() == [False, True, True, False]
    assert membership["DEAD"].all() and prices["DEAD"].isna().all()
    # An off-membership price is retained for potential liquidation, never filled.
    assert prices.loc["2020-01-07", "BBB"] == pytest.approx(1.9)
    assert np.isnan(prices.loc["2020-01-06", "AAA"])
    assert audit["tickers_without_close_data"] == ["DEAD"]
    assert audit["per_ticker"]["DEAD"]["member_sessions_missing_close"] == 4
    assert audit["membership_exits_without_next_session_close"][0]["ticker"] == "AAA"
    assert audit["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert audit["source_bytes"] == len(before)
    assert source.read_bytes() == before
    assert (volume * factor).loc["2020-01-02", "AAA"] == pytest.approx(1.0)
    assert (prices / factor).loc["2020-01-02", "AAA"] == pytest.approx(100)
    assert audit["zero_volume_observations"] == 1
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"../outside.txt": b"bad"}, "Unsafe ZIP"),
        ({"features\\aaa\\close.day.bin": b"bad"}, "Unsafe ZIP"),
        ({"calendars/day.txt": b"2020-01-02\n2020-01-02\n"}, "strictly increasing"),
        ({"calendars/day.txt": b"2020-1-02\n"}, "Invalid ISO date"),
        ({"features/aaa/close.day.bin": b"12345"}, "truncated binary"),
        ({"features/aaa/close.day.bin": payload(0.5, [1, 2])}, "start index"),
        ({"features/aaa/close.day.bin": payload(-1, [1, 2])}, "start index"),
        ({"features/aaa/close.day.bin": payload(np.nan, [1, 2])}, "start index"),
        ({"features/aaa/close.day.bin": payload(3, [1, 2])}, "span exceeds"),
        ({"instruments/sp500.txt": b"../AAA\t2020-01-02\t2020-01-03\n"}, "Unsafe membership"),
        ({"instruments/sp500.txt": b"AAA\t2020-01-03\t2020-01-02\n"}, "start exceeds"),
    ],
)
def test_rejects_unsafe_or_malformed_input(tmp_path, overrides, error):
    with pytest.raises(ValueError, match=error):
        load_qlib_archive(archive(tmp_path, overrides))


def test_invalid_economic_observations_become_audited_nan(tmp_path):
    source = archive(
        tmp_path,
        {
            "features/aaa/close.day.bin": payload(0, [0, np.inf, -1, 1]),
            "features/aaa/volume.day.bin": payload(0, [-1, 0, np.nan, 1]),
            "features/aaa/factor.day.bin": payload(0, [0, 0.01, 0.01, 0.01]),
        },
    )
    before = source.read_bytes()
    prices, volume, factor, _, audit = load_qlib_archive(source)
    assert prices["AAA"].iloc[:3].isna().all()
    assert np.isnan(volume["AAA"].iloc[0]) and volume["AAA"].iloc[1] == 0
    assert np.isnan(factor["AAA"].iloc[0])
    assert audit["economic_invalid_observations"]["count"] == 5
    assert audit["economic_invalid_observations"]["per_ticker_field"]["AAA"]["close"] == 3
    assert audit["economic_invalid_observations"]["per_date_field"]["2020-01-02"] == {
        "close": 1,
        "volume": 1,
        "factor": 1,
    }
    assert source.read_bytes() == before
    assert audit["binary_start_indices"]["BBB"]["close"] == 1


def test_split_adjusted_units_recover_dollars_and_shares(tmp_path):
    # Hand-worked two-for-one split: dollar price 100 -> 50, shares 100 -> 200.
    # The provider's adjusted-return index and adjusted volume stay unchanged.
    source = archive(
        tmp_path,
        {
            "features/aaa/close.day.bin": payload(0, [1, 1, 1, 1]),
            "features/aaa/factor.day.bin": payload(0, [0.01, 0.02, 0.02, 0.02]),
            "features/aaa/volume.day.bin": payload(0, [10000, 10000, 10000, 10000]),
        },
    )
    prices, volume, factor, _, _ = load_qlib_archive(source)
    np.testing.assert_allclose((prices / factor)["AAA"].iloc[:2], [100, 50], rtol=1e-6)
    np.testing.assert_allclose((volume * factor)["AAA"].iloc[:2], [100, 200], rtol=1e-6)
    np.testing.assert_allclose((prices * volume)["AAA"].iloc[:2], [10000, 10000])
    assert prices["AAA"].pct_change(fill_method=None).iloc[1] == 0


def test_explicit_symbol_mapping_recovers_lookup_and_audits_baseline(tmp_path):
    overrides = {}
    for field in ("open", "high", "low", "close", "volume", "factor"):
        overrides[f"features/dead-vendor/{field}.day.bin"] = payload(0, [1, 2, 3, 4])
    source = archive(tmp_path, overrides)
    before = source.read_bytes()
    literal, _, _, literal_member, literal_audit = load_qlib_archive(source)
    mapped, _, _, member, audit = load_qlib_archive(source, symbol_map={"DEAD": "DEAD-VENDOR"})
    assert literal["DEAD"].isna().all()
    assert mapped["DEAD"].tolist() == [1, 2, 3, 4]
    assert mapped.columns.equals(literal.columns)
    assert member.equals(literal_member)
    assert audit["vendor_symbol_map"] == {"DEAD": "DEAD-VENDOR"}
    assert audit["symbol_map_comparison"]["recovered_names"] == ["DEAD"]
    assert audit["symbol_map_comparison"]["recovered_member_sessions"] == 4
    assert audit["symbol_map_comparison"]["lost_member_sessions"] == 0
    assert (
        audit["symbol_map_comparison"]["member_close_sessions_before_mapping"]
        == literal_audit["membership_observations_with_close"]
    )
    assert (
        audit["symbol_map_comparison"]["member_close_sessions_after_mapping"]
        == audit["membership_observations_with_close"]
    )
    assert source.read_bytes() == before
    assert audit["source_sha256"] == literal_audit["source_sha256"]
    assert audit["vendor_symbol_map_sha256"] != literal_audit["vendor_symbol_map_sha256"]


@pytest.mark.parametrize(
    ("symbol_map", "error"),
    [
        ({"AAA": "BBB"}, "collision"),
        ({"AAA": "bbb"}, "collision"),
        ({"NOT-IN-MEMBERSHIP": "AAA"}, "Unknown membership"),
        ({"AAA": "../AAA"}, "Unsafe vendor"),
        ({"AAA": ".."}, "Unsafe vendor"),
        ({"AAA": "UNAVAILABLE"}, "vendor close member is missing"),
        ({"AAA": 123}, "must be strings"),
        (["AAA", "BBB"], "must be a dictionary"),
    ],
)
def test_bad_or_ambiguous_symbol_map_is_rejected(tmp_path, symbol_map, error):
    with pytest.raises((ValueError, TypeError), match=error):
        load_qlib_archive(archive(tmp_path), symbol_map=symbol_map)


def test_membership_quality_finding_is_scoped_to_verified_archive_hash(tmp_path):
    intervals = {"BRK.B": [(pd.Timestamp("1999-01-01"), pd.Timestamp("2099-12-31"))]}
    actual_hash = "20e3283009784843dcfe690488bf3aa739e64159f1da51a9ec33dd3fb647187f"
    known = classify_membership_quality(intervals, actual_hash)
    assert known["status"] == "known_inaccurate"
    assert known["historical_index_universe_claim_supported"] is False
    assert known["known_issues"][0]["source_first_membership_start"] == "1999-01-01"
    assert known["known_issues"][0]["issuer_new_inclusion_announcement_date"] == "2010-01-26"
    assert known["known_issues"][0]["source_url"] == "https://www.berkshirehathaway.com/news/jan2810.pdf"
    unknown = classify_membership_quality(intervals, "0" * 64)
    assert unknown["status"] == "unverified" and not unknown["known_issues"]
    _, _, _, _, fixture_audit = load_qlib_archive(archive(tmp_path))
    assert fixture_audit["membership_quality"]["status"] == "unverified"
