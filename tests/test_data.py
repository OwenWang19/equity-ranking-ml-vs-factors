"""Data-contract tests use hand-worked prices, not the private research source."""

import json

import numpy as np
import pandas as pd
import pytest

from equity_alpha.data import load_prices


def sample():
    return pd.DataFrame(
        [
            ("2020-01-02", "AAA", "Technology", 100.0, 0.02),
            ("2020-01-03", "AAA", "Technology", 110.0, 0.10),
            ("2020-01-06", "AAA", "Technology", 99.0, -0.10),
            ("2020-01-02", "BBB", "Utilities", 50.0, -0.03),
            ("2020-01-03", "BBB", "Utilities", 45.0, -0.10),
            ("2020-01-06", "BBB", "Utilities", 54.0, 0.20),
        ],
        columns=["date", "ticker", "sector", "adj_close", "adj_return"],
    )


def write_source(tmp_path, frame):
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    return path


def test_handworked_prices_returns_and_audit(tmp_path):
    # Row order has no semantic meaning. First supplied returns are uncheckable.
    path = write_source(tmp_path, sample().iloc[::-1])
    prices, sectors, audit = load_prices(path)
    np.testing.assert_allclose(prices.to_numpy(), [[100, 50], [110, 45], [99, 54]])
    assert prices.columns.tolist() == ["AAA", "BBB"]
    assert sectors.to_dict() == {"AAA": "Technology", "BBB": "Utilities"}
    assert audit["n_rows"] == 6
    assert audit["n_dates"] == 3
    assert audit["n_tickers"] == 2
    assert audit["return_verification_rows"] == 4
    assert audit["ignored_first_returns"] == 2
    assert audit["return_discrepancy_count"] == 0
    assert audit["start_date"] == "2020-01-02"
    assert audit["end_date"] == "2020-01-06"
    assert len(audit["source_sha256"]) == 64
    assert audit["sector_counts"] == {"Technology": 1, "Utilities": 1}
    json.dumps(audit, allow_nan=False)


def test_price_return_disagreement_is_rejected(tmp_path):
    frame = sample()
    frame.loc[1, "adj_return"] = 0.11
    with pytest.raises(ValueError, match="returns disagree with prices"):
        load_prices(write_source(tmp_path, frame))


def test_common_gaps_are_not_filled(tmp_path):
    frame = sample().drop(index=[1, 4]).copy()
    frame.loc[2, "adj_return"] = -0.01
    frame.loc[5, "adj_return"] = 0.08
    prices, _, audit = load_prices(write_source(tmp_path, frame))
    assert len(prices) == 2
    assert audit["return_verification_rows"] == 2


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "Duplicate ticker-date"),
        ("missing_calendar", "complete identical"),
        ("unstable_sector", "Sector labels must be stable"),
        ("missing_column", "Missing required"),
        ("single_ticker", "at least two tickers"),
        ("zero_price", "strictly positive"),
        ("infinite_price", "Nonfinite adj_close"),
        ("missing_price", "Nonfinite adj_close"),
        ("missing_return", "Nonfinite adj_return"),
        ("infinite_return", "Nonfinite adj_return"),
        ("minus_one_return", "greater than -1"),
        ("missing_ticker", "Missing or blank ticker"),
        ("blank_sector", "Missing or blank sector"),
        ("whitespace_ticker", "whitespace in ticker"),
        ("missing_date", "Missing date"),
        ("intraday_date", "midnight"),
        ("empty", "must contain observations"),
    ],
)
def test_rejects_bad_source_contract(tmp_path, case, message):
    frame = sample()
    if case == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif case == "missing_calendar":
        frame = frame.drop(index=5)
    elif case == "unstable_sector":
        frame.loc[1, "sector"] = "Other"
    elif case == "missing_column":
        frame = frame.drop(columns="adj_return")
    elif case == "single_ticker":
        frame = frame.loc[frame.ticker == "AAA"]
    elif case == "zero_price":
        frame.loc[1, "adj_close"] = 0.0
    elif case == "infinite_price":
        frame.loc[1, "adj_close"] = np.inf
    elif case == "missing_price":
        frame.loc[1, "adj_close"] = np.nan
    elif case == "missing_return":
        frame.loc[1, "adj_return"] = np.nan
    elif case == "infinite_return":
        frame.loc[1, "adj_return"] = np.inf
    elif case == "minus_one_return":
        frame.loc[0, "adj_return"] = -1.0
    elif case == "missing_ticker":
        frame.loc[0, "ticker"] = None
    elif case == "blank_sector":
        frame.loc[0, "sector"] = " "
    elif case == "whitespace_ticker":
        frame.loc[0, "ticker"] = " AAA"
    elif case == "missing_date":
        frame.loc[0, "date"] = None
    elif case == "intraday_date":
        frame["date"] = frame["date"] + " 12:30:00"
    elif case == "empty":
        frame = frame.iloc[:0]
    with pytest.raises(ValueError, match=message):
        load_prices(write_source(tmp_path, frame))
