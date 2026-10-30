"""Offline fixtures for public-data acquisition; no network access in tests."""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from equity_alpha.public_data import (
    MAX_BYTES,
    START,
    acquire,
    audit_stock_sample,
    development_benchmarks,
    read_french_daily,
)


def french_zip(tmp_path, content, name="daily.csv"):
    target = tmp_path / "factors.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(name, content)
    return target


def test_french_percent_units_sentinels_and_no_annual_rows(tmp_path):
    source = french_zip(
        tmp_path,
        "Preamble\n,Mom\n20050103,1.25\n20050104,-99.99\n"
        "20050105,-999.00\n20050106,-2.00\n\n2005,99.9\nCopyright\n",
    )
    before = source.read_bytes()
    frame, audit = read_french_daily(source, ["Mom"])
    assert frame.index.strftime("%Y-%m-%d").tolist() == [
        "2005-01-03",
        "2005-01-04",
        "2005-01-05",
        "2005-01-06",
    ]
    assert frame.iloc[0, 0] == pytest.approx(0.0125)
    assert frame.iloc[3, 0] == pytest.approx(-0.02)
    assert frame.iloc[1:3, 0].isna().all()
    assert audit["sentinel_missing_count"] == 2
    assert audit["not_point_in_time_vintages"]
    assert before == source.read_bytes()


@pytest.mark.parametrize("rows", ["20050103,1\n20050103,2", "20050104,1\n20050103,2"])
def test_rejects_duplicate_or_unordered_daily_dates(tmp_path, rows):
    source = french_zip(tmp_path, ",Mom\n" + rows)
    with pytest.raises(ValueError, match="unique and increasing"):
        read_french_daily(source, ["Mom"])


@pytest.mark.parametrize(
    "body", [",Wrong\n20050103,1", ",Mom\n20050103,nan", ",Mom\n20050103,1,2", ",Mom\n20050231,1"]
)
def test_rejects_bad_factor_schema_or_values(tmp_path, body):
    with pytest.raises(ValueError):
        read_french_daily(french_zip(tmp_path, body), ["Mom"])


def test_rejects_large_uncompressed_zip_member(tmp_path):
    source = french_zip(tmp_path, b"x" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        read_french_daily(source, ["Mom"])


def factor_frames():
    dates = pd.to_datetime(["2004-12-31", "2005-01-03", "2024-12-31", "2025-01-02"])
    ff5 = pd.DataFrame(
        np.arange(24).reshape(4, 6) / 100, index=dates, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    )
    mom = pd.DataFrame({"Mom": [0.1, 0.2, 0.3, 0.4]}, index=dates)
    return ff5, mom


def test_only_development_data_emitted_and_rf_not_subtracted_twice():
    ff5, mom = factor_frames()
    table = development_benchmarks(ff5, mom)
    assert len(table) == 2
    assert table.index.min() >= pd.Timestamp(START)
    assert table.index.max() < pd.Timestamp("2025-01-01")
    assert table["mkt_rf"].tolist() == pytest.approx([0.06, 0.12])
    assert table["rf"].tolist() == pytest.approx([0.11, 0.17])
    assert table["mom"].tolist() == pytest.approx([0.2, 0.3])
    assert len(ff5) == len(mom) == 4  # No in-place source mutation.


def test_no_silent_factor_calendar_intersection_or_missing_value_fill():
    ff5, mom = factor_frames()
    with pytest.raises(ValueError, match="calendars differ"):
        development_benchmarks(ff5, mom.drop(pd.Timestamp("2005-01-03")))
    mom.loc["2005-01-03", "Mom"] = np.nan
    with pytest.raises(ValueError, match="missing observations"):
        development_benchmarks(ff5, mom)


def test_cannot_silently_relabel_reordered_factor_columns():
    ff5, mom = factor_frames()
    with pytest.raises(ValueError, match="column order/schema"):
        development_benchmarks(ff5[ff5.columns[::-1]], mom)
    with pytest.raises(ValueError, match="column order/schema"):
        development_benchmarks(ff5, mom.rename(columns={"Mom": "Wrong"}))


def test_unexpected_middle_row_is_not_silently_truncated(tmp_path):
    source = french_zip(tmp_path, ",Mom\n20050103,1\nBADROW,2\n20050105,3\n")
    with pytest.raises(ValueError, match="Unexpected row"):
        read_french_daily(source, ["Mom"])


def stock_sample(tmp_path, change=None):
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": ["2024-12-30", "2024-12-31"],
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 2000],
            "closeadj": [100, 101],
            "closeunadj": [101, 102],
        }
    )
    if change:
        frame.loc[0, change[0]] = change[1]
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_single_stock_is_not_broad_research_and_truncation_is_visible(tmp_path):
    reference = pd.to_datetime(["2005-01-03", "2024-12-30", "2024-12-31"])
    audit = audit_stock_sample(stock_sample(tmp_path), reference)
    assert audit["rows"] == 2
    assert not audit["eligible_for_cross_sectional_research"]
    assert not audit["approved_for_public_research_report"]
    assert audit["query_history_truncated"]
    assert audit["missing_reference_sessions"] == 1
    assert audit["observed_reference_coverage"] == pytest.approx(2 / 3)
    assert not audit["matches_reference_calendar"]


@pytest.mark.parametrize(
    "change", [("ticker", "LEH"), ("date", "2025-01-02"), ("volume", -1), ("close", 0), ("high", 99)]
)
def test_rejects_invalid_stock_data_and_reserved_rows(tmp_path, change):
    with pytest.raises(ValueError):
        audit_stock_sample(stock_sample(tmp_path, change))


class FakeResponse(io.BytesIO):
    def __init__(self, payload, declared=None, status=200, final_url="https://official.example/data"):
        super().__init__(payload)
        self.headers = {
            "Content-Type": "text/csv",
            "Content-Length": str(len(payload) if declared is None else declared),
        }
        self.status = status
        self.final_url = final_url

    def geturl(self):
        return self.final_url


def test_download_hash_and_no_overwrite(tmp_path):
    target = tmp_path / "raw.csv"
    body = b"a,b\n1,2\n"
    evidence = acquire("https://official.example/data", target, opener=lambda *a, **kw: FakeResponse(body))
    assert target.read_bytes() == body
    assert evidence["sha256"] == hashlib.sha256(body).hexdigest()
    assert evidence["http_status"] == 200
    with pytest.raises(FileExistsError):
        acquire("https://official.example/data", target)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(b"a", declared=5),
        FakeResponse(b""),
        FakeResponse(b"x", declared=MAX_BYTES + 1),
        FakeResponse(b"x", status=206),
        FakeResponse(b"x", final_url="http://bad.example/data"),
    ],
)
def test_download_rejects_truncation_size_status_and_https_downgrade(tmp_path, response):
    target = tmp_path / "raw.csv"
    with pytest.raises(ValueError):
        acquire("https://official.example/data", target, opener=lambda *a, **kw: response)
    assert not target.exists()
