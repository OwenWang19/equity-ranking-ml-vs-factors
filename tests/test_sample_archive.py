"""Small archived examples are invented schemas, never market-alpha evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_alpha.wide_data import adjustment_factors, features_and_labels

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def manifest():
    return json.loads((EXAMPLES / "sample_manifest.json").read_text())


def test_example_manifest_hashes_fields_types_and_counts():
    specification = manifest()
    assert "SYNTHETIC ONLY" in specification["origin"]
    assert specification["contains_vendor_data"] is False
    assert specification["used_for_reported_market_results"] is False
    assert specification["sufficient_for_registered_model_training"] is False
    assert "no NYSE" in specification["calendar"]
    for name, schema in specification["files"].items():
        data = (EXAMPLES / name).read_bytes()
        frame = pd.read_csv(EXAMPLES / name)
        assert len(frame) == schema["rows"]
        assert frame.columns.tolist() == [field["name"] for field in schema["fields"]]
        assert all(field["type"] and field["units_or_meaning"] for field in schema["fields"])
        assert len(data) == schema["bytes"]
        assert hashlib.sha256(data).hexdigest() == schema["sha256"]
    assert sum(path.stat().st_size for path in EXAMPLES.iterdir() if path.is_file()) < 150_000


def test_invented_ohlcv_and_identity_contract():
    bars = pd.read_csv(EXAMPLES / "daily_bars.csv", parse_dates=["date"])
    assets = pd.read_csv(EXAMPLES / "assets.csv")
    assert bars.columns.tolist() == ["date", "asset_id", "open", "high", "low", "close", "volume"]
    assert assets.columns.tolist() == ["asset_id", "sid", "symbol"]
    assert len(bars) == 420 and len(assets) == 3
    assert bars.date.nunique() == 140
    assert not bars.duplicated(["date", "asset_id"]).any()
    assert set(bars.asset_id) == set(assets.asset_id)
    assert assets.asset_id.str.startswith("SYNTHETIC-").all()
    assert assets.symbol.str.startswith("SYNTH_").all()
    assert assets.asset_id.is_unique and assets.sid.is_unique
    assert (bars.date.dt.day_of_week < 5).all()
    assert (bars.groupby("asset_id").size() == 140).all()
    assert (bars[["open", "high", "low", "close"]] > 0).all().all()
    assert (bars.volume >= 0).all() and pd.api.types.is_integer_dtype(bars.volume)
    assert (bars.high >= bars[["open", "close", "low"]].max(axis=1)).all()
    assert (bars.low <= bars[["open", "close", "high"]].min(axis=1)).all()


def test_synthetic_action_joins_and_dividend_convention():
    bars = pd.read_csv(EXAMPLES / "daily_bars.csv", parse_dates=["date"])
    assets = pd.read_csv(EXAMPLES / "assets.csv")
    for name in ("split_ratios.csv", "dividend_ratios.csv", "merger_ratios.csv"):
        table = pd.read_csv(EXAMPLES / name)
        assert table.columns.tolist() == ["sid", "effective_date", "ratio"]
        assert set(table.sid).issubset(set(assets.sid))
        assert (table.ratio > 0).all()
    dividend = pd.read_csv(EXAMPLES / "dividend_ratios.csv", parse_dates=["effective_date"]).iloc[0]
    payout = pd.read_csv(EXAMPLES / "dividend_payouts.csv", parse_dates=["ex_date"]).iloc[0]
    assert dividend.sid == payout.sid and dividend.effective_date == payout.ex_date
    identity = assets.loc[assets.sid == payout.sid, "asset_id"].item()
    before = bars.loc[(bars.asset_id == identity) & (bars.date < payout.ex_date)].iloc[-1].close
    assert 1 - payout.amount / before == pytest.approx(dividend.ratio, abs=1e-12)
    assert len(pd.read_csv(EXAMPLES / "merger_ratios.csv")) == 0


def test_tiny_examples_exercise_feature_warmup_without_model_fits():
    bars = pd.read_csv(EXAMPLES / "daily_bars.csv", parse_dates=["date"])
    assets = pd.read_csv(EXAMPLES / "assets.csv")
    dates = pd.DatetimeIndex(sorted(bars.date.unique()))
    identities = assets.asset_id.tolist()
    raw = {
        field: bars.pivot(index="date", columns="asset_id", values=field)
        .reindex(columns=identities)
        .to_numpy()
        for field in ("open", "high", "low", "close", "volume")
    }
    split, dividend, merger, payout = [
        pd.read_csv(EXAMPLES / name)
        for name in ("split_ratios.csv", "dividend_ratios.csv", "merger_ratios.csv", "dividend_payouts.csv")
    ]
    qprice, qsplit, bad, audit = adjustment_factors(
        dates, assets.sid.to_numpy(), split, dividend, merger, payout
    )
    assert audit["unresolved_action_intervals"] == 0
    result = features_and_labels(raw, dates, qprice, qsplit, bad)
    assert result["features"].shape == (140, 3, 12)
    assert result["history_valid"][126:].all()
    assert (result["label_status"][-6:] == 4).all()
    assert np.isnan(result["label_proxy"][-6:]).all()
    assert np.isfinite(result["label_proxy"][:-6]).all()


def test_fixture_generator_reproduces_exact_public_examples():
    spec = importlib.util.spec_from_file_location(
        "synthetic_archive_examples", EXAMPLES / "generate_sample.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, text in module.sample_files().items():
        assert (EXAMPLES / name).read_bytes() == text.encode("utf-8")
