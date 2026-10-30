"""Offline, mocked checks; no QuantRocket account, daemon, or market-data export."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "infra/quantrocket/research/audit_learning.py"
BUNDLE = "usstock-learn-1d"


@pytest.fixture
def script():
    return runpy.run_path(str(SCRIPT))


def fake_asset(sid, symbol, end="2011-12-30", auto_close=None):
    return SimpleNamespace(
        sid=sid,
        real_sid=f"FI{sid}",
        symbol=symbol,
        asset_name=f"Company {symbol}",
        country_code="US",
        start_date=pd.Timestamp("2007-01-03"),
        end_date=pd.Timestamp(end),
        auto_close_date=pd.Timestamp(auto_close) if auto_close else pd.NaT,
    )


@pytest.fixture
def bundle():
    sessions = pd.DatetimeIndex(["2007-01-03", "2008-09-12", "2011-12-30"])
    assets = [fake_asset(1, "LEH", "2008-09-12", "2008-09-15"), fake_asset(2, "KKD")]
    finder = Mock()
    finder.equities_sids = (1, 2)
    finder.retrieve_all.return_value = assets
    finder.lifetimes.return_value = pd.DataFrame(
        [[True, True], [True, True], [False, True]], index=sessions, columns=[1, 2]
    )
    finder.lookup_symbol.side_effect = lambda symbol: {"LEH": assets[0], "KKD": assets[1]}[symbol]
    reader = Mock()
    reader.sessions = sessions
    reader.first_trading_day = sessions[0]
    reader.last_available_dt = sessions[-1]

    def raw_arrays(fields, first, last, sids):
        base = np.array([[10.0, 20.0], [9.0, 21.0], [np.nan, 22.0]])[:, np.array(sids) - 1]
        volume = np.array([[100.0, 200.0], [90.0, 0.0], [0.0, 220.0]])[:, np.array(sids) - 1]
        return [base.copy(), base + 1, base - 1, base.copy(), volume]

    reader.load_raw_arrays.side_effect = raw_arrays
    return SimpleNamespace(asset_finder=finder, equity_daily_bar_reader=reader)


def test_import_does_not_need_or_load_quantrocket(script, capsys):
    assert callable(script["collect_audit"])
    assert capsys.readouterr().out == ""


def test_date_missing_and_utc_normalization(script):
    assert script["date_string"](pd.NaT) is None
    assert script["date_string"](None) is None
    assert script["date_string"](pd.Timestamp("2007-01-03", tz="UTC")) == "2007-01-03"


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["2008-01-02", "2007-01-03"],
        ["2007-01-03"] * 2,
        ["2006-12-29", "2011-12-30"],
        ["2007-01-03", "2025-01-02"],
    ],
)
def test_wrong_session_scope_is_rejected_before_reading(script, values):
    with pytest.raises(ValueError):
        script["validate_sessions"](values)


def test_derived_observation_coverage_is_not_forward_filled(script, bundle):
    report = script["collect_audit"](bundle, chunk_size=1)
    assert report["sessions"]["first"] == "2007-01-03"
    assert report["sessions"]["last"] == "2011-12-30"
    assert report["sessions"]["count"] == 3
    assert report["coverage"]["securities"] == 2
    assert report["coverage"]["common_stock_count"] is None
    assert report["coverage"]["metadata_auto_close_before_cutoff"] == 1
    assert report["coverage"]["verified_delisted_security_count"] is None
    assert report["coverage"]["observed_close_on_cutoff_right_censoring_candidates"] == 1
    assert report["observation_totals"]["observed_close_sessions"] == 5
    assert report["observation_totals"]["complete_ohlcv_sessions"] == 5
    leh = report["identity_examples"]["LEH"]["metadata_and_observation_summary"]
    kkd = report["identity_examples"]["KKD"]["metadata_and_observation_summary"]
    assert leh["last_observed_close"] == "2008-09-12"
    assert leh["complete_ohlcv_sessions"] == 2
    assert kkd["zero_volume_with_complete_ohlc_sessions"] == 1
    assert kkd["positive_volume_ohlcv_sessions"] == 2
    assert report["identity_examples"]["MON"]["resolution"] == "unresolved"
    assert report["identity_examples"]["LEH"]["company_identity_independently_verified"] is False
    bundle.asset_finder.lifetimes.assert_called_once()
    args, kwargs = bundle.asset_finder.lifetimes.call_args
    assert args[0].equals(bundle.equity_daily_bar_reader.sessions)
    assert kwargs == {"include_start_date": True, "country_codes": ("US",)}
    assert bundle.equity_daily_bar_reader.load_raw_arrays.call_count == 2
    assert report["qualifications"]["model_ready"] is False
    assert report["qualifications"]["adjusted_return_labels"] == "unqualified"
    assert report["scope"]["original_2025_holdout_protocol_modified"] is False


def test_future_auto_close_is_not_counted_as_precutoff_death(script):
    rows = [
        {"auto_close_date": "2026-01-02", "end_date": "2011-12-30", "last_observed_close": "2011-12-30"},
        {"auto_close_date": None, "end_date": "2011-12-30", "last_observed_close": None},
    ]
    result = script["summarize_lifetimes"](rows, "2011-12-30")
    assert result["metadata_auto_close_before_cutoff"] == 0
    assert result["metadata_auto_close_after_cutoff"] == 1
    assert result["metadata_end_on_cutoff"] == 2
    assert result["no_observed_close"] == 1
    assert result["verified_delisted_security_count"] is None


def test_invalid_shapes_and_ohlc_invariants(script):
    sessions = pd.DatetimeIndex(["2007-01-03"])
    with pytest.raises(ValueError, match="five"):
        script["summarize_bars"](sessions, [], 1)
    with pytest.raises(ValueError, match="shape"):
        script["summarize_bars"](sessions, [np.ones((1, 2))] * 5, 1)
    arrays = [
        np.array([[10.0]]),
        np.array([[9.0]]),
        np.array([[8.0]]),
        np.array([[10.0]]),
        np.array([[-1.0]]),
    ]
    (row,) = script["summarize_bars"](sessions, arrays, 1)
    assert row["ohlc_order_violation_sessions"] == 1
    assert row["negative_volume_sessions"] == 1
    assert row["complete_ohlcv_sessions"] == 0
    assert row["observed_close_sessions"] == 1


@pytest.mark.parametrize("chunk_size", [0, 513, -1])
def test_invalid_chunk_size_does_not_query_reader(script, bundle, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        script["collect_audit"](bundle, chunk_size)
    bundle.equity_daily_bar_reader.load_raw_arrays.assert_not_called()


def test_does_not_guess_ambiguous_identity_or_join_symbols(script, bundle):
    class MultipleSymbolsFound(LookupError):
        pass

    bundle.asset_finder.lookup_symbol.side_effect = MultipleSymbolsFound("ambiguous")
    result = script["resolve_identity_examples"](bundle.asset_finder, [])
    assert result["LEH"] == {"resolution": "unresolved", "error_type": "MultipleSymbolsFound"}


def test_only_derived_json_is_written_and_existing_directory_is_preserved(script, bundle, tmp_path):
    report = script["collect_audit"](bundle)
    target = tmp_path / "new-audit"
    destination = script["write_audit"](report, target)
    assert list(target.iterdir()) == [destination]
    saved = json.loads(destination.read_text())
    assert saved["scope"]["raw_data_exported"] is False
    assert "asset_lifetimes_and_observed_coverage" not in saved
    assert len(saved["identity_examples"]) <= 3
    example = saved["identity_examples"]["LEH"]["metadata_and_observation_summary"]
    assert "volume" not in example
    assert "close" not in example
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        script["write_audit"](report, target)
    assert destination.read_bytes() == original


def fake_platform(monkeypatch, bundle):
    api = Mock()
    api.list_bundles.return_value = {BUNDLE: True}
    api.load.return_value = bundle
    quantrocket = ModuleType("quantrocket")
    quantrocket.__path__ = []
    qr_zipline = ModuleType("quantrocket.zipline")
    qr_zipline.list_bundles = api.list_bundles
    zipline = ModuleType("zipline")
    zipline.__path__ = []
    data = ModuleType("zipline.data")
    data.bundles = SimpleNamespace(load=api.load)
    research = ModuleType("zipline.research")
    research.use_bundle = api.use_bundle
    errors = ModuleType("zipline.errors")
    errors.MultipleSymbolsFound = LookupError
    errors.SymbolNotFound = KeyError
    for name, module in {
        "quantrocket": quantrocket,
        "quantrocket.zipline": qr_zipline,
        "zipline": zipline,
        "zipline.data": data,
        "zipline.research": research,
        "zipline.errors": errors,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return api


def test_main_registers_before_load_and_uses_no_mutating_apis(script, bundle, monkeypatch, tmp_path, capsys):
    api = fake_platform(monkeypatch, bundle)
    output = tmp_path / "new"
    script["main"](["--ingestion-complete", "--output-dir", str(output)])
    assert api.mock_calls == [call.list_bundles(), call.use_bundle(BUNDLE), call.load(BUNDLE)]
    saved = json.loads((output / "learning_bundle_audit.json").read_text())
    assert saved["ingestion_completion"]["independently_verified_by_this_script"] is False
    assert json.loads(capsys.readouterr().out)["model_ready"] is False


def test_main_requires_completion_attestation_before_imports(script, tmp_path):
    with pytest.raises(SystemExit):
        script["main"](["--output-dir", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_main_refuses_not_ingested_bundle(script, bundle, monkeypatch, tmp_path):
    api = fake_platform(monkeypatch, bundle)
    api.list_bundles.return_value = {BUNDLE: False}
    with pytest.raises(RuntimeError, match="not listed as ingested"):
        script["main"](["--ingestion-complete", "--output-dir", str(tmp_path / "new")])
    api.load.assert_not_called()
    api.use_bundle.assert_not_called()
    assert not (tmp_path / "new").exists()
