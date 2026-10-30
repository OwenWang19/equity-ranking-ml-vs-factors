"""Offline preparation tests: no daemon, downloads, raw export, or model fitting."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "infra/quantrocket/research/prepare_wide.py"


@pytest.fixture
def script():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def config():
    return {
        "bundle": "usstock-learn-1d",
        "development_end_exclusive": "2011-01-01",
        "authorized_start": "2007-01-03",
        "reserved_start": "2011-01-01",
        "chunk_size": 2,
        "max_assets_per_date": 2,
        "min_price": 5.0,
        "min_adv20": 10_000_000.0,
        "warmup_sessions": 126,
        "max_cache_bytes": 2_000_000_000,
        "min_free_disk_bytes": 8_000_000_000,
        "bundle_reference_hash": "f" * 64,
        "models": ["ignored-runner-setting"],
    }


def enough_disk(_):
    return SimpleNamespace(free=20_000_000_000)


def test_import_without_platform_dependencies(script):
    assert callable(script["prepare_wide"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("bundle", "other"),
        ("development_end_exclusive", "2012-01-01"),
        ("reserved_start", "2012-01-01"),
        ("chunk_size", 257),
        ("chunk_size", 1.5),
        ("max_assets_per_date", 2501),
        ("max_cache_bytes", 2_000_000_001),
        ("min_free_disk_bytes", 7_999_999_999),
        ("min_price", np.nan),
    ],
)
def test_config_scope_and_resource_bounds(script, config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        script["validate_config"](config)


def test_reserved_sessions_truncated_before_feature_and_label_calculation(script, config):
    sessions = pd.bdate_range("2009-01-01", periods=132).append(pd.DatetimeIndex(["2011-01-03"]))
    result = script["development_sessions"](sessions, config)
    assert len(result) == 132
    assert result.max() < pd.Timestamp("2011-01-01")
    with pytest.raises(ValueError):
        script["development_sessions"](pd.DatetimeIndex(["2006-01-01", "2011-01-03"]), config)


def test_eligibility_has_no_label_status_or_future_input(script, config):
    history = np.ones((129, 3), dtype=bool)
    close = np.full((129, 3), 5.0)
    adv = np.full((129, 3), 10_000_000.0)
    close[127, 0] = 4.99
    adv[127, 1] = 9_999_999.0
    history[127, 2] = False
    eligibility = script["causal_eligibility"](history, close, adv, config)
    assert not eligibility[:126].any()
    assert eligibility[126].all()
    assert not eligibility[127].any()
    adv[128] = 1e15
    close[128] = np.nan
    changed = script["causal_eligibility"](history, close, adv, config)
    np.testing.assert_array_equal(eligibility[:128], changed[:128])
    assert list(__import__("inspect").signature(script["causal_eligibility"]).parameters) == [
        "history_valid",
        "current_close",
        "adv20",
        "config",
    ]


def test_persistent_ids_not_tickers_and_never_filter_lifetime_or_type(script):
    assets = [
        SimpleNamespace(sid=3, real_sid="FI-C"),
        SimpleNamespace(sid=1, real_sid="FI-A"),
        SimpleNamespace(sid=2, real_sid="FI-B"),
    ]
    ordered = script["persistent_asset_order"](assets)
    assert [asset.real_sid for asset in ordered] == ["FI-A", "FI-B", "FI-C"]
    assert {id(asset) for asset in ordered} == {id(asset) for asset in assets}
    assets[2].real_sid = "FI-A"
    with pytest.raises(ValueError, match="Duplicate"):
        script["persistent_asset_order"](assets)


def test_dynamic_topk_uses_only_that_day_and_stable_identity_ties(script):
    ids = np.array(["FI-C", "FI-A", "FI-B", "FI-D"])
    mask = np.ones(4, dtype=bool)
    first = script["select_daily_assets"](mask, np.array([30, 30, 30, 10]), ids, 2)
    assert ids[first].tolist() == ["FI-A", "FI-B"]
    second = script["select_daily_assets"](mask, np.array([100, 1, 2, 90]), ids, 2)
    assert ids[second].tolist() == ["FI-C", "FI-D"]
    assert ids[first].tolist() == ["FI-A", "FI-B"]
    mask[0] = False
    assert ids[script["select_daily_assets"](mask, np.array([1e9, 1, 2, 90]), ids, 2)].tolist() == [
        "FI-B",
        "FI-D",
    ]


def test_cross_sectional_rank_is_local_tie_aware_and_not_target_rank(script):
    values = np.array([[1.0, 7.0], [2.0, 7.0], [2.0, 7.0]])
    ranked = script["rank_feature_cross_section"](values)
    np.testing.assert_array_equal(ranked, np.array([[-0.5, 0], [0.25, 0], [0.25, 0]], dtype=np.float32))
    assert ranked.dtype == np.float32
    np.testing.assert_array_equal(script["rank_feature_cross_section"](values[:1]), [[0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        script["rank_feature_cross_section"](np.array([[np.nan]]))


def test_resources_fail_before_writing_and_existing_output_is_never_reused(script, config, tmp_path):
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="max_cache"):
        script["check_resources"](output, 2_000_000_001, config, enough_disk)
    with pytest.raises(OSError, match="floor"):
        script["check_resources"](output, 1_000_000, config, lambda _: SimpleNamespace(free=8_000_000_000))
    assert not output.exists()
    output.mkdir()
    marker = output / "user-owned"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        script["check_resources"](output, 1, config, enough_disk)
    assert marker.read_text() == "preserve"


def fake_platform(flip_labels=False):
    sessions = pd.bdate_range("2009-01-01", periods=132).append(pd.DatetimeIndex(["2011-01-03"]))
    assets = [
        SimpleNamespace(sid=30, real_sid="FI-C"),
        SimpleNamespace(sid=10, real_sid="FI-A"),
        SimpleNamespace(sid=20, real_sid="FI-B"),
    ]
    finder = SimpleNamespace(equities_sids=[30, 10, 20], retrieve_all=lambda _: assets)
    reader = Mock()
    reader.sessions = sessions

    def read(fields, first, last, sids):
        assert last < pd.Timestamp("2011-01-01")
        assert fields == ["open", "high", "low", "close", "volume"]
        n = int((sessions <= last).sum())
        price = np.full((n, len(sids)), 10.0)
        volume = np.broadcast_to(np.asarray(sids)[None, :] * 100_000.0, price.shape).copy()
        return [price, price + 1, price - 1, price, volume]

    reader.load_raw_arrays.side_effect = read
    adjustment = Mock()

    def events(table):
        column = "ex_date" if table == "dividend_payouts" else "effective_date"
        return pd.DataFrame({"sid": [10, 10], column: [1230854400, 1294012800]})

    adjustment.get_df_from_table.side_effect = events
    core = SimpleNamespace(
        FEATURE_NAMES=tuple(f"f{i}" for i in range(12)),
        LABEL_STATUS_NAMES=("complete", "missing", "gap", "bad", "right_boundary"),
    )

    def factors(dates, sids, *tables):
        assert dates.max() < pd.Timestamp("2011-01-01")
        for table in tables:
            column = "ex_date" if "ex_date" in table else "effective_date"
            assert table[column].max() < 1293840000 if len(table) else True
        shape = (len(dates), len(sids))
        return np.ones(shape), np.ones(shape), np.zeros(shape, dtype=bool), {"rows": len(dates)}

    def features(raw, dates, *unused):
        assert dates.max() < pd.Timestamp("2011-01-01")
        shape = raw["close"].shape
        feature = np.broadcast_to(raw["volume"][:, :, None], (*shape, 12)).astype(np.float32)
        labels = np.full(shape, -1.5 if flip_labels else 0.25, dtype=np.float32)
        statuses = np.ones(shape, dtype=np.uint8) if flip_labels else np.zeros(shape, dtype=np.uint8)
        labels[-1] = np.nan
        statuses[-1] = 4
        return {
            "features": feature,
            "adv20": raw["close"] * raw["volume"],
            "history_valid": np.ones(shape, dtype=bool),
            "label_proxy": labels,
            "label_status": statuses,
        }

    core.adjustment_factors = Mock(side_effect=factors)
    core.features_and_labels = Mock(side_effect=features)
    return SimpleNamespace(
        asset_finder=finder, equity_daily_bar_reader=reader, adjustment_reader=adjustment
    ), core


def test_full_preparation_retains_missing_labels_all_ids_and_right_boundary(script, config, tmp_path):
    bundle, core = fake_platform()
    output = tmp_path / "data-new"
    manifest = script["prepare_wide"](bundle, config, output, core, {"source": "test"}, enough_disk)
    assert manifest["dimensions"] == {"dates": 132, "assets": 3, "features": 12, "rows": 12}
    assert manifest["persistent_sids"] == ["FI-A", "FI-B", "FI-C"]
    assert bundle.equity_daily_bar_reader.load_raw_arrays.call_count == 2
    assert core.features_and_labels.call_count == 2
    assert manifest["scope"]["reserved_2011_raw_read"] is False
    assert manifest["bundle_reference"]["kind"] == "prior_audit_reference_not_recomputed"
    assert manifest["universe"]["static_classification_used"] is False
    assert manifest["universe"]["eligibility_observes_labels_or_status"] is False
    np.testing.assert_array_equal(np.load(output / "asset_ids.npy"), ["FI-A", "FI-B", "FI-C"])
    np.testing.assert_array_equal(np.load(output / "asset_index.npy"), [1, 2] * 6)
    np.testing.assert_array_equal(np.load(output / "date_index.npy"), np.repeat(np.arange(126, 132), 2))
    assert np.isnan(np.load(output / "y.npy")[-2:]).all()
    np.testing.assert_array_equal(np.load(output / "status.npy")[-2:], [4, 4])
    np.testing.assert_array_equal(np.load(output / "X.npy")[:2, 0], [-0.5, 0.5])
    assert json.loads((output / "manifest.json").read_text())["source_hashes"] == {"source": "test"}
    assert all(np.load(output / filename).dtype != object for filename in manifest["files"].values())
    with pytest.raises(FileExistsError):
        script["prepare_wide"](bundle, config, output, core, disk_usage=enough_disk)


def test_label_and_status_changes_cannot_change_selection_or_features(script, config, tmp_path):
    for flip in (False, True):
        bundle, core = fake_platform(flip_labels=flip)
        script["prepare_wide"](bundle, config, tmp_path / str(flip), core, disk_usage=enough_disk)
    for name in ("X", "eligible_wide", "date_index", "asset_index"):
        np.testing.assert_array_equal(
            np.load(tmp_path / "False" / f"{name}.npy"), np.load(tmp_path / "True" / f"{name}.npy")
        )
    assert not np.array_equal(np.load(tmp_path / "False/y.npy"), np.load(tmp_path / "True/y.npy"))


def test_real_core_api_integration_without_platform_access(script, config, tmp_path):
    from equity_alpha import wide_data

    bundle, _ = fake_platform()

    def empty_table(table):
        date = "ex_date" if table == "dividend_payouts" else "effective_date"
        return pd.DataFrame(columns=["sid", date, "ratio"])

    bundle.adjustment_reader.get_df_from_table.side_effect = empty_table
    manifest = script["prepare_wide"](
        bundle, config, tmp_path / "real-core", wide_data, disk_usage=enough_disk
    )
    assert manifest["feature_names"] == list(wide_data.FEATURE_NAMES)
    assert manifest["preparation_complete"] is True
    assert manifest["dimensions"]["rows"] == 12
    assert manifest["label_coverage"]["selected_status_counts"]["right_boundary"] == 12
    assert np.isfinite(np.load(tmp_path / "real-core/X.npy")).all()
