"""Research model lifecycle checks use a fitted tiny model, never a future label."""

from importlib import metadata

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from equity_alpha.experiment import _atomic_joblib_dump
from equity_alpha.features import build_features, feature_columns
from equity_alpha.inference import predict_asof


@pytest.fixture
def research_artifact(tmp_path):
    rng = np.random.default_rng(34)
    dates = pd.bdate_range("2019-01-01", periods=360)
    names = pd.Index(["A", "B", "C", "D"], name="ticker")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.015, (360, 4)), axis=0)), index=dates, columns=names
    )
    sectors = pd.Series(["x", "x", "y", "y"], index=names)
    columns = feature_columns()
    features = build_features(prices, sectors).dropna()
    training = features.loc[features.index.get_level_values("date") < dates[-20]]
    estimator = Ridge().fit(training[columns], rng.normal(size=len(training)))
    bundle = {
        "schema_version": 1,
        "estimator": estimator,
        "feature_names": columns,
        "training_cutoff": dates[-19].date().isoformat(),
        "fit_max_label_end": dates[-20].date().isoformat(),
        "config_sha256": "test-spec",
        "universe": names.tolist(),
        "sectors": sectors.to_dict(),
        "packages": {"scikit-learn": metadata.version("scikit-learn")},
    }
    path = tmp_path / "fitted.joblib"
    _atomic_joblib_dump(bundle, path)
    return prices, sectors, path, bundle


def test_latest_close_requires_no_future_data_or_labels(research_artifact):
    prices, sectors, path, bundle = research_artifact
    asof = prices.index[-1]
    scores = predict_asof(prices, sectors, path, asof)
    x = build_features(prices, sectors).xs(asof, level="date")[bundle["feature_names"]]
    np.testing.assert_array_equal(scores.to_numpy(), bundle["estimator"].predict(x))
    assert scores.attrs["scope"] == "research_forecast_only"


def test_future_changes_cannot_change_asof_prediction(research_artifact):
    prices, sectors, path, _ = research_artifact
    asof = prices.index[-10]
    altered = prices.copy()
    altered.loc[altered.index > asof] *= [10, 0.02, 4, 0.1]
    expected = predict_asof(prices.loc[:asof], sectors, path, asof)
    actual = predict_asof(altered, sectors, path, asof)
    pd.testing.assert_series_equal(actual, expected)


def test_fit_maturity_and_cutoff_are_enforced(research_artifact):
    prices, sectors, path, bundle = research_artifact
    with pytest.raises(ValueError, match="training cutoff"):
        predict_asof(prices, sectors, path, prices.index[-21])
    bundle["fit_max_label_end"] = prices.index[-1].date().isoformat()
    _atomic_joblib_dump(bundle, path)
    with pytest.raises(ValueError, match="fully observable"):
        predict_asof(prices, sectors, path, prices.index[-1])


def test_universe_and_schema_changes_are_rejected(research_artifact):
    prices, sectors, path, bundle = research_artifact
    with pytest.raises(ValueError, match="universe"):
        predict_asof(prices.iloc[:, ::-1], sectors.iloc[::-1], path, prices.index[-1])
    bundle["feature_names"] = list(reversed(bundle["feature_names"]))
    _atomic_joblib_dump(bundle, path)
    with pytest.raises(ValueError, match="feature schema"):
        predict_asof(prices, sectors, path, prices.index[-1])


def test_atomic_artifact_roundtrip_and_no_temporary_residue(research_artifact):
    _, _, path, bundle = research_artifact
    restored = joblib.load(path)
    assert restored["config_sha256"] == bundle["config_sha256"]
    assert list(path.parent.glob("*.tmp")) == []
