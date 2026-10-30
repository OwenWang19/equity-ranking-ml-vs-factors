"""As-of scoring for trusted local research artifacts, without future labels."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .features import build_features, feature_columns


def predict_asof(prices: pd.DataFrame, sectors: pd.Series, bundle_path: Path, asof) -> pd.Series:
    """Score one observed decision close using only history through that close.

    ``bundle_path`` must identify a trusted local artifact written by the research
    pipeline. The score is a forecast, not an order or a guarantee of tradability.
    The panel universe and sectors must match because cross-sectional features
    depend on that information set. No forward label or future price is required.
    """
    bundle = joblib.load(Path(bundle_path))
    required = {
        "schema_version",
        "estimator",
        "feature_names",
        "training_cutoff",
        "fit_max_label_end",
        "config_sha256",
        "universe",
        "sectors",
        "packages",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Incomplete research-model bundle")
    if bundle["schema_version"] != 1:
        raise ValueError("Unsupported research-model bundle schema")
    trained_version = bundle["packages"].get("scikit-learn")
    if trained_version != metadata.version("scikit-learn"):
        raise ValueError("scikit-learn version differs from the fitted research artifact")

    decision = pd.Timestamp(asof)
    if pd.isna(decision) or decision.tz is not None or decision != decision.normalize():
        raise ValueError("asof must be a timezone-naive daily date")
    if decision < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("asof precedes the model's training cutoff")
    if pd.Timestamp(bundle["fit_max_label_end"]) >= decision:
        raise ValueError("Fitted labels were not fully observable before asof")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique:
        raise ValueError("Prices require unique daily date keys")
    if not prices.index.is_monotonic_increasing or prices.index.hasnans:
        raise ValueError("Prices require sorted, non-missing date keys")
    if decision not in prices.index:
        raise ValueError("asof must be an observed close; stale dates are not inferred")
    if prices.columns.tolist() != bundle["universe"]:
        raise ValueError("Prediction universe/order differs from fitted cross-sectional features")
    if not isinstance(sectors, pd.Series) or not sectors.index.equals(prices.columns):
        raise ValueError("Sector labels must exactly align with the prediction universe")
    if sectors.to_dict() != bundle["sectors"]:
        raise ValueError("Prediction sectors differ from the fitted research information set")

    history = prices.loc[prices.index <= decision]
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("As-of history must contain complete positive finite prices")
    columns = bundle["feature_names"]
    if (
        not isinstance(columns, list)
        or not columns
        or len(set(columns)) != len(columns)
        or set(columns) - set(feature_columns())
    ):
        raise ValueError("Invalid feature schema in research-model bundle")
    estimator = bundle["estimator"]
    fitted_names = getattr(estimator, "feature_names_in_", None)
    if fitted_names is not None and list(fitted_names) != columns:
        raise ValueError("Estimator feature schema disagrees with bundle metadata")
    frame = build_features(history, sectors).xs(decision, level="date")[columns]
    if frame.index.tolist() != prices.columns.tolist() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Insufficient or invalid as-of features for every universe member")
    with threadpool_limits(limits=2):
        scores = np.asarray(estimator.predict(frame), dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError("Estimator returned invalid cross-sectional scores")
    result = pd.Series(scores, index=frame.index, name="score")
    if bundle.get("target_mode") == "sector_relative":
        from .alignment import residualize_scores

        index = pd.MultiIndex.from_product([[decision], result.index], names=["date", "ticker"])
        projected = residualize_scores(pd.Series(result.to_numpy(), index=index), sectors)
        result = projected.xs(decision, level="date").rename("score")
    if not bundle.get("deployment_active", True):
        result *= 0.0
    result.attrs.update(
        {
            "asof": decision.date().isoformat(),
            "config_sha256": bundle["config_sha256"],
            "fit_max_label_end": bundle["fit_max_label_end"],
            "scope": "research_forecast_only",
            "deployment_active": bundle.get("deployment_active", True),
            "deployment_gate_reason": bundle.get("deployment_gate_reason", "legacy_no_gate"),
            "candidate": bundle.get("candidate"),
        }
    )
    return result
