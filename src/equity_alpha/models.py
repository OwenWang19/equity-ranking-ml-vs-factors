"""Small predeclared candidate set; all learned transforms fit on training only."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_SPECS = {
    "ridge_1": {"family": "ridge", "alpha": 1.0},
    "ridge_100": {"family": "ridge", "alpha": 100.0},
    "hgb_shallow": {"family": "hgb", "max_leaf_nodes": 7, "l2_regularization": 10.0},
    "hgb_medium": {"family": "hgb", "max_leaf_nodes": 15, "l2_regularization": 30.0},
}


def make_model(name, seed=1729):
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name}")
    spec = MODEL_SPECS[name]
    if spec["family"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"]))
    # Internal random validation/early stopping would violate temporal separation.
    return HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=spec["max_leaf_nodes"],
        min_samples_leaf=150,
        l2_regularization=spec["l2_regularization"],
        early_stopping=False,
        random_state=seed,
    )


def rule_predictions(frame):
    return {
        "momentum": frame["ramom"],
        "reversal": frame["varev"],
        "factor_composite": (frame["ramom"] + frame["varev"]) / 2,
        "zero": frame["ramom"] * 0,
    }
