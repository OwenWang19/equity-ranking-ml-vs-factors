"""Past-only price features and explicitly delayed forward-return labels.

Every transformation is either rolling within a name or cross-sectional on one
decision date. A forward label is never an input to feature preprocessing.
"""

import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "momentum": ("momentum_126_21", "momentum_252_21", "ramom"),
    "reversal": ("return_1", "return_5", "varev", "distance_ma20"),
    "risk": ("vol20", "vol60", "vol_ratio", "downside60", "beta126", "drawdown126"),
    "relative": ("sector_relative20", "panel_residual20", "correlation60"),
}


def feature_columns(groups=None):
    selected = tuple(FEATURE_GROUPS) if groups is None else tuple(groups)
    if not selected or set(selected) - set(FEATURE_GROUPS):
        raise ValueError("Unknown or empty feature group selection")
    return [name for group in selected for name in FEATURE_GROUPS[group]]


def build_features(prices: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    if not prices.index.is_monotonic_increasing or not prices.columns.equals(sectors.index):
        raise ValueError("Prices and sectors must be sorted/aligned")
    returns = prices.pct_change(fill_method=None)
    panel_return = returns.mean(axis=1)
    vol20 = returns.rolling(20, min_periods=20).std()
    vol60 = returns.rolling(60, min_periods=60).std()
    cov = returns.rolling(126, min_periods=126).cov(panel_return)
    beta = cov.div(panel_return.rolling(126).var(), axis=0)
    ret20 = prices.pct_change(20, fill_method=None)
    sector_mean = ret20.T.groupby(sectors, sort=True).transform("mean").T
    features = {
        "momentum_126_21": prices.shift(21).div(prices.shift(126)) - 1,
        "momentum_252_21": prices.shift(21).div(prices.shift(252)) - 1,
        "ramom": (prices.shift(21).div(prices.shift(126)) - 1)
        / (returns.shift(21).rolling(105).std() * np.sqrt(105)),
        "return_1": returns,
        "return_5": prices.pct_change(5, fill_method=None),
        "varev": -(prices.pct_change(21, fill_method=None)) / (vol20 * np.sqrt(21)),
        "distance_ma20": prices.div(prices.rolling(20).mean()) - 1,
        "vol20": vol20,
        "vol60": vol60,
        "vol_ratio": vol20 / vol60,
        "downside60": returns.clip(upper=0).pow(2).rolling(60).mean().pow(0.5),
        "beta126": beta,
        "drawdown126": prices.div(prices.rolling(126).max()) - 1,
        "sector_relative20": ret20 - sector_mean,
        "panel_residual20": returns.sub(beta.mul(panel_return, axis=0)).rolling(20).sum(),
        "correlation60": returns.rolling(60).corr(panel_return),
    }
    # Ranks are computed on each date independently, never across time.
    ranks = {}
    for name, values in features.items():
        values = values.replace([np.inf, -np.inf], np.nan)
        ranks[name] = values.rank(axis=1, pct=True, method="average").stack(future_stack=True)
    frame = pd.DataFrame(ranks)
    frame.index.names = ["date", "ticker"]
    return frame.sort_index()


def build_panel(prices, sectors, *, horizon=5, execution_delay=1, target_mode="panel_demean"):
    """Return feature rows, labels, and label interval metadata.

    At close t features become known. Entry is close t+delay and exit is close
    t+delay+horizon. Target is the future raw return minus that date's panel mean.
    The price-adjustment history is retrospective; data-card caveats still apply.
    """
    if horizon < 1 or execution_delay < 1:
        raise ValueError("Labels require positive horizon and execution delay")
    frame = build_features(prices, sectors)
    raw = prices.shift(-execution_delay - horizon).div(prices.shift(-execution_delay)) - 1
    target = raw.sub(raw.mean(axis=1), axis=0)
    if target_mode == "sector_relative":
        from .alignment import sector_relative_target

        raw_long = raw.stack(future_stack=True)
        raw_long.index.names = ["date", "ticker"]
        target_long = sector_relative_target(raw_long, sectors)
    elif target_mode == "panel_demean":
        target_long = target.stack(future_stack=True)
    else:
        raise ValueError("Unknown target mode")
    frame["target"] = target_long
    frame["raw_forward_return"] = raw.stack(future_stack=True)
    calendar = pd.Series(prices.index, index=prices.index)
    row_dates = frame.index.get_level_values("date")
    frame["entry_date"] = calendar.shift(-execution_delay).reindex(row_dates).to_numpy()
    frame["label_end"] = calendar.shift(-execution_delay - horizon).reindex(row_dates).to_numpy()
    frame["sector"] = sectors.reindex(frame.index.get_level_values("ticker")).to_numpy()
    return frame
