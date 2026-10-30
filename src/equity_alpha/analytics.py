"""Date-level forecasting diagnostics and paired block uncertainty estimates."""

import numpy as np
import pandas as pd


def daily_rank_ic(scores: pd.Series, targets: pd.Series) -> pd.Series:
    joined = pd.concat([scores.rename("score"), targets.rename("target")], axis=1).dropna()
    values = {}
    for date, group in joined.groupby(level="date", sort=True):
        if len(group) < 3 or group.target.nunique() < 2:
            values[date] = np.nan
        elif group.score.nunique() < 2:
            values[date] = 0.0
        else:
            a = group.score.rank().to_numpy()
            b = group.target.rank().to_numpy()
            values[date] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(values, name="rank_ic", dtype=float)


def sharpe(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or np.std(a, ddof=1) < 1e-14:
        return 0.0
    return float(np.mean(a) / np.std(a, ddof=1) * np.sqrt(252))


def performance(daily):
    r = daily["return"].to_numpy()
    eq = daily.equity.to_numpy()
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return {
        "sessions": len(r),
        "total_return": float(eq[-1] - 1),
        "annualized_return": float(eq[-1] ** (252 / len(r)) - 1),
        "annualized_volatility": float(np.std(r, ddof=1) * np.sqrt(252)),
        "net_sharpe": sharpe(r),
        "gross_sharpe": sharpe(daily.gross_return),
        "max_drawdown": float(np.min(eq / peaks - 1)),
        "annualized_turnover": float(daily.turnover.mean() * 252),
        "total_transaction_cost": float(daily.transaction_cost.sum()),
        "total_borrow_cost": float(daily.borrow_cost.sum()),
        "mean_gross_exposure": float(daily.gross_exposure.mean()),
        "max_abs_net_exposure": float(daily.net_exposure.abs().max()),
        "max_abs_weight_with_drift": float(daily.max_abs_weight.max()),
        "max_abs_sector_exposure_with_drift": float(daily.sector_max_abs.max()),
    }


def circular_indices(n, block, rng):
    starts = rng.integers(0, n, size=(n + block - 1) // block)
    return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]


def bootstrap_interval(values, *, statistic="mean", samples=1000, block=20, seed=1729):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < block * 2:
        raise ValueError("Too few dates for requested block uncertainty estimate")
    rng = np.random.default_rng(seed)
    fn = np.mean if statistic == "mean" else sharpe
    stats = [fn(a[circular_indices(len(a), block, rng)]) for _ in range(samples)]
    return {
        "estimate": float(fn(a)),
        "lower_95": float(np.quantile(stats, 0.025)),
        "upper_95": float(np.quantile(stats, 0.975)),
        "dates": len(a),
        "block_sessions": block,
        "resamples": samples,
    }


def paired_sharpe_interval(a, b, *, samples=1000, block=20, seed=1729):
    # Portfolio Series inherit DataFrame-valued attribution attrs. They are not
    # statistical inputs and pandas cannot compare these dictionaries on concat.
    a_clean = pd.Series(a.to_numpy(dtype=float), index=a.index, name="a")
    b_clean = pd.Series(b.to_numpy(dtype=float), index=b.index, name="b")
    x = pd.concat([a_clean, b_clean], axis=1).dropna()
    if len(x) < 2 * block:
        raise ValueError("Too few shared dates for paired comparison")
    av, bv = x.a.to_numpy(), x.b.to_numpy()
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(samples):
        ix = circular_indices(len(x), block, rng)
        stats.append(sharpe(av[ix]) - sharpe(bv[ix]))
    return {
        "estimate": sharpe(av) - sharpe(bv),
        "lower_95": float(np.quantile(stats, 0.025)),
        "upper_95": float(np.quantile(stats, 0.975)),
        "dates": len(x),
        "block_sessions": block,
        "resamples": samples,
    }


def realized_exposure(strategy_return, panel_return):
    x = pd.concat([strategy_return.rename("r"), panel_return.rename("m")], axis=1).dropna()
    design = np.c_[np.ones(len(x)), x.m.to_numpy()]
    alpha, beta = np.linalg.lstsq(design, x.r.to_numpy(), rcond=None)[0]
    return {"annualized_intercept": float(alpha * 252), "realized_panel_beta": float(beta)}
