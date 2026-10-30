"""Map decision-time forecasts to delayed trades and comparable diagnostics."""

import numpy as np
import pandas as pd

from .alignment import daily_cross_sectional_mse, residualize_scores, sector_rank_ic, sector_relative_target
from .analytics import (
    bootstrap_interval,
    daily_rank_ic,
    paired_sharpe_interval,
    performance,
    realized_exposure,
)
from .portfolio import construct_weights, simulate_portfolio

META_COLUMNS = {"target", "raw_forward_return", "entry_date", "label_end", "fold"}


def make_schedule(predictions, prices, config, *, delay=None):
    lag = config.execution_delay if delay is None else delay
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution delay must be a positive number of sessions")
    decisions = predictions.index.get_level_values("date").unique().sort_values()[:: config.rebalance_every]
    rows = []
    for date in decisions:
        position = prices.index.get_indexer([date])[0]
        if position < 0:
            raise ValueError("Decision date missing from price calendar")
        if position + lag + config.horizon >= len(prices):
            continue
        rows.append(
            {
                "decision_date": date,
                "entry_date": prices.index[position + lag],
                "exit_date": prices.index[position + lag + config.horizon],
            }
        )
    if not rows:
        raise ValueError("No fully observable delayed holding interval")
    return pd.DataFrame(rows)


def backtest_scores(predictions, prices, sectors, config, name, *, cost=None, delay=None):
    schedule = make_schedule(predictions, prices, config, delay=delay)
    targets = []
    for row in schedule.itertuples(index=False):
        scores = predictions.xs(row.decision_date, level="date")[name].reindex(prices.columns)
        weights = construct_weights(scores, sectors, config.gross_exposure, config.max_weight)
        weights.name = row.entry_date
        targets.append(weights)
    target_frame = pd.DataFrame(targets).reindex(columns=prices.columns)
    target_frame.index = pd.DatetimeIndex(target_frame.index)
    sample = prices.loc[schedule.decision_date.iloc[0] : schedule.exit_date.iloc[-1]]
    daily = simulate_portfolio(
        sample,
        target_frame,
        config.one_way_cost_bps if cost is None else cost,
        config.annual_short_borrow_bps,
        liquidate=True,
        sectors=sectors,
    )
    return daily, target_frame, schedule


def evaluate(predictions, prices, sectors, config):
    names = [c for c in predictions if c not in META_COLUMNS]
    summaries, fold_metrics, uncertainty, paired, sensitivity = [], [], [], [], []
    curves, ic_columns, raw_ic_columns, targets_by_model, contributions = {}, {}, {}, {}, {}
    market = prices.pct_change(fill_method=None).mean(axis=1)
    for name in names:
        daily, targets, schedule = backtest_scores(predictions, prices, sectors, config, name)
        curves[name], targets_by_model[name] = daily, targets
        ic = (
            sector_rank_ic(predictions[name], predictions.target, sectors)
            if config.target_mode == "sector_relative"
            else daily_rank_ic(predictions[name], predictions.target)
        )
        ic_columns[name] = ic
        raw_ic = daily_rank_ic(predictions[name], predictions.raw_forward_return)
        raw_ic_columns[name] = raw_ic
        summary = {
            "model": name,
            **performance(daily),
            "mean_daily_rank_ic": float(ic.mean()),
            "mean_daily_global_raw_rank_ic": float(raw_ic.mean()),
            "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()),
            **realized_exposure(daily["return"], market),
        }
        summaries.append(summary)
        uncertainty.append(
            {
                "model": name,
                "metric": "mean_daily_rank_ic",
                **bootstrap_interval(
                    ic,
                    samples=config.bootstrap_samples,
                    block=config.bootstrap_block_sessions,
                    seed=config.seed,
                ),
            }
        )
        uncertainty.append(
            {
                "model": name,
                "metric": "net_sharpe",
                **bootstrap_interval(
                    daily["return"],
                    statistic="sharpe",
                    samples=config.bootstrap_samples,
                    block=config.bootstrap_block_sessions,
                    seed=config.seed,
                ),
            }
        )
        for year, frame in predictions.groupby("fold"):
            year_ic = ic.loc[ic.index.year == year]
            realized = daily.loc[daily.index.year == year]
            # Year return uses the actual continuing NAV; do not reset positions or costs.
            fold_metrics.append(
                {
                    "model": name,
                    "year": year,
                    "rank_ic": float(year_ic.mean()),
                    "return": float((1 + realized["return"]).prod() - 1),
                    "sessions": len(realized),
                }
            )
        contrib = daily.attrs["contributions"]
        contributions[name] = contrib.sum().rename(name)
        if name in ("selected", "ridge", "hgb", "factor_composite"):
            for cost in config.cost_grid_bps:
                stress, _, _ = backtest_scores(predictions, prices, sectors, config, name, cost=cost)
                sensitivity.append(
                    {
                        "model": name,
                        "scenario": "one_way_fee",
                        "cost_bps": cost,
                        "execution_delay": config.execution_delay,
                        **performance(stress),
                    }
                )
            stress, _, _ = backtest_scores(
                predictions, prices, sectors, config, name, delay=config.execution_delay + 1
            )
            sensitivity.append(
                {
                    "model": name,
                    "scenario": "extra_session_delay",
                    "cost_bps": config.one_way_cost_bps,
                    "execution_delay": config.execution_delay + 1,
                    **performance(stress),
                }
            )
    for a, b in (("hgb", "ridge"), ("selected", "factor_composite"), ("hgb", "factor_composite")):
        paired.append(
            {
                "model_a": a,
                "model_b": b,
                **paired_sharpe_interval(
                    curves[a]["return"],
                    curves[b]["return"],
                    samples=config.bootstrap_samples,
                    block=config.bootstrap_block_sessions,
                    seed=config.seed,
                ),
            }
        )
    # Forecast horizon diagnostics use the same frozen predictions, no retraining/selection.
    decay = []
    for horizon in (1, 5, 10, 20):
        lag = config.execution_delay
        returns = prices.shift(-lag - horizon).div(prices.shift(-lag)) - 1
        labels = returns.sub(returns.mean(axis=1), axis=0).stack(future_stack=True)
        labels.index.names = ["date", "ticker"]
        for name in ("momentum", "reversal", "factor_composite", "ridge", "hgb", "selected"):
            if config.target_mode == "sector_relative":
                labels = sector_relative_target(labels, sectors)
                paired_labels = labels.reindex(predictions.index)
                ic = sector_rank_ic(predictions[name], paired_labels, sectors)
            else:
                ic = daily_rank_ic(predictions[name], labels)
            decay.append(
                {
                    "model": name,
                    "horizon_sessions": horizon,
                    "rank_ic": float(ic.mean()),
                    "dates": int(ic.count()),
                }
            )
    quintiles = []
    for name in ("momentum", "factor_composite", "ridge", "hgb", "selected"):
        for date, group in predictions.groupby(level="date"):
            ranks = group[name].rank(method="average", pct=True)
            # Fixed rank intervals preserve score ties instead of ticker-based artificial rankings.
            bucket = np.minimum(np.ceil(ranks * 5), 5).astype(int)
            for q, subset in group.groupby(bucket):
                quintiles.append(
                    {
                        "model": name,
                        "date": date,
                        "quintile": q,
                        "forward_return": float(subset.raw_forward_return.mean()),
                        "target_forward_return": float(subset.target.mean()),
                    }
                )
    # Return-unit forecast comparison is distinct from arbitrary rule score ranks
    # and from cash-gated deployment. Never hide an ungated model's errors.
    mse_names = (
        ["ridge_ungated", "hgb_ungated", "selected_ungated"]
        if "hgb_ungated" in predictions
        else ["ridge", "hgb", "selected"]
    )
    mse_by_date = {}
    zero_mse = daily_cross_sectional_mse(predictions.zero, predictions.target)
    mse_rows = []
    for name in mse_names:
        scores = predictions[name]
        if config.target_mode == "sector_relative":
            scores = residualize_scores(scores, sectors)
        errors = daily_cross_sectional_mse(scores, predictions.target)
        mse_by_date[name] = errors
        mse_rows.append(
            {
                "model": name,
                "mean_mse_bps_squared": float(errors.mean() * 1e8),
                "zero_mse_bps_squared": float(zero_mse.mean() * 1e8),
                "oos_mse_skill_vs_zero": float(1 - errors.mean() / zero_mse.mean()),
            }
        )
    linear, tree = mse_names[:2]
    paired_mse = pd.DataFrame(
        [
            {
                "model_a": tree,
                "model_b": linear,
                "metric": "mse_reduction_bps_squared_positive_favors_a",
                **bootstrap_interval(
                    (mse_by_date[linear] - mse_by_date[tree]) * 1e8,
                    samples=config.bootstrap_samples,
                    block=config.bootstrap_block_sessions,
                    seed=config.seed,
                ),
            }
        ]
    )
    return {
        "forecast_mse": pd.DataFrame(mse_rows),
        "daily_mse": pd.DataFrame(mse_by_date),
        "paired_mse": paired_mse,
        "summary": pd.DataFrame(summaries).set_index("model"),
        "folds": pd.DataFrame(fold_metrics),
        "uncertainty": pd.DataFrame(uncertainty),
        "paired": pd.DataFrame(paired),
        "sensitivity": pd.DataFrame(sensitivity),
        "daily_ic": pd.DataFrame(ic_columns),
        "global_raw_ic": pd.DataFrame(raw_ic_columns),
        "ic_decay": pd.DataFrame(decay),
        "quintiles": pd.DataFrame(quintiles),
        "contributions": pd.DataFrame(contributions),
        "curves": curves,
        "targets": targets_by_model,
        "schedule": schedule,
    }
