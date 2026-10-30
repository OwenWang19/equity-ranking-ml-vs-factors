"""Execution-dated, self-financing portfolio construction and daily accounting.

Prices are positive adjusted-close total-return series. A holding is consequently a
unit of that series, rather than a literal share count through corporate actions.
This is a closing-price research approximation; it makes no claim about auction
participation, intraday fills, available borrow, volume, or executable capacity.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def construct_weights(
    scores: pd.Series,
    sectors: pd.Series,
    gross_exposure: float = 1.0,
    max_weight: float = 0.08,
) -> pd.Series:
    """Demean scores within sectors and return sector-neutral dollar weights.

    Requested gross exposure is an upper bound. If any name exceeds
    ``max_weight``, *all* weights are scaled down together, preserving neutrality
    without redistributing into weaker signals or forcing leverage. A constant
    sector, including a single-name sector, receives zero weights. Missing scores
    and missing sector assignments are errors, not implicit neutral predictions.
    """
    if not isinstance(scores, pd.Series) or not isinstance(sectors, pd.Series):
        raise TypeError("scores and sectors must be pandas Series")
    if not scores.index.is_unique or not sectors.index.is_unique:
        raise ValueError("score and sector indices must be unique")
    if scores.index.hasnans:
        raise ValueError("score indices must not contain missing names")
    if not np.isfinite(gross_exposure) or not 0 <= gross_exposure <= 1:
        raise ValueError("gross_exposure must be finite and between 0 and 1")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be finite and between 0 and 1")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("scores must be complete and finite")
    aligned_sectors = sectors.reindex(scores.index)
    if aligned_sectors.isna().any():
        raise ValueError("every score needs a non-missing sector assignment")
    # Subtract a group reference before its mean: constant decimal-valued scores
    # then produce exact zeros rather than a common floating-point residual that
    # would be normalized into a spurious directional portfolio. Global scaling
    # also avoids overflow in centering and in the absolute-weight normalizer.
    scale = float(np.abs(values).max()) if len(values) else 0.0
    scaled = scores.astype(float) / scale if scale else scores.astype(float)
    offsets = scaled - scaled.groupby(aligned_sectors).transform("first")
    centered = offsets - offsets.groupby(aligned_sectors).transform("mean")
    absolute_sum = float(centered.abs().sum())
    weights = pd.Series(0.0, index=scores.index, name="weight")
    if absolute_sum > 0 and gross_exposure > 0:
        weights = (centered * (gross_exposure / absolute_sum)).rename("weight")
        largest = float(weights.abs().max())
        if largest > max_weight:
            weights *= max_weight / largest
    return weights


def _validate_inputs(prices: pd.DataFrame, targets: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame) or not isinstance(targets, pd.DataFrame):
        raise TypeError("prices and targets must be pandas DataFrames")
    if prices.empty:
        raise ValueError("prices must contain at least one date and one asset")
    for name, frame in (("prices", prices), ("targets", targets)):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name} needs a DatetimeIndex")
        if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} dates must be non-missing, unique, and increasing")
        if not frame.index.normalize().is_unique:
            raise ValueError(f"{name} must have at most one observation per calendar day")
        if not frame.columns.is_unique or frame.columns.hasnans:
            raise ValueError(f"{name} asset columns must be non-missing and unique")
    if not prices.columns.equals(targets.columns):
        raise ValueError("targets must have exactly the same ordered asset columns as prices")
    if not targets.index.isin(prices.index).all():
        raise ValueError("every target execution date must be present in prices")
    price_values = prices.to_numpy(dtype=float)
    target_values = targets.to_numpy(dtype=float)
    if not np.isfinite(price_values).all() or np.any(price_values <= 0):
        raise ValueError("prices must be complete, positive, and finite; no implicit forward filling")
    if not np.isfinite(target_values).all():
        raise ValueError("target weights must be complete and finite")
    if np.any(np.abs(target_values).sum(axis=1) > 1 + 1e-12):
        raise ValueError("target gross exposure must not exceed 1")


def _post_cost_nav(nav: float, current_values: np.ndarray, weights: np.ndarray, fee_rate: float) -> float:
    """Solve E + fee * |w E - current dollars|_1 = pre-trade NAV."""
    if fee_rate == 0:
        return nav
    if fee_rate * float(np.abs(current_values).sum()) >= nav:
        raise ValueError("portfolio cannot fund transaction costs without non-positive equity")
    # The function is strictly increasing because fee_rate * |w|_1 < 1.
    low, high = 0.0, nav
    for _ in range(80):
        middle = (low + high) / 2
        residual = middle + fee_rate * float(np.abs(weights * middle - current_values).sum()) - nav
        if residual > 0:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def simulate_portfolio(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    one_way_cost_bps: float = 5.0,
    annual_short_borrow_bps: float = 100.0,
    liquidate: bool = True,
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """Mark holdings to market daily and trade only on explicit execution dates.

    Initial cash is 1. Existing holdings earn the current close-to-close price
    move; ACT/365 borrow is deducted; target weights then execute at that close.
    Targets must already reflect signal availability and execution delay: this
    function deliberately cannot infer when a score became observable. It never
    shifts a target or forward-fills a missing price. Holdings drift between trades.

    Trade targets are fractions of *post-fee* NAV, obtained by solving the cash
    equation including the actual absolute traded notional. ``turnover`` is that
    notional divided by prior closing NAV (initially 1), so transaction-cost return
    equals ``turnover * one_way_cost_bps / 10_000``. It is two-sided gross traded
    notional, not half-turnover. Absolute costs and cash are in initial-NAV units.
    Borrow uses previous-close short market value over elapsed calendar days;
    no interest is earned on cash and no rebate is assumed on short proceeds.

    With ``liquidate=True``, final-date targets are ignored, remaining holdings
    are closed once, and closing costs are charged. With one price date this
    means the portfolio remains cash. Otherwise final holdings remain invested.
    Gross/net exposure and maximum weight describe *post-trade* holdings and
    can exceed original target limits through price drift. ``gross_return`` is
    price P&L divided by actual prior NAV; ``gross_equity`` compounds these
    diagnostic returns, not a separately simulated no-cost portfolio.

    Optional ``sectors`` or ``targets.attrs['sectors']`` may supply asset sectors.
    Explicit ``sectors`` takes precedence. It enables realized ``sector_max_abs``
    measurement; absent sector metadata this column is NaN, never a fabricated zero. Per-asset closing units
    and weights are returned as ``units::<name>`` and ``weight::<name>`` columns.
    ``result.attrs['contributions']`` contains per-name *pre-cost price return*
    contributions (price P&L / actual prior NAV), and ``result.attrs['weights']``
    contains closing post-trade weights. Costs and borrow are not allocated by name.
    """
    _validate_inputs(prices, targets)
    if not np.isfinite(one_way_cost_bps) or not 0 <= one_way_cost_bps < 10_000:
        raise ValueError("one_way_cost_bps must be finite and in [0, 10000)")
    if not np.isfinite(annual_short_borrow_bps) or annual_short_borrow_bps < 0:
        raise ValueError("annual_short_borrow_bps must be finite and non-negative")
    names = [str(name) for name in prices.columns]
    if len(set(names)) != len(names):
        raise ValueError("asset names must have unique string representations")
    sector_data = sectors if sectors is not None else targets.attrs.get("sectors")
    sectors = None
    if sector_data is not None:
        if not isinstance(sector_data, (pd.Series, Mapping)):
            raise TypeError("targets.attrs['sectors'] must be a Series or mapping")
        sectors = pd.Series(sector_data).reindex(prices.columns)
        if sectors.isna().any() or not sectors.index.is_unique:
            raise ValueError("sector metadata must cover every asset exactly once")

    fee_rate = one_way_cost_bps / 10_000
    borrow_rate = annual_short_borrow_bps / 10_000
    units = np.zeros(len(prices.columns), dtype=float)
    cash = prior_nav = gross_equity = 1.0
    prior_price = prices.iloc[0].to_numpy(dtype=float)
    target_lookup = {date: row.to_numpy(dtype=float) for date, row in targets.iterrows()}
    records = []
    contribution_records = []
    weight_records = []

    for index, (date, row) in enumerate(prices.iterrows()):
        price = row.to_numpy(dtype=float)
        price_pnl_by_name = units * (price - prior_price) if index else np.zeros_like(units)
        price_pnl = float(price_pnl_by_name.sum())
        contribution_records.append(price_pnl_by_name / prior_nav)
        calendar_days = (date.date() - prices.index[index - 1].date()).days if index else 0
        previous_short_notional = float(-np.minimum(units * prior_price, 0).sum())
        borrow_cost = previous_short_notional * borrow_rate * calendar_days / 365
        cash -= borrow_cost
        current_values = units * price
        nav_before_trade = float(cash + current_values.sum())
        if not np.isfinite(nav_before_trade) or nav_before_trade <= 0:
            raise ValueError(f"portfolio equity is non-positive on {date}")

        is_final_liquidation = liquidate and index == len(prices) - 1
        target = np.zeros_like(units) if is_final_liquidation else target_lookup.get(date)
        traded_notional = transaction_cost = 0.0
        if target is not None:
            after_cost_nav = _post_cost_nav(nav_before_trade, current_values, target, fee_rate)
            target_values = target * after_cost_nav
            trade_values = target_values - current_values
            traded_notional = float(np.abs(trade_values).sum())
            transaction_cost = traded_notional * fee_rate
            cash -= float(trade_values.sum()) + transaction_cost
            units = target_values / price
            current_values = target_values

        nav = float(cash + current_values.sum())
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"portfolio equity is non-positive after costs on {date}")
        weights = current_values / nav
        weight_records.append(weights.copy())
        gross_return = price_pnl / prior_nav
        net_return = nav / prior_nav - 1
        transaction_cost_fraction = transaction_cost / prior_nav
        borrow_cost_fraction = borrow_cost / prior_nav
        if not np.isclose(
            net_return,
            gross_return - transaction_cost_fraction - borrow_cost_fraction,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ArithmeticError(f"self-financing return reconciliation failed on {date}")
        gross_equity *= 1 + gross_return
        sector_max_abs = np.nan
        if sectors is not None:
            sector_max_abs = float(
                pd.Series(weights, index=prices.columns).groupby(sectors).sum().abs().max()
            )
        record = {
            "return": net_return,
            "equity": nav,
            "gross_return": gross_return,
            "gross_equity": gross_equity,
            "turnover": traded_notional / prior_nav,
            "traded_notional": traded_notional,
            "transaction_cost": transaction_cost,
            "transaction_cost_fraction": transaction_cost_fraction,
            "borrow_cost": borrow_cost,
            "borrow_cost_fraction": borrow_cost_fraction,
            "price_pnl": price_pnl,
            "cash": cash,
            "gross_exposure": float(np.abs(weights).sum()),
            "net_exposure": float(weights.sum()),
            "short_exposure": float(-np.minimum(weights, 0).sum()),
            "max_abs_weight": float(np.abs(weights).max()),
            "sector_max_abs": sector_max_abs,
        }
        record.update({f"units::{name}": float(value) for name, value in zip(names, units, strict=True)})
        record.update({f"weight::{name}": float(value) for name, value in zip(names, weights, strict=True)})
        records.append(record)
        prior_nav, prior_price = nav, price

    result = pd.DataFrame(records, index=prices.index)
    result.attrs["accounting"] = (
        "Initial cash=1; adjusted-close total-return units; explicit close execution; "
        "post-cost NAV targets; prior-NAV gross turnover; ACT/365 prior-close short borrow."
    )
    result.attrs["contributions"] = pd.DataFrame(
        contribution_records,
        index=prices.index,
        columns=prices.columns,
    )
    result.attrs["contribution_units"] = "pre-cost price P&L divided by actual prior closing NAV"
    result.attrs["weights"] = pd.DataFrame(weight_records, index=prices.index, columns=prices.columns)
    return result
