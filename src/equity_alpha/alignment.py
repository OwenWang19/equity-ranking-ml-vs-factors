"""Forecast diagnostics in the same sector-neutral subspace as the portfolio.

Sector means of future returns are labels only. These functions do not add any
forward information to a feature or a decision-time estimate. Classifications
are the supplied static sectors and inherit the data card's availability limits.
"""

import numpy as np
import pandas as pd


def _validate_panel(values: pd.Series, *, allow_missing: bool) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series")
    if not isinstance(values.index, pd.MultiIndex) or values.index.names != ["date", "ticker"]:
        raise ValueError("values require a MultiIndex named date, ticker")
    if not values.index.is_unique:
        raise ValueError("date/ticker observations must be unique")
    if any(values.index.get_level_values(level).hasnans for level in ("date", "ticker")):
        raise ValueError("date/ticker labels must not be missing")
    if not isinstance(values.index.get_level_values("date"), pd.DatetimeIndex):
        raise TypeError("date labels must be pandas timestamps")
    result = values.astype(float)
    array = result.to_numpy()
    if np.isinf(array).any() or (not allow_missing and np.isnan(array).any()):
        raise ValueError("values must be finite; only targets may contain missing observations")
    return result


def _aligned_sectors(values: pd.Series, sectors: pd.Series) -> pd.Series:
    if not isinstance(sectors, pd.Series):
        raise TypeError("sectors must be a pandas Series indexed by ticker")
    if not sectors.index.is_unique or sectors.index.hasnans:
        raise ValueError("sector ticker labels must be unique and non-missing")
    aligned = sectors.reindex(values.index.get_level_values("ticker"))
    if aligned.isna().any():
        raise ValueError("every ticker requires a non-missing sector label")
    return pd.Series(aligned.to_numpy(), index=values.index, name="sector")


def _project(values: pd.Series, sectors: pd.Series, *, allow_missing: bool) -> pd.Series:
    values = _validate_panel(values, allow_missing=allow_missing)
    aligned = _aligned_sectors(values, sectors)
    keys = [values.index.get_level_values("date"), aligned]
    # A reference subtraction avoids turning identical decimal forecasts into
    # tiny floating-point signals. Single-name sectors therefore become zero.
    reference = values.groupby(keys, sort=False).transform("first")
    offsets = values - reference
    projected = offsets - offsets.groupby(keys, sort=False).transform("mean")
    if allow_missing:
        # Never silently change a sector's label universe after one missing
        # return. An incomplete sector/date receives missing targets throughout.
        counts = values.groupby(keys, sort=False).transform("count")
        sizes = values.groupby(keys, sort=False).transform("size")
        projected = projected.where(counts.eq(sizes))
    return projected.rename(values.name)


def sector_relative_target(raw_forward: pd.Series, sectors: pd.Series) -> pd.Series:
    """Subtract each date's equal-weight sector mean from forward gross returns.

    Missing future returns remain unavailable. If any member of a sector/date
    has a missing outcome, that entire group is missing. A complete singleton
    sector has target zero: no within-sector relative trade is possible.
    """
    return _project(raw_forward, sectors, allow_missing=True)


def residualize_scores(scores: pd.Series, sectors: pd.Series) -> pd.Series:
    """Project complete decision-time scores onto sector-neutral directions."""
    return _project(scores, sectors, allow_missing=False)


def _paired(scores: pd.Series, targets: pd.Series) -> pd.DataFrame:
    scores = _validate_panel(scores, allow_missing=False)
    targets = _validate_panel(targets, allow_missing=True)
    if len(scores) != len(targets) or not scores.index.isin(targets.index).all():
        raise ValueError("scores and targets must contain the same date/ticker observations")
    # Explicit alignment prevents row-order differences from pairing wrong names.
    return pd.DataFrame({"score": scores, "target": targets.reindex(scores.index)}).dropna()


def sector_rank_ic(scores: pd.Series, targets: pd.Series, sectors: pd.Series) -> pd.Series:
    """Daily Spearman correlation after projecting both inputs by date/sector.

    This is a cross-sectional correlation of sector-residual ranks, not an
    average of two-name sector correlations. Untradeable singleton sectors are
    excluded. A flat score has IC zero; fewer than three eligible names or a
    flat target has undefined IC. Missing target groups do not become zeros.
    """
    forecast = residualize_scores(scores, sectors)
    outcome = sector_relative_target(targets, sectors)
    paired = _paired(forecast, outcome)
    aligned = _aligned_sectors(scores, sectors)
    keys = [scores.index.get_level_values("date"), aligned]
    counts = aligned.groupby(keys, sort=False).transform("size")
    eligible = counts.gt(1).reindex(paired.index)
    paired = paired.loc[eligible]
    result = {}
    for date in scores.index.get_level_values("date").unique().sort_values():
        group = paired.loc[paired.index.get_level_values("date") == date]
        if len(group) < 3 or group.target.nunique() < 2:
            result[date] = np.nan
        elif group.score.nunique() < 2:
            result[date] = 0.0
        else:
            result[date] = float(group.score.rank().corr(group.target.rank()))
    return pd.Series(result, name="sector_rank_ic", dtype=float).rename_axis("date")


def daily_cross_sectional_mse(scores: pd.Series, targets: pd.Series) -> pd.Series:
    """Equally weighted date-level squared error in forecast return units.

    Callers supply sector-relative return forecasts and targets. This metric is
    meaningful for calibrated return-unit predictions, not arbitrary rule ranks.
    Each date is a statistical observation; names are not independent samples.
    """
    paired = _paired(scores, targets)
    squared_error = paired.score.sub(paired.target).pow(2)
    dates = scores.index.get_level_values("date").unique().sort_values()
    result = squared_error.groupby(level="date").mean().reindex(dates)
    return result.rename("cross_sectional_mse").rename_axis("date")
