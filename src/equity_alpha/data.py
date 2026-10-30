"""Read immutable source prices and validate the research data contract.

The loader never downloads, fills, forward-fills, or repairs observations. A
failed check is an explicit data decision for the researcher, not permission to
silently change the sample. Source CSV files remain outside the repository.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "ticker", "sector", "adj_close", "adj_return"}
RETURN_TOLERANCE = 1e-8


def load_prices(path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return validated date × ticker prices, sector labels, and a JSON audit.

    Adjusted returns are checked against consecutive adjusted prices. Each
    stock's first supplied return has no observable denominator and is excluded
    from that comparison; downstream returns must be recomputed from prices.
    All stocks must have identical observed calendars. This validates internal
    consistency, not vendor accuracy, exchange-calendar completeness, historical
    constituent membership, or point-in-time data availability.
    """
    source_path = Path(path).expanduser().resolve(strict=True)
    source = pd.read_csv(source_path, dtype={"ticker": "string", "sector": "string"})
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")
    if source.empty:
        raise ValueError("Price source must contain observations")

    for column in ("ticker", "sector"):
        # Reject malformed identifiers rather than normalize them into collisions.
        values = source[column]
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"Missing or blank {column} labels")
        if values.str.strip().ne(values).any():
            raise ValueError(f"Leading/trailing whitespace in {column} labels")
        source[column] = values.astype(str)

    dates = pd.to_datetime(source["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing date keys")
    if not isinstance(dates.dtype, np.dtype) or dates.dt.tz is not None:
        raise ValueError("Expected timezone-naive daily date keys")
    if not dates.dt.normalize().equals(dates):
        raise ValueError("Expected daily dates at midnight, not intraday timestamps")
    source["date"] = dates
    if source.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate ticker-date keys")

    for column in ("adj_close", "adj_return"):
        source[column] = pd.to_numeric(source[column], errors="raise")
        if not np.isfinite(source[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {column} observations")
    if (source["adj_close"] <= 0).any():
        raise ValueError("Adjusted prices must be strictly positive")
    if (source["adj_return"] <= -1).any():
        raise ValueError("Adjusted returns must be greater than -1")
    if source["ticker"].nunique() < 2:
        raise ValueError("Cross-sectional research requires at least two tickers")
    if (source.groupby("ticker")["sector"].nunique() != 1).any():
        raise ValueError("Sector labels must be stable within each ticker")

    source = source.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = source.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    prices = prices.sort_index(axis=1).astype(float)
    if prices.isna().any().any():
        raise ValueError("Tickers must have complete identical observed calendars")

    reconstructed = source.groupby("ticker", sort=False)["adj_close"].pct_change(fill_method=None)
    verifiable = reconstructed.notna()
    differences = (source.loc[verifiable, "adj_return"] - reconstructed.loc[verifiable]).abs()
    discrepancies = differences > RETURN_TOLERANCE
    if discrepancies.any():
        maximum = float(differences.max())
        raise ValueError(
            f"Adjusted returns disagree with prices: {int(discrepancies.sum())} observations "
            f"exceed {RETURN_TOLERANCE:g}; maximum absolute discrepancy {maximum:.12g}"
        )

    sectors = source.groupby("ticker")["sector"].first().reindex(prices.columns)
    sectors.name = "sector"
    constraints = [
        "Fixed selected universe: historical membership and survivorship are not verified.",
        "No volume, raw OHLC, spread, borrow, or order data: capacity and market impact are not measured.",
        "No vendor as-of or publication timestamps: point-in-time availability is not established.",
        "Adjusted-close consistency does not independently verify dividends, splits, or vendor accuracy.",
        "Identical observed calendars do not prove completeness against the exchange trading calendar.",
    ]
    audit = {
        "schema_version": 1,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "n_rows": len(source),
        "n_dates": len(prices),
        "n_tickers": len(prices.columns),
        "start_date": prices.index.min().date().isoformat(),
        "end_date": prices.index.max().date().isoformat(),
        "tickers": prices.columns.tolist(),
        "sector_counts": {str(k): int(v) for k, v in sectors.value_counts().sort_index().items()},
        "return_verification_rows": int(verifiable.sum()),
        "return_discrepancy_count": int(discrepancies.sum()),
        "max_abs_return_discrepancy": float(differences.max()) if len(differences) else 0.0,
        "return_tolerance": RETURN_TOLERANCE,
        "ignored_first_returns": int((~verifiable).sum()),
        "constraints": constraints,
    }
    return prices, sectors, audit
