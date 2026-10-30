"""Planted and null engineering controls for the chronological research pipeline.

The controls are deliberately artificial and are never evidence of market alpha.
For idiosyncratic log return x[t] = rho*x[t-1] + epsilon[t], the conditional
expectation of x[t+k] is rho**k*x[t]. Price returns observed by decision close t
therefore contain a predictable component when rho is positive. With rho=0 all
future innovations are independent of the observed history and expected simple
returns are identical across names, so expected within-sector targets are zero.

The benchmark parameters are fixed here before its first run. Recovery failure
must be reported, not repaired by searching seeds or increasing persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .alignment import daily_cross_sectional_mse, sector_rank_ic
from .config import ResearchConfig
from .experiment import run_forecasts

MARKET_INNOVATION_SD = 0.0015
SECTOR_INNOVATION_SD = 0.0015
IDIOSYNCRATIC_INNOVATION_SD = 0.0025


def generate_control_prices(
    seed=1729, rho=0.85, n_names=12, start="2016-01-04", sessions=1500
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate causal synthetic adjusted-price units and two static sectors.

    The idiosyncratic AR(1) starts in its stationary distribution. Independent
    random streams for initialization, market, sector and name innovations make
    an existing path invariant when only the requested number of sessions grows.
    Business-day dates are synthetic labels, not an actual exchange calendar.
    """
    for name, value, minimum in (("seed", seed, 0), ("n_names", n_names, 4), ("sessions", sessions, 2)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    first = pd.Timestamp(start)
    if pd.isna(first) or first.tz is not None or first != first.normalize():
        raise ValueError("start must be a timezone-naive daily date")
    streams = np.random.SeedSequence(int(seed)).spawn(4)
    initialization, market_rng, sector_rng, name_rng = [np.random.default_rng(s) for s in streams]
    names = pd.Index([f"SIM{i:03d}" for i in range(n_names)], name="ticker")
    sector_number = np.arange(n_names) % 2
    sectors = pd.Series([f"SYNTHETIC_SECTOR_{i}" for i in sector_number], index=names, name="sector")
    market = market_rng.normal(0, MARKET_INNOVATION_SD, size=sessions)
    sector = sector_rng.normal(0, SECTOR_INNOVATION_SD, size=(sessions, 2))
    innovations = name_rng.normal(0, IDIOSYNCRATIC_INNOVATION_SD, size=(sessions, n_names))
    state = initialization.normal(0, IDIOSYNCRATIC_INNOVATION_SD / np.sqrt(1 - rho**2), size=n_names)
    log_returns = np.empty((sessions, n_names))
    for date in range(sessions):
        state = rho * state + innovations[date]
        log_returns[date] = market[date] + sector[date, sector_number] + state
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(log_returns, axis=0)),
        index=pd.bdate_range(first, periods=sessions, name="date"),
        columns=names,
    )
    if not np.isfinite(prices.to_numpy()).all() or not (prices > 0).all().all():
        raise ValueError("requested synthetic horizon produces non-finite prices")
    return prices, sectors


def run_control_benchmark(output: Path) -> dict:
    """Run the frozen seed/parameters once into a new immutable artifact folder."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = ResearchConfig(
        experiment_id="synthetic_engineering_control_v1",
        test_years=(2019, 2020),
        holdout_start="2021-01-01",
        min_train_dates=200,
        min_validation_dates=200,
        models=("ridge_1", "hgb_shallow"),
        target_mode="sector_relative",
        selection_metric="sector_mse",
        cash_hurdle=True,
    )
    parameters = {"seed": 1729, "n_names": 12, "start": "2016-01-04", "sessions": 1500}
    summary = {
        "benchmark_id": "synthetic_engineering_control_v1",
        "context": "artificial engineering positive/null controls; not financial alpha evidence",
        "seed_search_performed": False,
        "generator_parameters": parameters,
        "innovation_standard_deviations": {
            "market": MARKET_INNOVATION_SD,
            "sector": SECTOR_INNOVATION_SD,
            "idiosyncratic": IDIOSYNCRATIC_INNOVATION_SD,
        },
        "config": asdict(config),
        "config_sha256": config.digest(),
        "source_sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "cases": {},
    }
    (output / "run_started.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    for name, rho in (("planted", 0.85), ("null", 0.0)):
        case_dir = output / name
        case_dir.mkdir()
        prices, sectors = generate_control_prices(rho=rho, **parameters)
        with threadpool_limits(limits=2):
            forecasts, trials, splits, importance = run_forecasts(
                prices, sectors, config, controls=False, model_dir=case_dir / "models"
            )
        forecasts.to_csv(case_dir / "forecasts.csv")
        trials.to_csv(case_dir / "trial_ledger.csv", index=False)
        splits.to_csv(case_dir / "split_audit.csv", index=False)
        importance.to_csv(case_dir / "validation_feature_importance.csv", index=False)
        cases = []
        for year in ("all", *config.test_years):
            frame = forecasts if year == "all" else forecasts.loc[forecasts.fold == year]
            zero_mse = float(daily_cross_sectional_mse(frame.zero, frame.target).mean())
            for family in ("ridge", "hgb", "selected"):
                score = frame[f"{family}_ungated"]
                mse = float(daily_cross_sectional_mse(score, frame.target).mean())
                cases.append(
                    {
                        "fold": year,
                        "model": f"{family}_ungated",
                        "dates": int(frame.index.get_level_values("date").nunique()),
                        "mean_mse_bps_squared": mse * 1e8,
                        "zero_mse_bps_squared": zero_mse * 1e8,
                        "mse_skill_vs_zero": 1 - mse / zero_mse,
                        "mean_sector_rank_ic": float(sector_rank_ic(score, frame.target, sectors).mean()),
                    }
                )
        gate_rows = trials.loc[trials.kind.str.startswith("selected_")]
        gates = [
            {
                "fold": int(row.fold),
                "procedure": row.kind.removeprefix("selected_"),
                "candidate": row.candidate,
                "active": bool(row.active),
                "reason": row.gate_reason,
            }
            for row in gate_rows.itertuples(index=False)
        ]
        summary["cases"][name] = {
            "rho": rho,
            "universe": prices.columns.tolist(),
            "sectors": sectors.to_dict(),
            "source_price_sha256": hashlib.sha256(prices.to_csv().encode()).hexdigest(),
            "predictive_metrics_before_gate": cases,
            "deployment_gates": gates,
        }
    planted_ridge = [
        row
        for row in summary["cases"]["planted"]["predictive_metrics_before_gate"]
        if row["model"] == "ridge_ungated"
    ]
    recovered = all(row["mse_skill_vs_zero"] > 0 and row["mean_sector_rank_ic"] > 0 for row in planted_ridge)
    summary["planted_ridge_recovered_in_both_folds_and_aggregate"] = recovered
    summary["interpretation"] = (
        "The pipeline recovers this deliberately strong artificial persistent signal. "
        "This validates an engineering capability only, not market predictability."
        if recovered
        else "The pipeline failed to recover the fixed planted signal consistently; investigate and retain this failure."
    )
    summary["null_interpretation"] = (
        "Finite-sample null IC or MSE skill may be positive by chance. This one frozen null run "
        "is a diagnostic, not a calibrated null distribution or significance test."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_control_benchmark(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recovered": result["planted_ridge_recovered_in_both_folds_and_aggregate"],
            }
        )
    )


if __name__ == "__main__":
    main()
