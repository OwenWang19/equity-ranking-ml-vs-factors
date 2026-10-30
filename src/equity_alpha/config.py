"""A small, strict, serializable experiment specification."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    experiment_id: str = "price_alpha_v1"
    execution_delay: int = 1
    horizon: int = 5
    rebalance_every: int = 5
    test_years: tuple[int, ...] = (2022, 2023, 2024)
    holdout_start: str = "2025-01-01"
    feature_lookback: int = 252
    min_train_dates: int = 252
    min_validation_dates: int = 120
    seed: int = 1729
    gross_exposure: float = 1.0
    max_weight: float = 0.08
    one_way_cost_bps: float = 5.0
    annual_short_borrow_bps: float = 100.0
    cost_grid_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    bootstrap_samples: int = 1000
    bootstrap_block_sessions: int = 20
    feature_groups: tuple[str, ...] = ("momentum", "reversal", "risk", "relative")
    models: tuple[str, ...] = ("ridge_1", "ridge_100", "hgb_shallow", "hgb_medium")
    target_mode: str = "panel_demean"
    selection_metric: str = "rank_ic"
    cash_hurdle: bool = False

    def __post_init__(self):
        date.fromisoformat(self.holdout_start)
        if self.target_mode not in ("panel_demean", "sector_relative"):
            raise ValueError("Unknown prediction target")
        if self.selection_metric not in ("rank_ic", "sector_mse"):
            raise ValueError("Unknown model-selection metric")
        if self.selection_metric == "sector_mse" and self.target_mode != "sector_relative":
            raise ValueError("Sector-MSE selection requires a sector-relative target")
        if self.cash_hurdle and self.selection_metric != "sector_mse":
            raise ValueError("Cash hurdle requires the predeclared sector-MSE protocol")
        if self.execution_delay < 1 or self.horizon < 1:
            raise ValueError("Require positive execution delay and horizon")
        if self.rebalance_every != self.horizon:
            raise ValueError("This experiment requires non-overlapping holding intervals")
        if self.feature_lookback != 252:
            raise ValueError("Feature specification v1 has a fixed 252-session warmup")
        if not self.test_years or tuple(sorted(set(self.test_years))) != self.test_years:
            raise ValueError("test_years must be unique and increasing")
        if str(self.test_years[-1] + 1) + "-01-01" > self.holdout_start:
            raise ValueError("Development tests overlap the reserved holdout")
        if not 0 < self.gross_exposure <= 1 or not 0 < self.max_weight <= 0.5:
            raise ValueError("Invalid portfolio constraints")
        if min(self.one_way_cost_bps, self.annual_short_borrow_bps, *self.cost_grid_bps) < 0:
            raise ValueError("Costs cannot be negative")
        numeric = (
            self.gross_exposure,
            self.max_weight,
            self.one_way_cost_bps,
            self.annual_short_borrow_bps,
            *self.cost_grid_bps,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("All numeric constraints must be finite")
        known_models = {"ridge_1", "ridge_100", "hgb_shallow", "hgb_medium"}
        if not self.models or set(self.models) - known_models or len(set(self.models)) != len(self.models):
            raise ValueError("Invalid candidate model set")
        if not any(m.startswith("ridge_") for m in self.models) or not any(
            m.startswith("hgb_") for m in self.models
        ):
            raise ValueError("The experiment requires both linear and tree families")
        if self.min_train_dates < 2 or self.min_validation_dates < 2 or not self.cost_grid_bps:
            raise ValueError("Invalid sample count or cost grid")
        if self.bootstrap_samples < 100 or self.bootstrap_block_sessions < 1:
            raise ValueError("Insufficient or invalid bootstrap configuration")

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def load_config(path: Path) -> ResearchConfig:
    raw = json.loads(path.read_text())
    extra = set(raw) - {f.name for f in fields(ResearchConfig)}
    if extra:
        raise ValueError(f"Unknown configuration keys: {sorted(extra)}")
    for key in ("test_years", "cost_grid_bps", "feature_groups", "models"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return ResearchConfig(**raw)
