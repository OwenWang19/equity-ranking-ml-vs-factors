"""Nested annual research with validation-only selection and explicit trial logs."""

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .alignment import daily_cross_sectional_mse, residualize_scores, sector_rank_ic
from .analytics import daily_rank_ic
from .features import FEATURE_GROUPS, build_panel, feature_columns
from .models import MODEL_SPECS, make_model, rule_predictions
from .selection import select_candidate
from .splits import walk_forward_folds


def within_date_permutation(y: pd.Series, seed):
    """Null control preserving date-level return distributions, breaking name alignment."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for _, values in y.groupby(level="date", sort=True):
        out.loc[values.index] = rng.permutation(values.to_numpy())
    return out


def _atomic_joblib_dump(payload, destination: Path):
    """Publish a complete local artifact without exposing a partially written file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_forecasts(prices, sectors, config, *, controls=True, model_dir: Path | None = None):
    # Holdout prices never enter development features, labels, fitting or scoring.
    dev_prices = prices.loc[prices.index < pd.Timestamp(config.holdout_start)]
    panel = build_panel(
        dev_prices,
        sectors,
        horizon=config.horizon,
        execution_delay=config.execution_delay,
        target_mode=config.target_mode,
    )
    columns = feature_columns(config.feature_groups)
    usable = panel.dropna(subset=columns + ["target", "label_end"])
    folds = walk_forward_folds(usable, config)
    predictions, trials, split_records, importance_rows = [], [], [], []
    for fold in folds:
        train = usable.loc[fold.train]
        validation = usable.loc[fold.validation]
        test = usable.loc[fold.test]
        refit = usable.loc[fold.refit]
        print(
            f"Forecast fold {fold.year}: {len(train):,} train / {len(validation):,} validation / "
            f"{len(test):,} test rows",
            flush=True,
        )
        split_records.append(
            {
                "year": fold.year,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "refit_rows": len(refit),
                "test_rows": len(test),
                "train_max_label_end": str(train.label_end.max().date()),
                "validation_max_label_end": str(validation.label_end.max().date()),
                "refit_max_label_end": str(refit.label_end.max().date()),
                "test_first_decision": str(test.index.get_level_values("date").min().date()),
                "test_last_decision": str(test.index.get_level_values("date").max().date()),
            }
        )
        validation_scores = {}
        validation_metrics = {}
        with threadpool_limits(limits=2):
            for name in config.models:
                model = make_model(name, config.seed)
                model.fit(train[columns], train.target)
                scores = pd.Series(model.predict(validation[columns]), index=validation.index)
                if config.target_mode == "sector_relative":
                    scores = residualize_scores(scores, sectors)
                    ic = sector_rank_ic(scores, validation.target, sectors)
                else:
                    ic = daily_rank_ic(scores, validation.target)
                mean_ic = float(ic.mean())
                mse = float(daily_cross_sectional_mse(scores, validation.target).mean())
                zero_mse = float(daily_cross_sectional_mse(scores * 0, validation.target).mean())
                if not np.isfinite(mean_ic):
                    raise ValueError(f"No usable validation IC for {name} in {fold.year}")
                validation_metrics[name] = {"rank_ic": mean_ic, "mse": mse, "zero_mse": zero_mse}
                validation_scores[name] = mean_ic if config.selection_metric == "rank_ic" else mse
                trials.append(
                    {
                        "fold": fold.year,
                        "candidate": name,
                        "kind": "selection_candidate",
                        "selection_metric": config.selection_metric,
                        "value": validation_scores[name],
                        "validation_rank_ic": mean_ic,
                        "validation_mse": mse,
                        "validation_zero_mse": zero_mse,
                        "parameters": json.dumps(MODEL_SPECS[name], sort_keys=True),
                    }
                )

            # Deterministic ties follow the declared candidate order, never test results.
            def choose(names, metrics=validation_metrics):
                return select_candidate(
                    names, metrics, metric=config.selection_metric, cash_hurdle=config.cash_hurdle
                )

            family_decisions = {
                "ridge": choose([n for n in config.models if MODEL_SPECS[n]["family"] == "ridge"]),
                "hgb": choose([n for n in config.models if MODEL_SPECS[n]["family"] == "hgb"]),
                "selected": choose(list(config.models)),
            }
            family_names = {family: decision["candidate"] for family, decision in family_decisions.items()}
            out = test[["target", "raw_forward_return", "entry_date", "label_end"]].copy()
            out["fold"] = fold.year
            for name, scores in rule_predictions(test).items():
                out[name] = scores
            fitted = {}
            for candidate in set(family_names.values()):
                model = make_model(candidate, config.seed)
                model.fit(refit[columns], refit.target)
                fitted[candidate] = model
            for family, candidate in family_names.items():
                scores = pd.Series(fitted[candidate].predict(test[columns]), index=test.index)
                if config.target_mode == "sector_relative":
                    scores = residualize_scores(scores, sectors)
                    out[f"{family}_ungated"] = scores
                out[family] = scores if family_decisions[family]["active"] else 0.0
                if model_dir is not None:
                    bundle = {
                        "schema_version": 1,
                        "estimator": fitted[candidate],
                        "feature_names": list(columns),
                        "fold": fold.year,
                        "family": family,
                        "candidate": candidate,
                        "test_start": fold.test_start.date().isoformat(),
                        "training_cutoff": fold.test_start.date().isoformat(),
                        "fit_max_label_end": refit.label_end.max().date().isoformat(),
                        "config_sha256": config.digest(),
                        "universe": prices.columns.tolist(),
                        "sectors": sectors.reindex(prices.columns).to_dict(),
                        "python": platform.python_version(),
                        "packages": {
                            p: metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "joblib")
                        },
                        "scope": "trusted_local_research_model_not_live_trading",
                        "target_mode": config.target_mode,
                        "deployment_active": family_decisions[family]["active"],
                        "deployment_gate_reason": family_decisions[family]["reason"],
                    }
                    _atomic_joblib_dump(bundle, Path(model_dir) / f"fold_{fold.year}_{family}.joblib")
                trials.append(
                    {
                        "fold": fold.year,
                        "candidate": candidate,
                        "kind": f"selected_{family}",
                        "selection_metric": config.selection_metric,
                        "value": validation_scores[candidate],
                        "active": family_decisions[family]["active"],
                        "gate_reason": family_decisions[family]["reason"],
                        **{f"validation_{k}": v for k, v in validation_metrics[candidate].items()},
                        "parameters": "",
                    }
                )
            # Permutation importance uses validation only, before all available labels are refitted.
            model = make_model(family_names["hgb"], config.seed)
            model.fit(train[columns], train.target)
            base_ic = validation_metrics[family_names["hgb"]]["rank_ic"]
            rng = np.random.default_rng(config.seed + fold.year)
            for group in config.feature_groups:
                altered = validation[columns].copy()
                # Permute name alignment inside each date, jointly within a feature family.
                for _, rows in altered.groupby(level="date", sort=True):
                    order = rng.permutation(len(rows))
                    altered.loc[rows.index, list(FEATURE_GROUPS[group])] = rows[
                        list(FEATURE_GROUPS[group])
                    ].to_numpy()[order]
                altered_scores = pd.Series(model.predict(altered), index=altered.index)
                altered_ic = (
                    sector_rank_ic(altered_scores, validation.target, sectors).mean()
                    if config.target_mode == "sector_relative"
                    else daily_rank_ic(altered_scores, validation.target).mean()
                )
                importance_rows.append(
                    {
                        "fold": fold.year,
                        "feature_group": group,
                        "validation_ic_drop": float(base_ic - altered_ic),
                    }
                )
            if controls:
                for offset in range(3):
                    name = f"permuted_labels_{offset}"
                    model = make_model("hgb_shallow", config.seed + offset)
                    model.fit(refit[columns], within_date_permutation(refit.target, config.seed + offset))
                    scores = pd.Series(model.predict(test[columns]), index=test.index)
                    out[name] = (
                        residualize_scores(scores, sectors)
                        if config.target_mode == "sector_relative"
                        else scores
                    )
                    trials.append(
                        {
                            "fold": fold.year,
                            "candidate": name,
                            "kind": "negative_control",
                            "selection_metric": "not_eligible_for_selection",
                            "value": None,
                            "parameters": json.dumps(MODEL_SPECS["hgb_shallow"], sort_keys=True),
                        }
                    )
                for group in config.feature_groups:
                    remaining = [c for c in columns if c not in FEATURE_GROUPS[group]]
                    model = make_model(family_names["hgb"], config.seed)
                    model.fit(refit[remaining], refit.target)
                    scores = pd.Series(model.predict(test[remaining]), index=test.index)
                    out[f"without_{group}"] = (
                        residualize_scores(scores, sectors)
                        if config.target_mode == "sector_relative"
                        else scores
                    )
                    trials.append(
                        {
                            "fold": fold.year,
                            "candidate": f"without_{group}",
                            "kind": "prespecified_ablation",
                            "selection_metric": "diagnostic_only",
                            "value": None,
                            "parameters": family_names["hgb"],
                        }
                    )
        predictions.append(out)
    return (
        pd.concat(predictions).sort_index(),
        pd.DataFrame(trials),
        pd.DataFrame(split_records),
        pd.DataFrame(importance_rows),
    )


def build_manifest(config, audit, source_root: Path, command: list[str]):
    code = {}
    files = set()
    for directory in ("src", "tests", "configs", "docs", ".github"):
        files.update(
            p
            for p in (source_root / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.update(
        source_root / name
        for name in ("pyproject.toml", "uv.lock", "README.md")
        if (source_root / name).is_file()
    )
    for path in sorted(files):
        code[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "packages": {
            p: metadata.version(p)
            for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl")
        },
        "config": asdict(config),
        "config_sha256": config.digest(),
        "data": audit,
        "source_sha256": code,
        "command": command,
        "evaluation_status": "development_walk_forward_only",
        "reserved_holdout_start": config.holdout_start,
        "holdout_performance_evaluated": False,
        "prior_data_exposure": "Source prices were used in prior coursework; reserved period is retrospective.",
        "scope": "fixed_survivor_30_name_price_only_pilot",
    }
