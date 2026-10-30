"""Bounded, registered wide-security prediction study, run inside QuantRocket.

No portfolio simulator, Sharpe estimate, raw-data export or holdout-release path
is exposed. Row-level outputs and fitted estimators remain in the local platform.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import shutil
import signal
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from .integrity import canonical_digest, file_digest
from .registry import complete_run, fail_run, read_registry, start_run
from .wide_data import FEATURE_NAMES, LABEL_STATUS_NAMES
from .wide_models import make_wide_model, permute_training_targets, train_rows_for_year
from .wide_stats import daily_summary, paired_calendar_bootstrap

ARRAY_FILES = ("X", "y", "status", "date_index", "asset_index", "dates", "asset_ids")
SOURCE_FILES = (
    "wide_experiment.py",
    "wide_data.py",
    "wide_models.py",
    "wide_stats.py",
    "registry.py",
    "integrity.py",
)


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(finite_json(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_config(config):
    required = {
        "experiment_id",
        "bundle",
        "prediction_years",
        "reserved_start",
        "models",
        "seed",
        "primary_linear",
        "primary_nonlinear",
        "permutation_seeds",
        "max_total_model_fits",
        "bootstrap_samples",
        "bootstrap_block_sessions",
        "threads",
        "max_runtime_seconds",
    }
    if required - config.keys():
        raise ValueError(f"Missing configuration: {sorted(required - config.keys())}")
    if config["bundle"] != "usstock-learn-1d" or config["prediction_years"] != [2009, 2010]:
        raise ValueError("Only the separate 2009-2010 development study is authorized")
    if config["reserved_start"] != "2011-01-01" or config.get("development_end_exclusive") != "2011-01-01":
        raise ValueError("Reserved 2011 observations cannot enter this study")
    if config.get("reserved_performance_enabled") is not False or config.get("early_stopping") is not False:
        raise ValueError("Holdout evaluation and random early stopping must remain disabled")
    if config["models"] != ["ridge_10", "ridge_100", "hgb_7", "hgb_15"]:
        raise ValueError("Candidate order/grid differs from the registered design")
    if (config["primary_linear"], config["primary_nonlinear"]) != ("ridge_100", "hgb_7"):
        raise ValueError("Primary comparison must not be selected from development results")
    if config["permutation_seeds"] != [191, 727, 2027] or config["max_total_model_fits"] != 14:
        raise ValueError("Permutation trials and total fit budget are fixed")
    if config.get("label_exit_offset") != 6 or config.get("horizon_sessions") != 5:
        raise ValueError("The delayed five-session target cannot be changed")
    if config["threads"] != 2 or not 30 <= config["max_runtime_seconds"] <= 1800:
        raise ValueError("Runtime or thread limit exceeds the registered resource budget")
    if config["bootstrap_samples"] != 1000 or config["bootstrap_block_sessions"] != 20:
        raise ValueError("Uncertainty settings differ from the registered design")
    return config


def load_cache(path, config):
    """Validate the frozen cache without selecting rows on outcome availability."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("preparation_complete") is not True:
        raise ValueError("Cache preparation did not complete")
    if manifest["config"] != config or manifest["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Cache configuration/features differ from requested protocol")
    if manifest.get("label_status_names") != list(LABEL_STATUS_NAMES):
        raise ValueError("Cache label-status schema differs from the registered implementation")
    expected_core = file_digest(Path(__file__).parent / "wide_data.py")
    if manifest.get("source_hashes", {}).get("wide_data.py") != expected_core:
        raise ValueError("Cache feature/label source differs from the current implementation")
    preparation = Path("/codeload/equity-alpha-research/prepare_wide.py")
    if not preparation.is_file():
        preparation = Path(__file__).resolve().parents[2] / "infra/quantrocket/research/prepare_wide.py"
    if manifest.get("source_hashes", {}).get("prepare_wide.py") != file_digest(preparation):
        raise ValueError("Cache preparation source differs from the current implementation")
    if manifest.get("scope", {}).get("reserved_2011_raw_read") is not False:
        raise ValueError("Cache does not attest raw reserved-period exclusion")
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in ARRAY_FILES}
    x, y = arrays["X"], arrays["y"]
    n = len(y)
    if x.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(x).all() or np.isinf(y).any():
        raise ValueError("Invalid finite feature matrix or target proxy")
    for name in ("status", "date_index", "asset_index"):
        if arrays[name].shape != (n,) or not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"Invalid {name} vector")
    if not np.array_equal(np.isfinite(y), arrays["status"] == 0):
        raise ValueError("Outcome missingness must agree exactly with label statuses")
    dates = pd.DatetimeIndex(arrays["dates"])
    if (
        dates.empty
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or dates.max() >= pd.Timestamp(config["reserved_start"])
    ):
        raise ValueError("Cache calendar is invalid or includes reserved sessions")
    if not n or arrays["date_index"].min() < 0 or arrays["date_index"].max() >= len(dates):
        raise ValueError("Row date references are invalid")
    if np.any(np.diff(arrays["date_index"].astype(np.int64)) < 0):
        raise ValueError("Cache rows must be date-major")
    if arrays["asset_index"].min() < 0 or arrays["asset_index"].max() >= len(arrays["asset_ids"]):
        raise ValueError("Row security references are invalid")
    if len(set(arrays["asset_ids"].tolist())) != len(arrays["asset_ids"]):
        raise ValueError("Duplicate persistent security IDs")
    if arrays["status"].min() < 0 or arrays["status"].max() >= len(LABEL_STATUS_NAMES):
        raise ValueError("Unknown label statuses")
    pair_ids = arrays["date_index"].astype(np.int64) * len(arrays["asset_ids"]) + arrays["asset_index"]
    if len(np.unique(pair_ids)) != n:
        raise ValueError("Duplicate decision-date/security forecast rows")
    if np.bincount(arrays["date_index"], minlength=len(dates)).max() > config["max_assets_per_date"]:
        raise ValueError("Cache exceeds the declared daily universe cap")
    fingerprints = {f"{name}.npy": file_digest(path / f"{name}.npy") for name in ARRAY_FILES}
    fingerprints["manifest.json"] = file_digest(path / "manifest.json")
    return arrays, dates, manifest, fingerprints


def append_fit_event(path, event):
    """Persist every attempted/completed fit before later steps can fail."""
    event = {"timestamp_utc": datetime.now(UTC).isoformat(), **event}
    with Path(path).open("a") as handle:
        handle.write(json.dumps(finite_json(event), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def enforce_trial_budget(registry_path, experiment_id, maximum):
    """A new run cannot silently repeat trials consumed by earlier attempts."""
    events = [row for row in read_registry(registry_path) if row["experiment_id"] == experiment_id]
    used = 0
    for event in events:
        if event["event"] != "run_started":
            continue
        later = [row for row in events if row["run_id"] == event["run_id"]]
        if later[-1]["event"] == "run_started":
            raise RuntimeError(
                "An unfinished experiment attempt exists; inspect its lifecycle before resuming"
            )
        directory = event["payload"].get("output_dir")
        if directory is None:
            raise RuntimeError("Earlier attempt has no auditable output directory")
        ledger = Path(directory) / "fit_lifecycle.jsonl"
        if ledger.exists():
            used += sum(
                json.loads(line).get("event") == "fit_started" for line in ledger.read_text().splitlines()
            )
    if used:
        raise RuntimeError(
            f"{used}/{maximum} scientific fits already consumed; no automatic repeat or seed search"
        )
    return used


def evaluate_predictions(predictions, test_dates, labels, statuses, calendar, names):
    """All predictions are evaluated against the original, potentially missing cohort."""
    daily_rows, deciles = [], []
    for position, name in enumerate(names):
        for date in calendar:
            mask = test_dates == date
            summary = daily_summary(predictions[mask, position], labels[mask], statuses[mask])
            for item in summary.pop("score_deciles"):
                deciles.append(
                    {
                        "date": date,
                        "model": name,
                        "boundary_rows": item["label_status_counts"].get("4", 0),
                        **item,
                    }
                )
            summary.pop("label_status_counts")
            summary["right_boundary_rows"] = int((statuses[mask] == 4).sum())
            daily_rows.append({"date": date, "model": name, **summary})
    return pd.DataFrame(daily_rows), pd.DataFrame(deciles)


def summarize_results(daily, deciles, calendar, config):
    names = daily.model.drop_duplicates().tolist()
    by_model = {}
    for name in names:
        part = daily.loc[daily.model == name].set_index("date").reindex(calendar)
        ic = part.observed_spearman_ic.astype(float)
        years = {}
        for year in config["prediction_years"]:
            year_part = part.loc[part.index.year == year]
            years[str(year)] = {
                "mean_observed_ic": float(year_part.observed_spearman_ic.mean()),
                "dates_with_observed_ic": int(year_part.observed_spearman_ic.notna().sum()),
                "forecast_rows": int(year_part.predicted_rows.sum()),
                "observed_label_rows": int(year_part.observed_rows.sum()),
            }
        # Fixed right-boundary censoring is not endogenous attrition; it remains
        # in daily outputs and the bootstrap calendar but is reported separately.
        known_boundary = part.right_boundary_rows.eq(part.predicted_rows) & part.predicted_rows.gt(0)
        bound_dates = ~known_boundary
        by_model[name] = {
            "mean_observed_ic": float(ic.mean()),
            "ic_interval": paired_calendar_bootstrap(
                ic,
                pd.Series(0.0, index=calendar),
                seed=config["seed"],
                samples=config["bootstrap_samples"],
                block=config["bootstrap_block_sessions"],
            ),
            "years": years,
            "mean_full_cohort_kendall_lower_nonboundary": float(
                part.loc[bound_dates, "kendall_tau_a_lower"].mean()
            ),
            "mean_full_cohort_kendall_upper_nonboundary": float(
                part.loc[bound_dates, "kendall_tau_a_upper"].mean()
            ),
            "mean_full_cohort_kendall_lower_all_dates": float(part.kendall_tau_a_lower.mean()),
            "mean_full_cohort_kendall_upper_all_dates": float(part.kendall_tau_a_upper.mean()),
            "forecast_rows": int(part.predicted_rows.sum()),
            "observed_label_rows": int(part.observed_rows.sum()),
        }
    pivot = daily.pivot(index="date", columns="model", values="observed_spearman_ic").reindex(calendar)
    delta = paired_calendar_bootstrap(
        pivot[config["primary_nonlinear"]],
        pivot[config["primary_linear"]],
        seed=config["seed"],
        samples=config["bootstrap_samples"],
        block=config["bootstrap_block_sessions"],
    )
    if not all(np.isfinite(by_model[name]["mean_observed_ic"]) for name in config["models"]):
        raise ValueError("Undefined candidate IC; cannot rank development models")
    candidates = sorted(
        config["models"], key=lambda n: (-by_model[n]["mean_observed_ic"], config["models"].index(n))
    )
    coverage = deciles.groupby(["model", "decile"], as_index=False)[
        ["predicted_rows", "observed_rows", "missing_rows", "boundary_rows"]
    ].sum()
    coverage["coverage"] = coverage.observed_rows / coverage.predicted_rows.replace(0, np.nan)
    coverage["coverage_excluding_fixed_boundary"] = coverage.observed_rows / (
        coverage.predicted_rows - coverage.boundary_rows
    ).replace(0, np.nan)
    hgb = by_model[config["primary_nonlinear"]]
    permutation_means = {n: v["mean_observed_ic"] for n, v in by_model.items() if n.startswith("null_hgb_")}
    return {
        "study": "2009-2010 development; 2011 reserved performance not evaluated",
        "target": config["target_name"],
        "universe": config["universe_name"],
        "models": by_model,
        "primary_paired_ic_difference": delta,
        "primary_paired_ic_difference_by_year": {
            str(year): float(
                (pivot[config["primary_nonlinear"]] - pivot[config["primary_linear"]])
                .loc[calendar.year == year]
                .mean()
            )
            for year in config["prediction_years"]
        },
        "development_selected_candidate": candidates[0],
        "selection_is_not_confirmatory": True,
        "null_control_means": permutation_means,
        "evidence_checks": {
            "fixed_primary_delta_ci_excludes_zero_positive": delta["lower_95"] > 0,
            "fixed_primary_ic_ci_excludes_zero_positive": hgb["ic_interval"]["lower_95"] > 0,
            "fixed_primary_exceeds_all_permuted_training_means": all(
                hgb["mean_observed_ic"] > v for v in permutation_means.values()
            ),
            "missing_outcome_kendall_lower_positive_nonboundary": hgb[
                "mean_full_cohort_kendall_lower_nonboundary"
            ]
            > 0,
        },
        "model_or_trading_promotion_authorized": False,
        "net_sharpe_computed": False,
        "reserved_performance_computed": False,
        "coverage_by_score_decile": coverage.to_dict(orient="records"),
        "limitations": config["data_scope_limitations"],
        "interval_scope": "Calendar-block intervals conditional on this vintage/universe; do not correct selection or data defects",
    }


def write_report(output, summary, daily, config):
    """Only aggregate figures/statistics, no per-security examples or prices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    pivot = daily.pivot(index="date", columns="model", values="observed_spearman_ic")
    for name in (config["primary_linear"], config["primary_nonlinear"], "momentum21", "reversal5"):
        axes[0, 0].plot(pivot.index, pivot[name].rolling(63, min_periods=40).mean(), label=name, lw=1.3)
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_title("63-session mean observed-case Rank IC (not PnL)")
    axes[0, 0].legend(fontsize=8)
    gap = pivot[config["primary_nonlinear"]] - pivot[config["primary_linear"]]
    axes[0, 1].plot(gap.index, gap.rolling(63, min_periods=40).mean(), color="#086788")
    axes[0, 1].axhline(0, color="black", lw=0.6)
    axes[0, 1].set_title("Fixed HGB7 minus Ridge100: rolling IC difference")
    part = daily.loc[daily.model == config["primary_nonlinear"]].set_index("date")
    axes[1, 0].plot(part.index, part.coverage, color="#07a18f")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_title("Label coverage: missing and right-boundary rows retained")
    axes[1, 1].fill_between(part.index, part.kendall_tau_a_lower, part.kendall_tau_a_upper, alpha=0.25)
    axes[1, 1].plot(part.index, part.observed_kendall_tau_a, color="#075a85", lw=0.7, label="Observed only")
    axes[1, 1].axhline(0, color="black", lw=0.6)
    axes[1, 1].set_title("Full-cohort Kendall completion bounds, not Spearman CI")
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)
    rows = []
    for name, stats in summary["models"].items():
        ci = stats["ic_interval"]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{stats['mean_observed_ic']:.5f}</td>"
            f"<td>[{ci['lower_95']:.5f}, {ci['upper_95']:.5f}]</td>"
            f"<td>{stats['years']['2009']['mean_observed_ic']:.5f}</td>"
            f"<td>{stats['years']['2010']['mean_observed_ic']:.5f}</td></tr>"
        )
    interval = summary["primary_paired_ic_difference"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Wide ML V1 — development evidence</title>
<style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:38px auto;padding:0 22px;color:#143244}}
aside{{background:#fff2cd;padding:18px;border-left:5px solid #d58b00}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}img{{width:100%}}small{{color:#536a75}}</style>
<h1>Wide-security ML prediction: development evidence</h1><p>2009–2010 walk-forward development · 2011 reserved · No portfolio PnL</p>
<aside><b>Qualified scope:</b> past-price/liquidity-selected US equity securities, not a verified historical
common-stock universe. Labels are a Zipline-adjusted price return proxy, not qualified terminal wealth.
Observed-case IC is conditional on available labels. No investable-alpha or executable-Sharpe claim is made.</aside>
<h2>Fixed primary comparison</h2><p>HGB7 − Ridge100 daily IC: <b>{interval["estimate"]:.5f}</b>,
95% calendar-block interval [{interval["lower_95"]:.5f}, {interval["upper_95"]:.5f}].
This compares fixed specifications, not the best-performing candidate from each family.</p>
<table><tr><th>Model/control</th><th>Mean IC</th><th>95% block interval</th><th>2009 IC</th><th>2010 IC</th></tr>{"".join(rows)}</table>
<p>Development-ranked candidate: <b>{html.escape(summary["development_selected_candidate"])}</b>.
This selection is not independent confirmation and does not release 2011.</p><img src="diagnostics.png" alt="Forecast, coverage and missing-outcome diagnostics">
<h2>Remaining data and inference limitations</h2><ul>{"".join("<li>" + html.escape(x) + "</li>" for x in summary["limitations"])}</ul>
<small>12 fixed features; four candidates; three training-label permutation controls; 14 fits.
Raw data, row-level forecasts and fitted models remain inside the local research platform.</small></html>"""
    with (output / "report.html").open("x") as handle:
        handle.write(page)


def run_study(config_path, data_dir, output_dir, registry_path):
    config = validate_config(json.loads(Path(config_path).read_text()))
    enforce_trial_budget(registry_path, config["experiment_id"], config["max_total_model_fits"])
    arrays, dates, cache_manifest, data_hashes = load_cache(data_dir, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES}
    source_digest = canonical_digest(sources)
    config_digest = canonical_digest(config)
    run_id = "wide_" + uuid.uuid4().hex
    manifest = {
        "run_id": run_id,
        "config": config,
        "config_sha256": config_digest,
        "source_sha256": sources,
        "data_sha256": data_hashes,
        "data_digest": canonical_digest(data_hashes),
        "data_cache_path": str(Path(data_dir)),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "model_fit_budget": 14,
        "reserved_performance_enabled": False,
        "prediction_scope": "All decision-eligible test rows, including missing future labels",
        "data_qualification": "provisional; see registered scope limitations",
    }
    save_json(output / "run_started.json", manifest)
    save_json(output / "config.json", config)
    (output / "source_snapshot").mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(Path(__file__).parent / name, output / "source_snapshot" / name)
    start_run(
        registry_path,
        config["experiment_id"],
        run_id,
        config_sha256=config_digest,
        data_sha256=canonical_digest(data_hashes),
        source_sha256=source_digest,
        output_dir=str(output),
    )
    started = time.monotonic()
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def timeout_handler(*_):
        raise TimeoutError("Registered wall-clock budget exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config["max_runtime_seconds"])
    active_fit = None
    try:
        row_dates = dates.take(arrays["date_index"])
        test_mask = np.isin(row_dates.year, config["prediction_years"])
        test_indices = np.flatnonzero(test_mask)
        test_dates = row_dates[test_mask]
        calendar = dates[np.isin(dates.year, config["prediction_years"])]
        names = (
            config["models"] + [f"null_hgb_{s}" for s in config["permutation_seeds"]] + config["baselines"]
        )
        predictions = np.full((len(test_indices), len(names)), np.nan, dtype=np.float32)
        x = arrays["X"]
        for baseline, feature in (("momentum21", "mom21"), ("reversal5", "rev5")):
            predictions[:, names.index(baseline)] = x[test_indices, FEATURE_NAMES.index(feature)]
        predictions[:, names.index("composite_mom63_rev5")] = (
            x[test_indices, FEATURE_NAMES.index("mom63")] + x[test_indices, FEATURE_NAMES.index("rev5")]
        ) / 2
        trials = []
        (output / "models").mkdir()
        for year in config["prediction_years"]:
            train_idx, target, weights, audit = train_rows_for_year(
                arrays["date_index"],
                dates,
                arrays["y"],
                arrays["status"],
                arrays["asset_index"],
                arrays["asset_ids"],
                year,
                config,
            )
            if len(train_idx) < 10000 or len(np.unique(arrays["date_index"][train_idx])) < 100:
                raise ValueError("Insufficient matured training observations for the registered study")
            save_json(output / f"training_split_{year}.json", audit)
            fold_positions = np.flatnonzero(test_dates.year == year)
            predict_rows = test_indices[fold_positions]
            train_x = np.asarray(x[train_idx], dtype=np.float32)
            pred_x = np.asarray(x[predict_rows], dtype=np.float32)
            for name in names[:7]:
                active_fit = {
                    "year": year,
                    "model": name,
                    "train_rows": len(train_idx),
                    "forecast_rows": len(predict_rows),
                }
                append_fit_event(output / "fit_lifecycle.jsonl", {"event": "fit_started", **active_fit})
                model_name = "hgb_7" if name.startswith("null_hgb_") else name
                model = make_wide_model(model_name, config)
                fit_target = target
                if name.startswith("null_hgb_"):
                    seed = int(name.rsplit("_", 1)[-1]) + year
                    fit_target = permute_training_targets(target, arrays["date_index"][train_idx], seed)
                begin = time.monotonic()
                print(
                    f"Fitting {year} {name}: {len(train_idx):,} train / {len(predict_rows):,} forecast rows",
                    flush=True,
                )
                with threadpool_limits(limits=config["threads"]):
                    model.fit(train_x, fit_target, sample_weight=weights)
                    append_fit_event(output / "fit_lifecycle.jsonl", {"event": "fit_completed", **active_fit})
                    score = model.predict(pred_x)
                if not np.isfinite(score).all():
                    raise ValueError("Nonfinite forecasts; cohort must not be silently reduced")
                predictions[fold_positions, names.index(name)] = score
                joblib.dump(model, output / "models" / f"{year}_{name}.joblib", compress=3)
                trials.append(
                    {
                        "year": year,
                        "model": name,
                        "rows": len(train_idx),
                        "forecast_rows": len(predict_rows),
                        "seconds": time.monotonic() - begin,
                        "label_permutation": name.startswith("null_hgb_"),
                    }
                )
                append_fit_event(
                    output / "fit_lifecycle.jsonl", {"event": "forecast_completed", **active_fit}
                )
                active_fit = None
                del model
            del train_x, pred_x
        if len(trials) != config["max_total_model_fits"] or not np.isfinite(predictions).all():
            raise RuntimeError("Fit budget/cohort completeness mismatch")
        np.savez_compressed(
            output / "private_predictions.npz",
            predictions=predictions,
            date_index=arrays["date_index"][test_indices],
            asset_index=arrays["asset_index"][test_indices],
            model_names=np.asarray(names),
        )
        print("Evaluating calendar-aligned forecasts and missing-outcome bounds...", flush=True)
        daily, deciles = evaluate_predictions(
            predictions,
            test_dates,
            np.asarray(arrays["y"][test_indices]),
            np.asarray(arrays["status"][test_indices]),
            calendar,
            names,
        )
        summary = summarize_results(daily, deciles, calendar, config)
        summary["runtime_seconds"] = time.monotonic() - started
        summary["completed_model_fits"] = len(trials)
        summary["prepared_dimensions"] = cache_manifest["dimensions"]
        summary["prepared_label_status_counts"] = cache_manifest["label_coverage"]["selected_status_counts"]
        summary["test_label_status_counts"] = {
            name: int((arrays["status"][test_indices] == code).sum())
            for code, name in enumerate(LABEL_STATUS_NAMES)
        }
        save_json(output / "summary.json", summary)
        save_json(output / "trial_ledger.json", trials)
        daily.to_csv(output / "daily_forecast_metrics.csv", index=False)
        deciles.to_csv(output / "daily_score_decile_coverage.csv", index=False)
        write_report(output, summary, daily, config)
        if sources != {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES}:
            raise RuntimeError("Source changed during fitting; result cannot be promoted")
        current = {name: file_digest(Path(data_dir) / name) for name in data_hashes}
        if current != data_hashes:
            raise RuntimeError("Prepared data changed during fitting")
        inventory = {
            p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        }
        save_json(output / "artifact_hashes.json", inventory)
        complete_run(
            registry_path, config["experiment_id"], run_id, artifacts_sha256=canonical_digest(inventory)
        )
        return summary
    except BaseException as exc:
        if active_fit is not None:
            append_fit_event(
                output / "fit_lifecycle.jsonl",
                {"event": "fit_or_forecast_failed", "error_type": type(exc).__name__, **active_fit},
            )
        save_json(output / "FAILURE.json", {"type": type(exc).__name__, "error": str(exc)})
        fail_run(registry_path, config["experiment_id"], run_id, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("/codeload/equity-alpha-wide/registry.jsonl"))
    args = parser.parse_args(argv)
    root = Path("/codeload/equity-alpha-wide")
    if not Path("/var/lib/quantrocket").is_dir() or not args.data.resolve().is_relative_to(root):
        parser.error("Real data and row-level research must remain inside the local QuantRocket platform")
    if args.output.parent != root or not args.output.name.startswith("run-"):
        parser.error("Output must be a NEW platform run-... directory")
    summary = run_study(args.config, args.data, args.output, args.registry)
    print(
        json.dumps(
            finite_json(
                {
                    "output": str(args.output),
                    "primary_comparison": summary["primary_paired_ic_difference"],
                    "reserved_performance_computed": False,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
