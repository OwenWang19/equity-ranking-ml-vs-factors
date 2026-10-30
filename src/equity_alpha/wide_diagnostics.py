"""Frozen-forecast incremental-information diagnostics, not another model search.

This post-hoc analysis never calls fit(), loads models, constructs portfolios, or
reads 2011 observations. Same-date outcome projection is descriptive evaluation,
not a tradable residual forecast. Only aggregate tables leave the platform.
"""

from __future__ import annotations

import argparse
import html
import json
import platform
import shutil
import signal
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .incremental_stats import cross_sectional_groups, partial_rank_summary
from .integrity import canonical_digest, file_digest
from .registry import complete_run, fail_run, read_registry, start_run
from .wide_data import FEATURE_NAMES, LABEL_STATUS_NAMES
from .wide_experiment import ARRAY_FILES, finite_json, save_json
from .wide_stats import paired_calendar_bootstrap
from .wide_verify import verify_wide_run

EXPECTED_RUN = "wide_e0f795afd46f4f1e95a950ab8181ca7c"
EXPECTED_INVENTORY = "cccfaf6a749510dee368538436fc31425f6c0275c3af7932e7f29b4c5aee0032"
EXPECTED_PREDICTIONS = "d878d886328a635a93134c3fe5a2be4de478e85e4610f7447107e1a76c9f3520"
PLATFORM_ROOT = Path("/codeload/equity-alpha-wide")
MODEL_NAMES = (
    "ridge_10",
    "ridge_100",
    "hgb_7",
    "hgb_15",
    "null_hgb_191",
    "null_hgb_727",
    "null_hgb_2027",
    "momentum21",
    "reversal5",
    "composite_mom63_rev5",
)
SCOPES = ("all", "low_liquidity", "high_liquidity", "low_volatility", "high_volatility")
PARTIAL_MODELS = ("hgb_7", "null_hgb_191", "null_hgb_727", "null_hgb_2027")
SOURCE_FILES = (
    "wide_diagnostics.py",
    "incremental_stats.py",
    "wide_stats.py",
    "wide_verify.py",
    "wide_experiment.py",
    "wide_data.py",
    "registry.py",
    "integrity.py",
)


def validate_protocol(config):
    fixed = {
        "experiment_id": "wide_incremental_diagnostics_v1",
        "source_run_id": EXPECTED_RUN,
        "source_artifact_inventory_digest": EXPECTED_INVENTORY,
        "source_predictions_sha256": EXPECTED_PREDICTIONS,
        "prediction_years": [2009, 2010],
        "reserved_start": "2011-01-01",
        "reserved_performance_enabled": False,
        "new_model_fits": 0,
        "model": "hgb_7",
        "comparators": ["reversal5", "momentum21", "ridge_100"],
        "partial_rank_controls": ["reversal5", "momentum21"],
        "partial_null_controls": list(PARTIAL_MODELS[1:]),
        "scopes": list(SCOPES),
        "minimum_partial_observations": 20,
        "bootstrap_primary_block": 20,
        "bootstrap_sensitivity_blocks": [10, 20, 40],
        "bootstrap_samples": 1000,
        "seed": 1729,
        "max_analysis_runs": 1,
        "max_runtime_seconds": 600,
        "multiplicity_adjusted": False,
        "promotion_authorized": False,
    }
    for key, expected in fixed.items():
        if key not in config or type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"Diagnostic protocol differs from the fixed design: {key}")
    if not config.get("analysis_status", "").startswith("Post-hoc exploratory"):
        raise ValueError("Previously observed development results must be disclosed")
    return config


def validate_frozen_predictions(predictions, arrays, dates):
    """Exact, order-preserving binding: never inner-join or shrink the cohort."""
    if set(predictions) != {"predictions", "date_index", "asset_index", "model_names"}:
        raise ValueError("Frozen prediction schema differs")
    dates = pd.DatetimeIndex(dates)
    if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Invalid cache calendar")
    if dates.max() >= pd.Timestamp("2011-01-01"):
        raise ValueError("Reserved sessions cannot enter diagnostic arrays")
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("Cache sessions must be timezone-naive midnight labels")
    for key in ("date_index", "asset_index"):
        vector = np.asarray(arrays[key])
        if vector.ndim != 1 or vector.dtype.kind not in "iu" or (vector < 0).any():
            raise ValueError(f"Invalid cache {key}")
    n = len(arrays["date_index"])
    if len(arrays["asset_index"]) != n or not n or arrays["date_index"].max() >= len(dates):
        raise ValueError("Invalid cache row references")
    ids = np.asarray(arrays["asset_ids"])
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Cache security dictionary must be nonempty and unique")
    if arrays["asset_index"].max() >= len(ids):
        raise ValueError("Cache security references exceed the dictionary")
    if arrays["X"].shape != (n, len(FEATURE_NAMES)):
        raise ValueError("Cache features do not match the frozen row schema")
    if arrays["y"].shape != (n,) or arrays["status"].shape != (n,):
        raise ValueError("Cache outcomes/statuses must be parallel row vectors")
    row_dates = dates.take(arrays["date_index"])
    rows = np.flatnonzero(np.isin(row_dates.year, (2009, 2010)))
    for key in ("date_index", "asset_index"):
        vector = np.asarray(predictions[key])
        if vector.dtype.kind not in "iu" or vector.ndim != 1:
            raise ValueError(f"Invalid frozen {key}")
        if not np.array_equal(vector, arrays[key][rows]):
            raise ValueError(f"Frozen {key} does not exactly match cache cohort/order")
    names = np.asarray(predictions["model_names"])
    if names.ndim != 1 or tuple(names.tolist()) != MODEL_NAMES:
        raise ValueError("Frozen model names/order differ from the original run")
    scores = np.asarray(predictions["predictions"])
    if scores.shape != (len(rows), len(MODEL_NAMES)) or not np.isfinite(scores).all():
        raise ValueError("Invalid or incomplete frozen score matrix")
    pairs = np.column_stack((predictions["date_index"], predictions["asset_index"]))
    if len(np.unique(pairs, axis=0)) != len(rows):
        raise ValueError("Duplicate forecast decision/security pairs")
    if not len(rows) or np.any(np.diff(predictions["date_index"].astype(np.int64)) < 0):
        raise ValueError("Frozen rows must be nonempty and date-major")
    return rows


def _rank_correlation(a, b, *, constant_score_zero=False):
    if len(a) < 3 or np.unique(b).size < 2:
        return float("nan")
    if np.unique(a).size < 2:
        return 0.0 if constant_score_zero else float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def diagnostic_daily(scores, labels, statuses, row_dates, calendar, features):
    """All outputs aggregate at date/scope level; no per-security residuals."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    statuses = np.asarray(statuses)
    features = np.asarray(features)
    row_dates = pd.DatetimeIndex(row_dates).as_unit("ns")
    calendar = pd.DatetimeIndex(calendar).as_unit("ns")
    n = len(labels)
    if labels.ndim != 1:
        raise ValueError("Diagnostic labels must be a one-dimensional vector")
    if scores.shape != (n, len(MODEL_NAMES)) or features.shape != (n, len(FEATURE_NAMES)):
        raise ValueError("Invalid diagnostic score/feature dimensions")
    if statuses.shape != (n,) or len(row_dates) != n:
        raise ValueError("Invalid diagnostic row-vector dimensions")
    if not np.isfinite(scores).all() or not np.isfinite(features).all() or np.isinf(labels).any():
        raise ValueError("Nonfinite diagnostic inputs")
    if statuses.dtype.kind not in "iu" or (statuses < 0).any() or (statuses >= len(LABEL_STATUS_NAMES)).any():
        raise ValueError("Unknown diagnostic label statuses")
    if not np.array_equal(np.isfinite(labels), statuses == 0):
        raise ValueError("Diagnostic labels disagree with status")
    if (
        calendar.empty
        or calendar.hasnans
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or calendar.max() >= pd.Timestamp("2011-01-01")
        or not row_dates.isin(calendar).all()
        or not row_dates.is_monotonic_increasing
        or not set(calendar.year).issubset({2009, 2010})
    ):
        raise ValueError("Diagnostic calendar is outside frozen development scope")
    positions = {name: MODEL_NAMES.index(name) for name in MODEL_NAMES}
    control_positions = [positions["reversal5"], positions["momentum21"]]
    rows = []
    # searchsorted avoids rescanning the full forecast matrix for every session.
    starts = row_dates.searchsorted(calendar, side="left")
    ends = row_dates.searchsorted(calendar, side="right")
    for date, begin, end in zip(calendar, starts, ends, strict=True):
        day_scores, day_labels, day_status = scores[begin:end], labels[begin:end], statuses[begin:end]
        day_features = features[begin:end]
        groups = cross_sectional_groups(
            day_features[:, FEATURE_NAMES.index("log_adv20")],
            day_features[:, FEATURE_NAMES.index("vol21")],
        )
        for scope, mask in groups.items():
            group_scores, y, status = day_scores[mask], day_labels[mask], day_status[mask]
            observed = np.isfinite(y)
            boundary = int((status == 4).sum())
            row = {
                "date": date,
                "scope": scope,
                "predicted_rows": len(y),
                "observed_rows": int(observed.sum()),
                "missing_rows": int((~observed).sum()),
                "right_boundary_rows": boundary,
                "nonboundary_missing_rows": int((~observed).sum()) - boundary,
                "coverage": float(observed.mean()) if len(y) else float("nan"),
            }
            for name in ("hgb_7", "reversal5", "momentum21", "ridge_100"):
                row[f"{name}_ic"] = _rank_correlation(
                    group_scores[observed, positions[name]], y[observed], constant_score_zero=True
                )
            for name in PARTIAL_MODELS:
                partial = partial_rank_summary(
                    group_scores[:, positions[name]],
                    y,
                    group_scores[:, control_positions],
                    min_observed=20,
                )
                row[f"{name}_partial_ic"] = partial["partial_rank_correlation"]
                row[f"{name}_partial_status"] = partial["status"]
                if name == "hgb_7":
                    row["partial_design_rank"] = partial["design_rank"]
                    row["partial_score_residual_norm_ratio"] = partial["score_residual_norm_ratio"]
                    row["partial_outcome_residual_norm_ratio"] = partial["outcome_residual_norm_ratio"]
            for name in ("reversal5", "momentum21"):
                row[f"score_rank_corr_{name}"] = _rank_correlation(
                    group_scores[:, positions["hgb_7"]], group_scores[:, positions[name]]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_diagnostics(daily, calendar, config):
    validate_protocol(config)
    # Pandas 2.x compares datetime resolutions in Index.equals. Normalize units
    # only: no dates are added, dropped, floored, filtered or joined.
    calendar = pd.DatetimeIndex(calendar).as_unit("ns")
    result = {}
    for scope in SCOPES:
        part = daily.loc[daily.scope == scope].set_index("date")
        if not part.index.equals(calendar):
            raise ValueError("Each scope must retain the exact full calendar")
        contrasts = {
            f"hgb7_minus_{name}": (part.hgb_7_ic, part[f"{name}_ic"]) for name in config["comparators"]
        }
        contrasts["hgb7_partial_rank_ic"] = (part.hgb_7_partial_ic, pd.Series(0.0, index=calendar))
        metrics = {}
        for name, (a, b) in contrasts.items():
            sensitivity = {
                str(block): paired_calendar_bootstrap(
                    a, b, seed=config["seed"], samples=config["bootstrap_samples"], block=block
                )
                for block in config["bootstrap_sensitivity_blocks"]
            }
            metrics[name] = {
                "primary_interval": sensitivity[str(config["bootstrap_primary_block"])],
                "block_sensitivity": sensitivity,
                "years": {
                    str(year): float((a - b).loc[calendar.year == year].mean())
                    for year in config["prediction_years"]
                },
            }
        nulls = {}
        if scope == "all":
            for name in config["partial_null_controls"]:
                values = part[f"{name}_partial_ic"]
                nulls[name] = {
                    "partial_interval": paired_calendar_bootstrap(
                        values,
                        pd.Series(0.0, index=calendar),
                        seed=config["seed"],
                        samples=config["bootstrap_samples"],
                        block=config["bootstrap_primary_block"],
                    ),
                    "undefined_status_counts": part[f"{name}_partial_status"].value_counts().to_dict(),
                }
        result[scope] = {
            "predicted_rows": int(part.predicted_rows.sum()),
            "observed_rows": int(part.observed_rows.sum()),
            "right_boundary_rows": int(part.right_boundary_rows.sum()),
            "nonboundary_missing_rows": int(part.nonboundary_missing_rows.sum()),
            "partial_status_counts": part.hgb_7_partial_status.value_counts().to_dict(),
            "mean_score_rank_correlation": {
                name: float(part[f"score_rank_corr_{name}"].mean()) for name in ("reversal5", "momentum21")
            },
            "metrics": metrics,
            "overall_partial_null_controls": nulls,
        }
    return {
        "experiment_id": config["experiment_id"],
        "analysis_status": config["analysis_status"],
        "source_run_id": EXPECTED_RUN,
        "scopes": result,
        "new_model_fits": 0,
        "reserved_performance_computed": False,
        "promotion_authorized": False,
        "net_pnl_computed": False,
        "inference_limits": [
            "Post-hoc development diagnostics; all intervals are pointwise and not selection/multiplicity adjusted",
            "Partial rank correlation is descriptive linear projection on the common observed cohort, not a tradable forecast",
            "Cross-sectional groups are defined before labels and are not aggregate market regimes",
            "Missing-outcome Kendall bounds from V1 do not transfer to partial correlations",
            "All V1 data-vintage, historical classification, corporate-action and terminal-wealth limitations remain",
        ],
    }


def write_diagnostic_report(output, summary, daily):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = summary["scopes"]["all"]["metrics"]
    primary = main["hgb7_minus_reversal5"]["primary_interval"]
    partial = main["hgb7_partial_rank_ic"]["primary_interval"]
    # The headline is a measurement, not an automatic significance/promotion gate.
    rows = []
    for scope in SCOPES:
        stats = summary["scopes"][scope]
        difference = stats["metrics"]["hgb7_minus_reversal5"]["primary_interval"]
        incremental = stats["metrics"]["hgb7_partial_rank_ic"]["primary_interval"]
        rows.append(
            f"<tr><td>{html.escape(scope)}</td><td>{stats['predicted_rows']:,}</td>"
            f"<td>{difference['estimate']:.5f} [{difference['lower_95']:.5f}, {difference['upper_95']:.5f}]</td>"
            f"<td>{incremental['estimate']:.5f} [{incremental['lower_95']:.5f}, {incremental['upper_95']:.5f}]</td></tr>"
        )
    null_rows = []
    for name, stats in summary["scopes"]["all"]["overall_partial_null_controls"].items():
        ci = stats["partial_interval"]
        null_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{ci['estimate']:.5f}</td>"
            f"<td>[{ci['lower_95']:.5f}, {ci['upper_95']:.5f}]</td></tr>"
        )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    all_days = daily.loc[daily.scope == "all"].set_index("date")
    gap = all_days.hgb_7_ic - all_days.reversal5_ic
    axes[0].plot(gap.index, gap.rolling(63, min_periods=40).mean(), color="#126d8a")
    axes[0].set_title("HGB7 minus reversal5: 63-session mean daily IC difference (not PnL)")
    axes[1].plot(
        all_days.index,
        all_days.hgb_7_partial_ic.rolling(63, min_periods=40).mean(),
        color="#a45324",
        label="HGB7 partial rank correlation",
    )
    for name in PARTIAL_MODELS[1:]:
        axes[1].plot(
            all_days.index,
            all_days[f"{name}_partial_ic"].rolling(63, min_periods=40).mean(),
            alpha=0.5,
            lw=0.8,
            label=name,
        )
    axes[1].set_title("Descriptive rank projection on reversal5 + momentum21; observed outcomes only")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "incremental_diagnostics.png", dpi=150)
    plt.close(fig)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Frozen-forecast incremental diagnostics</title>
<style>body{{font:16px/1.65 system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#183344}}
aside{{background:#fff2d9;border-left:5px solid #bd791b;padding:18px}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #dae2e7}}img{{width:100%;height:auto}}
.scroll{{overflow:auto}}a{{color:#086f96}}</style></head><body>
<h1>Does the frozen ML forecast add information beyond simple factors?</h1>
<aside><b>Post-hoc exploratory analysis of 2009–2010 development.</b> Original V1 results were already observed.
No new model was fitted; 2011 performance remains disabled. Pointwise intervals do not correct selection or multiplicity.</aside>
<h2>Overall measurements</h2><p>Paired HGB7 minus reversal5 daily IC: <b>{primary["estimate"]:.5f}</b>,
95% 20-session calendar-block interval [{primary["lower_95"]:.5f}, {primary["upper_95"]:.5f}].</p>
<p>HGB7 partial rank correlation after linear projection of both score and outcome ranks on reversal5 and momentum21:
<b>{partial["estimate"]:.5f}</b>, interval [{partial["lower_95"]:.5f}, {partial["upper_95"]:.5f}].
This is an observed-case descriptive statistic, not a tradable residual signal or general conditional-independence test.</p>
<h2>All declared cross-sectional conditions</h2><div class="scroll"><table>
<tr><th>Condition</th><th>Forecast rows</th><th>HGB7 − reversal5 IC [95% interval]</th><th>Partial rank correlation [95% interval]</th></tr>
{"".join(rows)}</table></div><p>Liquidity and volatility groups use the full same-date forecast cohort before checking outcomes.
They are relative cross-sectional conditions, not aggregate market regimes. Both years and all fixed
10/20/40-session block sensitivities are reported in <a href="summary.json">summary.json</a>; none is selected as the headline result.</p>
<h2>Frozen shuffled-label control diagnostics</h2><table><tr><th>Original training-label control</th><th>Mean partial rank correlation</th><th>95% interval</th></tr>
{"".join(null_rows)}</table><p>These three controls are diagnostics, not a high-powered randomization test.</p>
<img src="incremental_diagnostics.png" alt="Rolling paired IC difference and partial rank correlations, with all fixed null controls">
<h2>Why this is not deployment evidence</h2><ul>{"".join("<li>" + html.escape(s) + "</li>" for s in summary["inference_limits"])}</ul>
<p>No combination weights, signs, features, subgroups or hyperparameters were searched. Outcome projection uses same-date
observed returns for evaluation only; no coefficients or per-security residuals are exported. Missing outcomes remain in
forecast denominators, and undefined partial correlations remain undefined. There is no PnL or trading-promotion gate.</p>
<p><a href="analysis_started.json">Input/protocol/runtime bindings</a> · <a href="config.json">Analysis protocol</a> ·
<a href="daily_incremental_metrics.csv">All aggregate daily diagnostics</a></p></body></html>"""
    with (output / "report.html").open("x") as handle:
        handle.write(page)


def authorize_analysis_attempt(registry_path, config, recovery_run_id=None):
    """One analysis, with one explicit pre-summary compatibility-error retry.

    The diagnosed failed directory and registry events remain untouched. This
    guard is local/single-process, not a distributed experiment lock.
    """
    events = [r for r in read_registry(registry_path) if r["experiment_id"] == config["experiment_id"]]
    if not events:
        if recovery_run_id is not None:
            raise RuntimeError("Cannot recover a nonexistent diagnostic attempt")
        return None
    starts = [r for r in events if r["event"] == "run_started"]
    if len(starts) != 1 or recovery_run_id is None or starts[0]["run_id"] != recovery_run_id:
        raise RuntimeError(
            "Diagnostic analysis already registered; no automatic repeat or favorable-interval search"
        )
    if len(events) != 2 or events[-1]["event"] != "run_failed" or events[-1]["run_id"] != recovery_run_id:
        raise RuntimeError("Only the single diagnosed failed engineering attempt can be recovered")
    directory = Path(starts[0]["payload"]["output_dir"])
    if not directory.resolve().is_relative_to(PLATFORM_ROOT):
        raise RuntimeError("Failed attempt must remain inside the platform root")
    failure = json.loads((directory / "FAILURE.json").read_text())
    if failure != {"type": "ValueError", "error": "Each scope must retain the exact full calendar"}:
        raise RuntimeError("Failure differs from the diagnosed datetime-unit compatibility issue")
    if any(
        (directory / name).exists()
        for name in ("summary.json", "daily_incremental_metrics.csv", "artifact_hashes.json", "report.html")
    ):
        raise RuntimeError("A statistical result exists; no engineering retry is authorized")
    old_config = json.loads((directory / "config.json").read_text())
    if old_config != config or starts[0]["payload"]["config_sha256"] != canonical_digest(config):
        raise RuntimeError("An engineering retry cannot change the scientific protocol")
    return {
        "failed_run_id": recovery_run_id,
        "failed_directory": str(directory),
        "failure_sha256": file_digest(directory / "FAILURE.json"),
        "reason": "Normalize identical midnight session labels from seconds to nanoseconds; no calendar filtering or statistical-design changes",
        "failed_before_statistical_summary": True,
        "failed_attempt_retained": True,
    }


def run_diagnostics(config_path, run_path, cache_path, output_path, registry_path, *, recovery_run_id=None):
    config = validate_protocol(json.loads(Path(config_path).read_text()))
    recovery = authorize_analysis_attempt(registry_path, config, recovery_run_id)
    verification = verify_wide_run(run_path, cache_path=cache_path, registry_path=registry_path)
    if (
        verification["run_id"] != EXPECTED_RUN
        or verification["artifact_inventory_digest"] != EXPECTED_INVENTORY
    ):
        raise ValueError("Diagnostics require the exact frozen V1 run")
    prediction_path = Path(run_path) / "private_predictions.npz"
    if file_digest(prediction_path) != EXPECTED_PREDICTIONS:
        raise ValueError("Frozen prediction hash differs")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES}
    started = time.monotonic()
    analysis_id = "diagnostic_" + uuid.uuid4().hex
    binding = {
        "run_id": analysis_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": canonical_digest(config),
        "source_sha256": sources,
        "frozen_input_verification": verification,
        "prediction_sha256": EXPECTED_PREDICTIONS,
        "new_model_fits": 0,
        "engineering_recovery": recovery,
        "libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    save_json(output / "analysis_started.json", binding)
    save_json(output / "config.json", config)
    (output / "source_snapshot").mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(Path(__file__).parent / name, output / "source_snapshot" / name)
    start_run(
        registry_path,
        config["experiment_id"],
        analysis_id,
        config_sha256=canonical_digest(config),
        data_sha256=verification["data_digest"],
        source_sha256=canonical_digest(sources),
        output_dir=str(output),
    )
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def timeout_handler(*_):
        raise TimeoutError("Diagnostic wall-clock budget exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config["max_runtime_seconds"])
    try:
        arrays = {
            name: np.load(Path(cache_path) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ARRAY_FILES
        }
        dates = pd.DatetimeIndex(arrays["dates"])
        with np.load(prediction_path, allow_pickle=False) as predictions:
            rows = validate_frozen_predictions(predictions, arrays, dates)
            scores = predictions["predictions"]
        calendar = dates[np.isin(dates.year, config["prediction_years"])]
        with threadpool_limits(limits=2):
            daily = diagnostic_daily(
                scores,
                np.asarray(arrays["y"][rows]),
                np.asarray(arrays["status"][rows]),
                dates.take(arrays["date_index"][rows]),
                calendar,
                np.asarray(arrays["X"][rows]),
            )
            summary = summarize_diagnostics(daily, calendar, config)
        summary["runtime_seconds"] = time.monotonic() - started
        daily.to_csv(output / "daily_incremental_metrics.csv", index=False)
        save_json(output / "summary.json", summary)
        write_diagnostic_report(output, summary, daily)
        if sources != {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES}:
            raise RuntimeError("Diagnostic source changed during analysis")
        after = verify_wide_run(run_path, cache_path=cache_path, registry_path=registry_path)
        if after != verification:
            raise RuntimeError("Frozen input verification changed during analysis")
        inventory = {
            p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        }
        save_json(output / "artifact_hashes.json", inventory)
        complete_run(
            registry_path, config["experiment_id"], analysis_id, artifacts_sha256=canonical_digest(inventory)
        )
        return summary
    except BaseException as exc:
        save_json(output / "FAILURE.json", {"type": type(exc).__name__, "error": str(exc)})
        fail_run(registry_path, config["experiment_id"], analysis_id, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "run", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("/codeload/equity-alpha-wide/registry.jsonl"))
    parser.add_argument("--recover-failed-run", default=None)
    args = parser.parse_args(argv)
    root = Path("/codeload/equity-alpha-wide")
    if not Path("/var/lib/quantrocket").is_dir():
        parser.error("Vendor-derived analysis must remain inside the local research platform")
    for path in (args.run, args.cache, args.output, args.registry):
        if not path.resolve().is_relative_to(root):
            parser.error("All data, outputs and registry must remain inside the platform research root")
    if args.output.parent != root or not args.output.name.startswith("diagnostics-"):
        parser.error("Output must be a NEW platform diagnostics-... directory")
    summary = run_diagnostics(
        args.config, args.run, args.cache, args.output, args.registry, recovery_run_id=args.recover_failed_run
    )
    print(
        json.dumps(
            finite_json(
                {
                    "output": str(args.output),
                    "new_model_fits": 0,
                    "reserved_performance_computed": False,
                    "overall_metrics": summary["scopes"]["all"]["metrics"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
