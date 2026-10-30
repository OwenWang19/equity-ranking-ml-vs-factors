"""Reproduce a development run, audit its data, or score a frozen local model."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config
from .data import load_prices
from .integrity import canonical_digest, verify_run
from .registry import RegistryError, complete_run, fail_run, start_run

ROOT = Path(__file__).resolve().parents[2]


def _emit_json(value: dict, output: Path | None = None):
    text = json.dumps(value, indent=2, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(text + "\n")
    print(text)


def _audit_qlib(archive: Path, output: Path | None, symbol_map_path: Path | None = None):
    from .qlib_data import load_qlib_archive

    try:
        if symbol_map_path is None:
            *_, audit = load_qlib_archive(archive)
        else:
            mapping_bytes = symbol_map_path.read_bytes()
            mapping = json.loads(mapping_bytes)
            if not isinstance(mapping, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
            ):
                raise ValueError("Symbol map must be a JSON object mapping symbols to strings")
            *_, audit = load_qlib_archive(archive, symbol_map=mapping)
            audit["symbol_map_source_path"] = str(symbol_map_path.resolve())
            audit["symbol_map_source_sha256"] = hashlib.sha256(mapping_bytes).hexdigest()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        _emit_json(
            {
                "audit": None,
                "gate": {
                    "eligible_for_broad_research": False,
                    "status": "audit_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            },
            output,
        )
        raise SystemExit(2) from exc
    coverage = audit["membership_close_coverage"]
    missing_exits = audit["membership_exits_without_next_session_close"]
    missing_tickers = audit["tickers_without_close_data"]
    coverage_ok = coverage is not None and math.isfinite(coverage) and coverage >= 0.995
    reasons = []
    if not coverage_ok:
        reasons.append("Historical member close coverage is below the required 0.995 minimum")
    if missing_exits:
        reasons.append("At least one historical membership exit lacks a next-session close")
    if missing_tickers:
        reasons.append("At least one historical member has no close observations")
    if audit.get("membership_quality", {}).get("status") == "known_inaccurate":
        reasons.append("Provider membership history contradicts a documented issuer announcement")
    result = {
        "audit": audit,
        "gate": {
            "eligible_for_broad_research": not reasons,
            "status": "coverage_checks_passed" if not reasons else "disqualified",
            "minimum_membership_close_coverage": 0.995,
            "observed_membership_close_coverage": coverage,
            "membership_exits_without_next_session_close_count": len(missing_exits),
            "tickers_without_close_data_count": len(missing_tickers),
            "reasons": reasons,
            "scope": "Coverage gate only; source constraints and unverified PIT/delisting history still apply",
        },
    }
    _emit_json(result, output)
    if reasons:
        raise SystemExit(2)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Validate the CSV and print a provenance audit")
    audit.add_argument("--data", type=Path, required=True)
    verify = sub.add_parser("verify", help="Read-only verification of a complete frozen run")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--data", type=Path, help="Optionally verify original source bytes")
    qlib = sub.add_parser("audit-qlib", help="Audit a Qlib archive and enforce broad-universe coverage gates")
    qlib.add_argument("--archive", type=Path, required=True)
    qlib.add_argument("--output", type=Path, help="Save audit JSON to a new file; never overwrite")
    qlib.add_argument("--symbol-map", type=Path, help="Explicit JSON ticker-to-archive-symbol mapping")
    run = sub.add_parser("run", help="Development walk-forward study; reserved period remains unevaluated")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--config", type=Path, default=ROOT / "configs/research.json")
    run.add_argument("--output", type=Path)
    run.add_argument("--registry", type=Path, default=ROOT / "artifacts/research_registry.jsonl")
    run.add_argument(
        "--no-controls", action="store_true", help="Smoke run only: omit permutation/ablation fits"
    )
    predict = sub.add_parser("predict", help="Score one as-of date using a trusted frozen local model")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--asof", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        _emit_json(verify_run(args.run, args.data))
        return
    if args.command == "audit-qlib":
        _audit_qlib(args.archive, args.output, args.symbol_map)
        return
    prices, sectors, audit_data = load_prices(args.data)
    if args.command == "audit":
        print(json.dumps(audit_data, indent=2, allow_nan=False))
        return
    if args.command == "predict":
        from .inference import predict_asof

        prediction = predict_asof(prices, sectors, args.bundle, args.asof)
        print(prediction.to_csv())
        return
    from .evaluation import evaluate
    from .experiment import build_manifest, run_forecasts
    from .reporting import write_report

    config = load_config(args.config)
    command = sys.argv if argv is None else ["equity-alpha", *argv]
    manifest = build_manifest(config, audit_data, ROOT, command)
    manifest["controls_enabled"] = not args.no_controls
    run_id = "run_" + uuid.uuid4().hex
    manifest["registry_run_id"] = run_id
    output = args.output or ROOT / "artifacts" / (
        config.experiment_id + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    if args.registry.resolve().is_relative_to(output.resolve()):
        parser.error("The mutable experiment registry must be outside the frozen run directory")
    output.mkdir(parents=True, exist_ok=False)
    registered = False
    try:
        # Freeze the protocol and actual executable revision before model results.
        (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        (output / "run_started.json").write_text(json.dumps(manifest, indent=2))
        for relative_path in manifest["source_sha256"]:
            destination = output / "source_snapshot" / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative_path, destination)
        start_run(
            args.registry,
            config.experiment_id,
            run_id,
            config_sha256=config.digest(),
            data_sha256=audit_data["source_sha256"],
            source_sha256=canonical_digest(manifest["source_sha256"]),
            output_dir=str(output.resolve()),
        )
        registered = True
        forecasts, trials, splits, importance = run_forecasts(
            prices,
            sectors,
            config,
            controls=not args.no_controls,
            model_dir=output / "models",
        )
        forecasts.to_csv(output / "forecasts.csv")
        trials.to_csv(output / "trial_ledger.csv", index=False)
        splits.to_csv(output / "split_audit.csv", index=False)
        importance.to_csv(output / "validation_feature_importance.csv", index=False)
        dev_prices = prices.loc[prices.index < config.holdout_start]
        print("Evaluating delayed portfolios, uncertainty, and predeclared controls...", flush=True)
        result = evaluate(forecasts, dev_prices, sectors, config)
        current = build_manifest(config, audit_data, ROOT, command)
        if current["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError(
                "Source changed during run; reject this result and rerun from a stable snapshot"
            )
        verdict = write_report(output, result, manifest, trials, splits, importance)
        hashes = {
            p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2))
        verified = verify_run(output, args.data)
        complete_run(
            args.registry,
            config.experiment_id,
            run_id,
            artifacts_sha256=verified["artifact_inventory_sha256"],
        )
    except BaseException as exc:
        failure = {
            "registry_run_id": run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if registered:
            try:
                fail_run(args.registry, config.experiment_id, run_id, error=f"{type(exc).__name__}: {exc}")
            except (RegistryError, OSError) as registry_exc:
                failure["registry_failure_record_error"] = f"{type(registry_exc).__name__}: {registry_exc}"
                print(f"Failure could not be appended to registry: {registry_exc}", file=sys.stderr)
        with (output / "FAILURE.json").open("x") as handle:
            json.dump(failure, handle, indent=2, allow_nan=False)
        print(f"Failed attempt retained at {output}; {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(result["summary"][["mean_daily_rank_ic", "net_sharpe", "max_drawdown"]].round(4).to_string())
    print(verdict)
    print(f"Report: {output / 'report.html'}")
    print(f"Reserved holdout: {config.holdout_start} onward; not evaluated")


if __name__ == "__main__":
    main()
