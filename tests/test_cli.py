"""Command dispatch, data qualification, and transactional run completion."""

import hashlib
import json

import pandas as pd
import pytest

from equity_alpha import cli
from equity_alpha.config import ResearchConfig
from equity_alpha.integrity import IntegrityError, canonical_digest, verify_run
from equity_alpha.registry import read_registry


def _write_hashes(run):
    hashes = {
        p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.rglob("*")
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    (run / "artifact_hashes.json").write_text(json.dumps(hashes))


@pytest.fixture
def tiny_frozen_run(tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    source = {}
    for name in (
        "README.md",
        "pyproject.toml",
        "src/model.py",
        "tests/test_model.py",
        "configs/research.json",
        "docs/RESEARCH_PROTOCOL.md",
    ):
        path = root / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = tmp_path / "source.csv"
    data.write_text("untouched source bytes")
    config = {"experiment_id": "tiny"}
    manifest = {
        "config": config,
        "config_sha256": canonical_digest(config),
        "data": {"source_sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        "source_sha256": source,
    }
    for name in ("manifest.json", "run_started.json"):
        (root / name).write_text(json.dumps(manifest))
    (root / "config.json").write_text(json.dumps(config))
    _write_hashes(root)
    return root, data


def test_verify_dispatch_never_invokes_price_loader(tiny_frozen_run, monkeypatch, capsys):
    root, data = tiny_frozen_run

    def forbidden(_):
        raise AssertionError("verify must not load market prices")

    monkeypatch.setattr(cli, "load_prices", forbidden)
    cli.main(["verify", "--run", str(root)])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["data_verified"] is False
    cli.main(["verify", "--run", str(root), "--data", str(data)])
    assert json.loads(capsys.readouterr().out)["data_verified"] is True


def test_verify_tampering_is_a_hard_error(tiny_frozen_run):
    root, _ = tiny_frozen_run
    (root / "config.json").write_text("{}")
    with pytest.raises(IntegrityError, match="checksum"):
        cli.main(["verify", "--run", str(root)])


def test_invalid_command_is_rejected_by_parser_without_loading_prices(monkeypatch):
    monkeypatch.setattr(cli, "load_prices", lambda _: pytest.fail("unexpected price loader"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["unknown-command"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "coverage,exits,missing,eligible",
    [
        (0.995, [], [], True),
        (0.9949, [], [], False),
        (1.0, [{"ticker": "X"}], [], False),
        (1.0, [], ["X"], False),
        (None, [], [], False),
    ],
)
def test_qlib_audit_gates_and_saves_failed_qualification(
    tmp_path,
    monkeypatch,
    capsys,
    coverage,
    exits,
    missing,
    eligible,
):
    import equity_alpha.qlib_data as qlib

    audit = {
        "membership_close_coverage": coverage,
        "membership_exits_without_next_session_close": exits,
        "tickers_without_close_data": missing,
    }
    monkeypatch.setattr(qlib, "load_qlib_archive", lambda _: (None, None, None, None, audit))
    monkeypatch.setattr(cli, "load_prices", lambda _: pytest.fail("unexpected CSV price loader"))
    output = tmp_path / "audit.json"
    argv = ["audit-qlib", "--archive", str(tmp_path / "archive.zip"), "--output", str(output)]
    if eligible:
        cli.main(argv)
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text())
    assert printed["gate"]["eligible_for_broad_research"] is eligible
    assert printed["audit"] == audit
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main(argv)
    assert output.read_bytes() == before


def test_missing_qlib_archive_is_transparent_and_does_not_require_data(tmp_path, capsys):
    output = tmp_path / "audit.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(["audit-qlib", "--archive", str(tmp_path / "missing.zip"), "--output", str(output)])
    assert exc.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["gate"]["status"] == "audit_failed"
    assert result["gate"]["error_type"] == "FileNotFoundError"
    assert json.loads(output.read_text()) == result


def test_qlib_known_membership_error_disqualifies_even_complete_quote_coverage(monkeypatch, capsys):
    import equity_alpha.qlib_data as qlib

    audit = {
        "membership_close_coverage": 1.0,
        "membership_exits_without_next_session_close": [],
        "tickers_without_close_data": [],
        "membership_quality": {"status": "known_inaccurate"},
    }
    monkeypatch.setattr(qlib, "load_qlib_archive", lambda _: (None, None, None, None, audit))
    with pytest.raises(SystemExit) as exc:
        cli.main(["audit-qlib", "--archive", "archive.zip"])
    assert exc.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert any("issuer announcement" in reason for reason in result["gate"]["reasons"])


def test_explicit_symbol_map_is_passed_and_fingerprinted(tmp_path, monkeypatch, capsys):
    import equity_alpha.qlib_data as qlib

    mapping_path = tmp_path / "symbols.json"
    mapping_path.write_text(json.dumps({"BRK.B": "BRK-B"}))

    def loader(archive, symbol_map):
        assert symbol_map == {"BRK.B": "BRK-B"}
        return (
            None,
            None,
            None,
            None,
            {
                "membership_close_coverage": 1.0,
                "membership_exits_without_next_session_close": [],
                "tickers_without_close_data": [],
            },
        )

    monkeypatch.setattr(qlib, "load_qlib_archive", loader)
    cli.main(["audit-qlib", "--archive", "archive.zip", "--symbol-map", str(mapping_path)])
    audit = json.loads(capsys.readouterr().out)["audit"]
    assert audit["symbol_map_source_sha256"] == hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    assert audit["symbol_map_source_path"] == str(mapping_path.resolve())


@pytest.fixture
def mocked_research(tmp_path, monkeypatch):
    from equity_alpha import evaluation, experiment, reporting

    project = tmp_path / "project"
    for name in (
        "README.md",
        "pyproject.toml",
        "src/model.py",
        "tests/test_model.py",
        "configs/research.json",
        "docs/RESEARCH_PROTOCOL.md",
    ):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    config = ResearchConfig(experiment_id="test_cli")
    monkeypatch.setattr(cli, "ROOT", project)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    data = tmp_path / "data.csv"
    data.write_text("fixture input bytes")
    audit = {"source_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    prices = pd.DataFrame({"A": [100, 101], "B": [100, 99]}, index=pd.bdate_range("2020-01-01", periods=2))
    sectors = pd.Series({"A": "x", "B": "y"})
    monkeypatch.setattr(cli, "load_prices", lambda _: (prices, sectors, audit))
    frame = pd.DataFrame({"fixture": [1]})
    monkeypatch.setattr(experiment, "run_forecasts", lambda *a, **k: (frame, frame, frame, frame))
    summary = pd.DataFrame({"mean_daily_rank_ic": [0.0], "net_sharpe": [0.0], "max_drawdown": [0.0]})
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {"summary": summary})

    def report(output, result, manifest, *args):
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "report.html").write_text("Test fixture, not research performance")
        return "fixture complete"

    monkeypatch.setattr(reporting, "write_report", report)
    return project, data, experiment


def test_run_registers_only_after_snapshot_and_completes_after_verification(mocked_research):
    project, data, _ = mocked_research
    output = project / "artifacts/run"
    cli.main(["run", "--data", str(data), "--output", str(output)])
    events = read_registry(project / "artifacts/research_registry.jsonl")
    assert [row["event"] for row in events] == ["run_started", "run_completed"]
    verified = verify_run(output, data)
    assert events[-1]["payload"]["artifacts_sha256"] == verified["artifact_inventory_sha256"]
    assert events[0]["payload"]["source_sha256"] == verified["source_manifest_sha256"]
    assert events[0]["payload"]["data_sha256"] == verified["data_sha256"]


def test_run_failure_preserves_snapshot_and_appends_failure(mocked_research):
    project, data, experiment = mocked_research

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate fitting failure")

    experiment.run_forecasts = fail
    output = project / "artifacts/failed-run"
    with pytest.raises(RuntimeError, match="deliberate"):
        cli.main(["run", "--data", str(data), "--output", str(output)])
    events = read_registry(project / "artifacts/research_registry.jsonl")
    assert [row["event"] for row in events] == ["run_started", "run_failed"]
    assert "deliberate fitting failure" in events[-1]["payload"]["error"]
    assert (output / "source_snapshot/docs/RESEARCH_PROTOCOL.md").is_file()
    assert json.loads((output / "FAILURE.json").read_text())["error_type"] == "RuntimeError"
    with pytest.raises(IntegrityError, match="Failed run"):
        verify_run(output)


def test_artifact_verification_failure_prevents_completed_event(mocked_research, monkeypatch):
    from equity_alpha import reporting

    project, data, _ = mocked_research
    original = reporting.write_report

    def tampered_report(output, *args):
        verdict = original(output, *args)
        (output / "config.json").write_text('{"unexpected": "changed during reporting"}')
        return verdict

    monkeypatch.setattr(reporting, "write_report", tampered_report)
    output = project / "artifacts/tampered-run"
    with pytest.raises(IntegrityError, match="configuration checksum"):
        cli.main(["run", "--data", str(data), "--output", str(output)])
    events = read_registry(project / "artifacts/research_registry.jsonl")
    assert [row["event"] for row in events] == ["run_started", "run_failed"]
    assert (output / "FAILURE.json").is_file()


def test_existing_output_and_mutable_registry_inside_run_are_rejected(mocked_research):
    project, data, _ = mocked_research
    output = project / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("existing user artifact")
    with pytest.raises(FileExistsError):
        cli.main(["run", "--data", str(data), "--output", str(output)])
    assert (output / "keep.txt").read_text() == "existing user artifact"
    output = project / "new_run"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "run",
                "--data",
                str(data),
                "--output",
                str(output),
                "--registry",
                str(output / "registry.jsonl"),
            ]
        )
    assert exc.value.code == 2
    assert not output.exists()
