"""Audit failures must never be treated as a successfully frozen run."""

import hashlib
import json

import pytest

from equity_alpha.integrity import IntegrityError, canonical_digest, verify_run


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_inventory(run):
    hashes = {
        p.relative_to(run).as_posix(): _sha(p)
        for p in run.rglob("*")
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    (run / "artifact_hashes.json").write_text(json.dumps(hashes))


@pytest.fixture
def frozen_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    data = tmp_path / "input.csv"
    data.write_text("date,ticker,price\n2020-01-01,A,100\n")
    config = {"experiment_id": "tiny", "horizon": 5, "models": ["linear", "tree"]}
    source = {}
    for name in (
        "README.md",
        "pyproject.toml",
        "src/model.py",
        "tests/test_model.py",
        "configs/research.json",
        "docs/RESEARCH_PROTOCOL.md",
    ):
        path = run / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Frozen source for {name}\n")
        source[name] = _sha(path)
    manifest = {
        "config": config,
        "config_sha256": canonical_digest(config),
        "data": {"source_sha256": _sha(data), "source_path": "/must/not/open/this.csv"},
        "source_sha256": source,
    }
    for name in ("manifest.json", "run_started.json"):
        (run / name).write_text(json.dumps(manifest, indent=2))
    (run / "config.json").write_text(json.dumps(config, indent=4))
    (run / "report.html").write_text("<p>result</p>")
    _refresh_inventory(run)
    return run, data


def test_verification_is_read_only_and_does_not_open_recorded_source(frozen_run):
    run, data = frozen_run
    before = {p: p.stat().st_mtime_ns for p in run.rglob("*")}
    result = verify_run(run)
    assert result["status"] == "verified"
    assert result["data_verified"] is False
    assert result["source_file_count"] == 6
    assert verify_run(run, data)["data_verified"] is True
    assert before == {p: p.stat().st_mtime_ns for p in run.rglob("*")}


@pytest.mark.parametrize("action", ["tamper", "remove", "add", "failure", "no_hashes"])
def test_missing_tampered_extra_or_failed_artifacts_fail_closed(frozen_run, action):
    run, _ = frozen_run
    if action == "tamper":
        (run / "report.html").write_text("changed")
    elif action == "remove":
        (run / "report.html").unlink()
    elif action == "add":
        (run / "unrecorded.txt").write_text("new")
    elif action == "failure":
        (run / "FAILURE.json").write_text("{}")
        _refresh_inventory(run)
    else:
        (run / "artifact_hashes.json").unlink()
    with pytest.raises(IntegrityError):
        verify_run(run)


@pytest.mark.parametrize(
    "unsafe",
    [
        "../secret",
        "/etc/passwd",
        "src/../../secret",
        "src\\model.py",
        "./report.html",
        "C:/file",
        "src//file",
    ],
)
def test_hash_inventory_paths_cannot_escape_or_alias_run(frozen_run, unsafe):
    run, _ = frozen_run
    hashes = json.loads((run / "artifact_hashes.json").read_text())
    hashes[unsafe] = "a" * 64
    (run / "artifact_hashes.json").write_text(json.dumps(hashes))
    with pytest.raises(IntegrityError, match="Unsafe"):
        verify_run(run)


def test_symlinks_are_rejected_even_when_they_reference_valid_inside_file(frozen_run):
    run, _ = frozen_run
    (run / "alias.html").symlink_to(run / "report.html")
    with pytest.raises(IntegrityError, match="Symlink"):
        verify_run(run)


def test_symlinked_directory_and_run_root_are_rejected(frozen_run, tmp_path):
    run, _ = frozen_run
    linked = tmp_path / "alias"
    linked.symlink_to(run, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlink"):
        verify_run(linked)
    (run / "subdir").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(IntegrityError, match="Symlink"):
        verify_run(run)


def test_source_manifest_checked_independently_of_artifact_inventory(frozen_run):
    run, _ = frozen_run
    (run / "source_snapshot/src/model.py").write_text("silently revised model")
    _refresh_inventory(run)
    with pytest.raises(IntegrityError, match="Frozen source checksum"):
        verify_run(run)


def test_missing_documentation_is_not_a_complete_source_snapshot(frozen_run):
    run, _ = frozen_run
    (run / "source_snapshot/docs/RESEARCH_PROTOCOL.md").unlink()
    for name in ("manifest.json", "run_started.json"):
        path = run / name
        value = json.loads(path.read_text())
        del value["source_sha256"]["docs/RESEARCH_PROTOCOL.md"]
        path.write_text(json.dumps(value))
    _refresh_inventory(run)
    with pytest.raises(IntegrityError, match="Incomplete frozen source: docs"):
        verify_run(run)


def test_config_uses_normalized_json_not_formatting(frozen_run):
    run, _ = frozen_run
    config = json.loads((run / "config.json").read_text())
    (run / "config.json").write_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    _refresh_inventory(run)
    assert verify_run(run)["status"] == "verified"
    config["horizon"] = 10
    (run / "config.json").write_text(json.dumps(config))
    _refresh_inventory(run)
    with pytest.raises(IntegrityError, match="configuration checksum"):
        verify_run(run)


def test_started_and_final_manifest_must_match(frozen_run):
    run, _ = frozen_run
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["data"]["source_sha256"] = "a" * 64
    (run / "manifest.json").write_text(json.dumps(manifest))
    _refresh_inventory(run)
    with pytest.raises(IntegrityError, match="Started/final"):
        verify_run(run)


def test_optional_input_must_match_frozen_input(frozen_run):
    run, data = frozen_run
    data.write_text("revised market data")
    with pytest.raises(IntegrityError, match="Source input checksum"):
        verify_run(run, data)
    assert verify_run(run)["data_verified"] is False


def test_duplicate_hash_keys_rejected(frozen_run):
    run, _ = frozen_run
    (run / "artifact_hashes.json").write_text('{"x": "a", "x": "b"}')
    with pytest.raises(IntegrityError, match="Duplicate"):
        verify_run(run)


def test_nonfinite_config_has_no_canonical_hash():
    with pytest.raises(IntegrityError, match="nonfinite"):
        canonical_digest({"threshold": float("nan")})
