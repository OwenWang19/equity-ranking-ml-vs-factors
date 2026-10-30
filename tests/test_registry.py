"""Lifecycle rules and concurrent single-use holdout authorization."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from equity_alpha.registry import (
    RegistryError,
    complete_run,
    fail_run,
    read_registry,
    release_holdout,
    start_run,
)


def _start(path, run_id="run_1", experiment="exp_1"):
    return start_run(
        path, experiment, run_id, config_sha256="a" * 64, data_sha256="b" * 64, source_sha256="c" * 64
    )


def test_append_only_complete_lifecycle_and_release_binding(tmp_path):
    path = tmp_path / "registry.jsonl"
    assert read_registry(path) == []
    first = _start(path)
    initial_bytes = path.read_bytes()
    second = complete_run(path, "exp_1", "run_1", artifacts_sha256="d" * 64)
    assert path.read_bytes().startswith(initial_bytes)
    release = release_holdout(path, "exp_1", "run_1", reason="Protocol frozen before final evaluation")
    assert second["previous_sha256"] == first["event_sha256"]
    assert release["payload"]["config_sha256"] == "a" * 64
    assert release["payload"]["release_consumed"] is True
    assert [x["sequence"] for x in read_registry(path)] == [1, 2, 3]


def test_failures_remain_in_registry_and_retry_requires_new_run_id(tmp_path):
    path = tmp_path / "registry.jsonl"
    _start(path)
    fail_run(path, "exp_1", "run_1", error="Training process failed before evaluation")
    with pytest.raises(RegistryError, match="already registered"):
        _start(path)
    with pytest.raises(RegistryError, match="completed"):
        release_holdout(path, "exp_1", "run_1", reason="retry")
    _start(path, "run_2")
    assert [x["event"] for x in read_registry(path)] == ["run_started", "run_failed", "run_started"]


def test_terminal_status_cannot_be_overwritten(tmp_path):
    path = tmp_path / "registry.jsonl"
    _start(path)
    complete_run(path, "exp_1", "run_1", artifacts_sha256="d" * 64)
    before = path.read_bytes()
    for operation in (
        lambda: complete_run(path, "exp_1", "run_1", artifacts_sha256="e" * 64),
        lambda: fail_run(path, "exp_1", "run_1", error="change result"),
    ):
        with pytest.raises(RegistryError, match="unterminated"):
            operation()
    assert path.read_bytes() == before


def test_one_release_per_experiment_even_from_a_different_completed_run(tmp_path):
    path = tmp_path / "registry.jsonl"
    for name in ("run_1", "run_2"):
        _start(path, name)
        complete_run(path, "exp_1", name, artifacts_sha256="d" * 64)
    release_holdout(path, "exp_1", "run_1", reason="release")
    before = path.read_bytes()
    with pytest.raises(RegistryError, match="consumed"):
        release_holdout(path, "exp_1", "run_2", reason="second look")
    assert path.read_bytes() == before


def test_experiments_have_separate_release_allowances(tmp_path):
    path = tmp_path / "registry.jsonl"
    for n in range(2):
        experiment, run = f"exp_{n}", f"run_{n}"
        _start(path, run, experiment)
        complete_run(path, experiment, run, artifacts_sha256="d" * 64)
        release_holdout(path, experiment, run, reason="release")
    assert sum(x["event"] == "holdout_released" for x in read_registry(path)) == 2


def test_development_cannot_continue_under_same_experiment_after_release(tmp_path):
    path = tmp_path / "registry.jsonl"
    _start(path)
    complete_run(path, "exp_1", "run_1", artifacts_sha256="d" * 64)
    release_holdout(path, "exp_1", "run_1", reason="locked protocol")
    with pytest.raises(RegistryError, match="frozen"):
        _start(path, "run_2")


def test_run_id_cannot_be_reused_under_a_different_experiment(tmp_path):
    path = tmp_path / "registry.jsonl"
    _start(path)
    with pytest.raises(RegistryError, match="different experiment"):
        _start(path, experiment="exp_2")


@pytest.mark.parametrize("raw", [b"{", b"{}\n", b"\n", b"{}\n{", b'{"a":1,"a":2}\n'])
def test_corrupt_or_partial_registry_fails_closed_on_read_and_append(tmp_path, raw):
    path = tmp_path / "registry.jsonl"
    path.write_bytes(raw)
    with pytest.raises(RegistryError):
        read_registry(path)
    with pytest.raises(RegistryError):
        _start(path)
    assert path.read_bytes() == raw


def test_changed_and_reordered_events_detected(tmp_path):
    path = tmp_path / "registry.jsonl"
    _start(path)
    complete_run(path, "exp_1", "run_1", artifacts_sha256="d" * 64)
    lines = path.read_text().splitlines()
    changed = json.loads(lines[0])
    changed["payload"]["config_sha256"] = "e" * 64
    path.write_text(json.dumps(changed) + "\n" + lines[1] + "\n")
    with pytest.raises(RegistryError, match="checksum"):
        read_registry(path)
    path.write_text(lines[1] + "\n" + lines[0] + "\n")
    with pytest.raises(RegistryError, match="sequence"):
        read_registry(path)


def test_concurrent_starts_do_not_lose_or_interleave_events(tmp_path):
    path = tmp_path / "registry.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        events = list(pool.map(lambda n: _start(path, f"run_{n}"), range(24)))
    assert len(read_registry(path)) == len(events) == 24
    assert len({row["sequence"] for row in events}) == 24


def test_concurrent_holdout_releases_have_exactly_one_winner(tmp_path):
    path = tmp_path / "registry.jsonl"
    for n in range(8):
        _start(path, f"run_{n}")
        complete_run(path, "exp_1", f"run_{n}", artifacts_sha256="d" * 64)

    def attempt(n):
        try:
            return release_holdout(path, "exp_1", f"run_{n}", reason="release")["event"]
        except RegistryError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))
    assert results.count("holdout_released") == 1
    assert results.count("rejected") == 7
    assert len(read_registry(path)) == 17


def test_invalid_metadata_and_registry_symlink_are_rejected(tmp_path):
    path = tmp_path / "registry.jsonl"
    with pytest.raises(RegistryError, match="config_sha256"):
        start_run(path, "exp", "run", config_sha256="invalid", data_sha256="a" * 64, source_sha256="b" * 64)
    real = tmp_path / "real.jsonl"
    real.write_text("")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(real)
    with pytest.raises(RegistryError, match="symlinks"):
        _start(alias)
    with pytest.raises(RegistryError, match="symlinks"):
        read_registry(alias)
