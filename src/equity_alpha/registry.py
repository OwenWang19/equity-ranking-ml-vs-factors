"""Append-only local experiment lifecycle with a hash chain and POSIX file lock.

The registry records every development attempt and consumes a holdout release at
authorization time. It cannot undo human exposure or prove file authenticity;
external immutable storage is needed to detect wholesale replacement/truncation.
"""

import fcntl
import hashlib
import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path


class RegistryError(ValueError):
    """The registry is invalid or a requested lifecycle transition is forbidden."""


_EVENTS = {"run_started", "run_completed", "run_failed", "holdout_released"}
_FIELDS = {
    "schema_version",
    "sequence",
    "timestamp_utc",
    "experiment_id",
    "run_id",
    "event",
    "payload",
    "previous_sha256",
    "event_sha256",
}


def _canonical(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise RegistryError("Registry payload must be finite JSON") from exc


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise RegistryError(f"Invalid {name}")


def _validate_digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RegistryError(f"Invalid {name}: expected lowercase SHA-256")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _transition(events, experiment_id, run_id, event, payload):
    _validate_id(experiment_id, "experiment_id")
    _validate_id(run_id, "run_id")
    if event not in _EVENTS or not isinstance(payload, dict):
        raise RegistryError("Unknown event or invalid payload")
    same_run = [row for row in events if row["run_id"] == run_id]
    if same_run and any(row["experiment_id"] != experiment_id for row in same_run):
        raise RegistryError("run_id already belongs to a different experiment")
    if event == "run_started":
        if same_run:
            raise RegistryError("run_id already registered; every attempt needs a new run_id")
        if any(
            row["experiment_id"] == experiment_id and row["event"] == "holdout_released" for row in events
        ):
            raise RegistryError("Experiment is frozen after its holdout release was consumed")
        for name in ("config_sha256", "data_sha256", "source_sha256"):
            _validate_digest(payload.get(name), name)
    elif event in {"run_completed", "run_failed"}:
        if not same_run or same_run[-1]["event"] != "run_started":
            raise RegistryError("Run must be started and unterminated")
        if event == "run_completed":
            _validate_digest(payload.get("artifacts_sha256"), "artifacts_sha256")
        elif not isinstance(payload.get("error"), str) or not payload["error"].strip():
            raise RegistryError("Failed run must record an error")
    else:
        if not same_run or same_run[-1]["event"] != "run_completed":
            raise RegistryError("Holdout release requires a completed development run")
        if any(
            row["experiment_id"] == experiment_id and row["event"] == "holdout_released" for row in events
        ):
            raise RegistryError("Holdout release already consumed for this experiment")
        if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
            raise RegistryError("Holdout release requires a reason")
        expected = same_run[0]["payload"]
        for name in ("config_sha256", "data_sha256", "source_sha256"):
            if payload.get(name) != expected.get(name):
                raise RegistryError(f"Holdout release binding mismatch: {name}")


def _decode(raw: bytes) -> list[dict]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise RegistryError("Incomplete registry tail; refuse to append or silently repair")
    events = []
    previous = "0" * 64
    for sequence, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Invalid registry JSON at sequence {sequence}") from exc
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise RegistryError(f"Invalid registry schema at sequence {sequence}")
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != 1
            or type(row["sequence"]) is not int
            or row["sequence"] != sequence
        ):
            raise RegistryError("Registry sequence mismatch")
        if row["previous_sha256"] != previous:
            raise RegistryError("Registry hash chain mismatch")
        expected = _digest({key: value for key, value in row.items() if key != "event_sha256"})
        if row["event_sha256"] != expected:
            raise RegistryError("Registry event checksum mismatch")
        try:
            timestamp = datetime.fromisoformat(row["timestamp_utc"])
            if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
                raise ValueError("timestamp must be UTC")
        except (ValueError, TypeError, AttributeError) as exc:
            raise RegistryError("Registry timestamp is invalid") from exc
        _transition(events, row["experiment_id"], row["run_id"], row["event"], row["payload"])
        events.append(row)
        previous = row["event_sha256"]
    return events


def read_registry(path: Path) -> list[dict]:
    """Read and verify every event without changing the registry."""
    path = Path(path)
    if path.is_symlink():
        raise RegistryError("Registry symlinks are forbidden")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RegistryError("Registry must be a regular file")
            fcntl.flock(handle, fcntl.LOCK_SH)
            return _decode(handle.read())
    except FileNotFoundError:
        return []


def _append(path, experiment_id, run_id, event, payload_factory):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RegistryError("Registry symlinks are forbidden")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "r+b", buffering=0) as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RegistryError("Registry must be a regular file")
        fcntl.flock(handle, fcntl.LOCK_EX)
        events = _decode(handle.read())
        payload = payload_factory(events)
        _transition(events, experiment_id, run_id, event, payload)
        row = {
            "schema_version": 1,
            "sequence": len(events) + 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "event": event,
            "payload": payload,
            "previous_sha256": events[-1]["event_sha256"] if events else "0" * 64,
        }
        row["event_sha256"] = _digest(row)
        encoded = _canonical(row) + b"\n"
        # The complete event is appended under one exclusive lock; a partial crash
        # tail is rejected by the next reader, never discarded automatically.
        offset = 0
        while offset < len(encoded):
            written = os.write(handle.fileno(), encoded[offset:])
            if written == 0:
                raise OSError("Registry append made no progress")
            offset += written
        os.fsync(handle.fileno())
        return row


def start_run(
    path: Path,
    experiment_id: str,
    run_id: str,
    *,
    config_sha256: str,
    data_sha256: str,
    source_sha256: str,
    output_dir: str | None = None,
) -> dict:
    payload = {"config_sha256": config_sha256, "data_sha256": data_sha256, "source_sha256": source_sha256}
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    return _append(path, experiment_id, run_id, "run_started", lambda _: payload)


def complete_run(path: Path, experiment_id: str, run_id: str, *, artifacts_sha256: str) -> dict:
    return _append(
        path, experiment_id, run_id, "run_completed", lambda _: {"artifacts_sha256": artifacts_sha256}
    )


def fail_run(path: Path, experiment_id: str, run_id: str, *, error: str) -> dict:
    return _append(path, experiment_id, run_id, "run_failed", lambda _: {"error": error})


def release_holdout(path: Path, experiment_id: str, run_id: str, *, reason: str) -> dict:
    """Consume the experiment's one release before evaluating any reserved result."""

    def payload(events):
        matching = [row for row in events if row["run_id"] == run_id and row["event"] == "run_started"]
        if not matching:
            raise RegistryError("Holdout release requires a registered run")
        start = matching[0]["payload"]
        return {name: start[name] for name in ("config_sha256", "data_sha256", "source_sha256")} | {
            "reason": reason,
            "release_consumed": True,
        }

    return _append(path, experiment_id, run_id, "holdout_released", payload)
