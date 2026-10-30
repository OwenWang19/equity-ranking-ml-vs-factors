"""Fail-closed, read-only verification of a frozen research run.

Hashes establish internal consistency, not authorship: an external trusted digest is
needed to detect a malicious replacement of an entire run and its hash inventory.
No executable model is deserialized during verification.
"""

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


class IntegrityError(ValueError):
    """A run cannot be established as a complete, internally consistent artifact."""


def canonical_digest(value) -> str:
    """Match the project's sorted-key JSON configuration fingerprint."""
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise IntegrityError("Noncanonical or nonfinite JSON content") from exc
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"Cannot read artifact: {path}") from exc
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path):
    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(IntegrityError(f"Nonfinite JSON: {value}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Cannot read valid JSON: {path.name}") from exc


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise IntegrityError(f"Unsafe artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise IntegrityError(f"Unsafe artifact path: {value!r}")
    return value


def _hash_map(value, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise IntegrityError(f"Missing or empty {name}")
    result = {}
    for key, digest in value.items():
        key = _safe_relative(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise IntegrityError(f"Invalid SHA-256 in {name}: {key}")
        result[key] = digest
    return result


def _inventory(root: Path) -> set[str]:
    files = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise IntegrityError(f"Symlink forbidden in frozen run: {path.relative_to(root)}")
            if not path.is_dir() and not path.is_file():
                raise IntegrityError(f"Nonregular artifact: {path.relative_to(root)}")
        files.update((Path(directory) / name).relative_to(root).as_posix() for name in names)
    return files


def verify_run(run_dir: Path, data_path: Path | None = None) -> dict:
    """Verify exact artifact/source inventories and config, optionally original data.

    Raises ``IntegrityError`` on missing, additional, unsafe, or modified files.
    The inventory intentionally excludes only ``artifact_hashes.json`` itself.
    It never repairs artifacts or reads the input path recorded in the manifest;
    original data is opened only when explicitly supplied by the caller.
    """
    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("Run directory is missing or is a symlink")
    inventory = _inventory(root)
    if "FAILURE.json" in inventory:
        raise IntegrityError("Failed run cannot be verified as complete")
    hashes = _hash_map(_read_json(root / "artifact_hashes.json"), "artifact inventory")
    required = {"manifest.json", "config.json", "run_started.json"}
    if not required.issubset(hashes):
        raise IntegrityError(f"Required artifacts missing: {sorted(required - hashes.keys())}")
    if "artifact_hashes.json" in hashes:
        raise IntegrityError("Artifact inventory cannot hash itself")
    actual = inventory - {"artifact_hashes.json"}
    if actual != hashes.keys():
        raise IntegrityError(
            f"Artifact inventory mismatch; missing={sorted(hashes.keys() - actual)}, "
            f"unexpected={sorted(actual - hashes.keys())}"
        )
    for relative, expected in hashes.items():
        if file_digest(root / relative) != expected:
            raise IntegrityError(f"Artifact checksum mismatch: {relative}")
    manifest = _read_json(root / "manifest.json")
    started = _read_json(root / "run_started.json")
    if not isinstance(manifest, dict) or not isinstance(started, dict):
        raise IntegrityError("Manifest must be a JSON object")
    for field in ("config", "config_sha256", "data", "source_sha256"):
        if field not in manifest or started.get(field) != manifest[field]:
            raise IntegrityError(f"Started/final manifest mismatch: {field}")
    config = _read_json(root / "config.json")
    if not isinstance(config, dict) or not isinstance(manifest["config"], dict):
        raise IntegrityError("Configuration must be a JSON object")
    config_sha = canonical_digest(config)
    if config_sha != manifest["config_sha256"] or canonical_digest(manifest["config"]) != config_sha:
        raise IntegrityError("Normalized configuration checksum mismatch")
    source = _hash_map(manifest["source_sha256"], "source manifest")
    for name in ("README.md", "pyproject.toml"):
        if name not in source:
            raise IntegrityError(f"Incomplete frozen source: {name}")
    for prefix in ("src/", "tests/", "configs/", "docs/"):
        if not any(name.startswith(prefix) for name in source):
            raise IntegrityError(f"Incomplete frozen source: {prefix}")
    snapshot_files = {
        name.removeprefix("source_snapshot/") for name in actual if name.startswith("source_snapshot/")
    }
    if snapshot_files != source.keys():
        raise IntegrityError("Frozen source inventory mismatch")
    for relative, expected in source.items():
        if hashes[f"source_snapshot/{relative}"] != expected:
            raise IntegrityError(f"Frozen source checksum mismatch: {relative}")
    data = manifest["data"]
    if not isinstance(data, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("source_sha256", ""))):
        raise IntegrityError("Invalid source input checksum")
    if data_path is not None and file_digest(Path(data_path)) != data["source_sha256"]:
        raise IntegrityError("Source input checksum mismatch")
    return {
        "status": "verified",
        "artifact_count": len(hashes),
        "source_file_count": len(source),
        "config_sha256": config_sha,
        "data_sha256": data["source_sha256"],
        "data_verified": data_path is not None,
        "artifact_inventory_sha256": file_digest(root / "artifact_hashes.json"),
        "source_manifest_sha256": canonical_digest(source),
        "assurance": "internal_consistency_only_not_authorship_or_immutable_storage",
    }
