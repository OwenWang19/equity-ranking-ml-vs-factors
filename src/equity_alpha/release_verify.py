"""Portable, read-only verification of a public aggregate release subset.

Only files inside the explicitly supplied release directory are read. Recorded
native paths and working-tree source hashes are provenance, not paths to follow
or requirements on the reviewer's checkout. Model/NumPy payloads are forbidden;
Python snapshots are hashed as inert bytes, never imported or executed.

Checksums establish local internal consistency, not authenticity, scientific
validity, immutable trial history, or permission to redistribute vendor data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

MANIFEST = "release_manifest.json"
SCHEMA = "wide_ml_public_subset_manifest_v1"
MAX_MANIFEST_BYTES = 8_000_000
FORBIDDEN_SUFFIXES = frozenset((".npy", ".npz", ".joblib", ".pkl", ".pickle"))


class ReleaseIntegrityError(ValueError):
    """An aggregate release is incomplete, inconsistent, or unsafe to inspect."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseIntegrityError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ReleaseIntegrityError("Release manifest exceeds the bounded JSON size")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ReleaseIntegrityError("Nonfinite JSON in release manifest")
        return number

    def invalid_constant(value):
        raise ReleaseIntegrityError(f"Nonfinite JSON in release manifest: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_float=finite_float,
            parse_constant=invalid_constant,
        )
    except ReleaseIntegrityError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ReleaseIntegrityError("Invalid JSON in release manifest") from exc


def _safe_name(value):
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\\", ":"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReleaseIntegrityError(f"Unsafe release path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ReleaseIntegrityError(f"Unsafe release path: {value!r}")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReleaseIntegrityError(f"Invalid SHA-256: {name}")
    return value


def _hash_map(value, name):
    if not isinstance(value, dict) or not value:
        raise ReleaseIntegrityError(f"Missing or empty {name}")
    for relative, digest in value.items():
        _safe_name(relative)
        _digest(digest, relative)
    return value


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root):
    # Do not resolve first: resolution would erase evidence of symlink components.
    absolute = root.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ReleaseIntegrityError("Symlink forbidden in explicitly supplied release path")
    if not absolute.is_dir():
        raise ReleaseIntegrityError("Explicit release directory is missing or not a directory")
    files, directories = set(), set()

    def walk_error(error):
        raise error

    for directory, dirs, names in os.walk(absolute, followlinks=False, onerror=walk_error):
        for name in dirs + names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            relative = path.relative_to(absolute).as_posix()
            if stat.S_ISLNK(mode):
                raise ReleaseIntegrityError(f"Symlink forbidden: {relative}")
            if stat.S_ISDIR(mode):
                directories.add(relative)
            elif stat.S_ISREG(mode):
                files.add(relative)
            else:
                raise ReleaseIntegrityError(f"Nonregular release artifact: {relative}")
    return files, directories


def verify_release(release_dir: Path) -> dict:
    """Verify only the explicit public subset; never open recorded external paths.

    The manifest itself is excluded from its file hash map. Every other file and
    directory must be declared or needed by a declared relative file path. This
    is a stable local-snapshot check, not protection against concurrent mutation
    by an active adversary or a maliciously replaced, coherently rehashed release.
    """
    root = Path(release_dir)
    try:
        files, directories = _inventory(root)
        if MANIFEST not in files:
            raise ReleaseIntegrityError("Release manifest is missing")
        manifest = _json(root / MANIFEST)
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ReleaseIntegrityError("Unsupported public release manifest schema")
        if manifest.get("manifest_self_excluded") is not True:
            raise ReleaseIntegrityError("Manifest must explicitly exclude its own checksum")
        if manifest.get("frozen_run_modified") is not False:
            raise ReleaseIntegrityError("Release must attest that the frozen run was not modified")
        if manifest.get("reserved_2011_performance_evaluated") is not False:
            raise ReleaseIntegrityError("Release must attest that 2011 performance remains reserved")
        run_id = manifest.get("native_run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ReleaseIntegrityError("Missing native run identity")
        _digest(manifest.get("native_artifact_inventory_digest"), "native artifact inventory")
        hashes = _hash_map(manifest.get("public_subset_hashes"), "public subset hashes")
        if MANIFEST in hashes:
            raise ReleaseIntegrityError("Self-referential manifest checksum is forbidden")
        if files != set(hashes) | {MANIFEST}:
            missing = sorted(set(hashes) - files)
            extra = sorted(files - set(hashes) - {MANIFEST})
            raise ReleaseIntegrityError(f"Release file inventory mismatch: missing={missing}, extra={extra}")
        required_directories = {
            str(parent) for name in hashes for parent in PurePosixPath(name).parents if str(parent) != "."
        }
        if directories != required_directories:
            raise ReleaseIntegrityError("Release directory inventory mismatch")
        for relative, digest in hashes.items():
            if PurePosixPath(relative).suffix.lower() in FORBIDDEN_SUFFIXES:
                raise ReleaseIntegrityError(
                    f"Private model/array payload forbidden in public subset: {relative}"
                )
            if _file_digest(root / relative) != digest:
                raise ReleaseIntegrityError(f"Release checksum mismatch: {relative}")
        # Validate provenance syntax only. Never follow recorded checkout paths.
        if "post_run_working_source_hashes" in manifest:
            _hash_map(manifest["post_run_working_source_hashes"], "working source provenance hashes")
        return {
            "status": "verified",
            "schema": SCHEMA,
            "native_run_id": run_id,
            "public_artifact_count": len(hashes),
            "manifest_self_excluded": True,
            "manifest_sha256": _file_digest(root / MANIFEST),
            "native_artifact_inventory_digest": manifest["native_artifact_inventory_digest"],
            "external_paths_opened": False,
            "assurance": "public_subset_internal_consistency_only_not_authenticity_or_scientific_validation",
        }
    except ReleaseIntegrityError:
        raise
    except (OSError, UnicodeError, TypeError) as exc:
        raise ReleaseIntegrityError(str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_release(args.release)
    except ReleaseIntegrityError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, allow_nan=False))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
