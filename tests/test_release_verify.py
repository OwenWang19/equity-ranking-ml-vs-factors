"""Portable public-subset checks; synthetic files only, no Docker or vendor data."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from equity_alpha.release_verify import SCHEMA, ReleaseIntegrityError, main, verify_release


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(release):
    return json.loads((release / "release_manifest.json").read_text())


def write_manifest(release, value):
    (release / "release_manifest.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    (root / "source_snapshot").mkdir(parents=True)
    (root / "report.html").write_text("<h1>Synthetic report</h1>", encoding="utf-8")
    (root / "summary.json").write_text('{"synthetic": true}', encoding="utf-8")
    # Python snapshots are inert bytes: executing this would make a passing test fail.
    (root / "source_snapshot" / "source.py").write_text("raise RuntimeError('MUST NOT EXECUTE')\n")
    write_manifest(
        root,
        {
            "schema": SCHEMA,
            "native_run_id": "wide_synthetic",
            "native_artifact_inventory_digest": "1" * 64,
            "public_subset_hashes": {
                path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()
            },
            "post_run_working_source_hashes": {"nonexistent/checkout/source.py": "2" * 64},
            "manifest_self_excluded": True,
            "frozen_run_modified": False,
            "reserved_2011_performance_evaluated": False,
        },
    )
    return root


def test_valid_release_is_portable_and_does_not_execute_or_follow_provenance(release, tmp_path):
    relocated = tmp_path / "relocated"
    shutil.copytree(release, relocated)
    result = verify_release(relocated)
    assert result["status"] == "verified"
    assert result["public_artifact_count"] == 3
    assert result["manifest_sha256"] == digest(relocated / "release_manifest.json")
    assert result["external_paths_opened"] is False
    assert "not_authenticity" in result["assurance"]


@pytest.mark.parametrize("change", ["tampered", "missing", "extra", "empty_directory"])
def test_exact_inventory_and_hashes(release, change):
    if change == "tampered":
        (release / "report.html").write_text("changed")
    elif change == "missing":
        (release / "report.html").unlink()
    elif change == "extra":
        (release / ".DS_Store").write_bytes(b"extra")
    else:
        (release / "unlisted-empty-directory").mkdir()
    with pytest.raises(ReleaseIntegrityError):
        verify_release(release)


@pytest.mark.parametrize(
    "unsafe",
    ["../secret", "/etc/passwd", "./report.html", "x//y", "x\\y", "C:/x", "x\x00y", "x\ny", ".", ""],
)
def test_unsafe_manifest_paths(release, unsafe):
    manifest = read_manifest(release)
    manifest["public_subset_hashes"] = {unsafe: "a" * 64}
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError, match="Unsafe"):
        verify_release(release)


@pytest.mark.parametrize("digest_value", ["f" * 63, "g" * 64, "F" * 64, 1, True, None, {}])
def test_invalid_sha256(release, digest_value):
    manifest = read_manifest(release)
    manifest["public_subset_hashes"]["report.html"] = digest_value
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError, match="SHA-256"):
        verify_release(release)


@pytest.mark.parametrize("value", [None, [], {}, "hashes", False])
def test_nonempty_object_required_for_hash_map(release, value):
    manifest = read_manifest(release)
    manifest["public_subset_hashes"] = value
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError):
        verify_release(release)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "unknown"},
        {"native_run_id": ""},
        {"native_run_id": None},
        {"native_artifact_inventory_digest": "wrong"},
        {"manifest_self_excluded": 1},
        {"manifest_self_excluded": False},
        {"frozen_run_modified": True},
        {"frozen_run_modified": 0},
        {"reserved_2011_performance_evaluated": True},
        {"reserved_2011_performance_evaluated": 0},
    ],
)
def test_release_metadata_is_fail_closed(release, changes):
    manifest = read_manifest(release)
    manifest.update(changes)
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError):
        verify_release(release)


def test_manifest_cannot_hash_itself(release):
    manifest = read_manifest(release)
    manifest["public_subset_hashes"]["release_manifest.json"] = "a" * 64
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError, match="Self-referential"):
        verify_release(release)


@pytest.mark.parametrize(
    "contents",
    [
        '{"a": 1, "a": 2}',
        '{"a": {"b": 1, "b": 2}}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
        '{"a": 1e999}',
        '{"a": 1} trailing',
        "[1, 2, 3]",
        "{",
    ],
)
def test_strict_json(release, contents):
    (release / "release_manifest.json").write_text(contents)
    with pytest.raises(ReleaseIntegrityError):
        verify_release(release)


def test_invalid_utf8_manifest(release):
    (release / "release_manifest.json").write_bytes(b"\xff")
    with pytest.raises(ReleaseIntegrityError):
        verify_release(release)


@pytest.mark.parametrize("location", ["file", "directory", "root", "ancestor", "broken"])
def test_symlinks_forbidden(release, tmp_path, location):
    target = release
    if location == "file":
        (release / "report.html").unlink()
        (release / "report.html").symlink_to(tmp_path / "outside")
    elif location == "directory":
        shutil.rmtree(release / "source_snapshot")
        (release / "source_snapshot").symlink_to(tmp_path / "outside", target_is_directory=True)
    elif location in {"root", "ancestor"}:
        alias = tmp_path / "alias"
        alias.symlink_to(release if location == "root" else release.parent, target_is_directory=True)
        target = alias if location == "root" else alias / release.name
    else:
        (release / "broken").symlink_to(tmp_path / "absent")
    with pytest.raises(ReleaseIntegrityError, match="Symlink"):
        verify_release(target)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO not available")
def test_special_file_rejected_without_blocking_read(release):
    os.mkfifo(release / "fifo")
    with pytest.raises(ReleaseIntegrityError, match="Nonregular"):
        verify_release(release)


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".joblib", ".pkl", ".pickle", ".NPZ"])
def test_private_payloads_rejected_even_if_coherently_hashed(release, suffix):
    path = release / f"private{suffix}"
    path.write_bytes(b"opaque potentially executable bytes")
    manifest = read_manifest(release)
    manifest["public_subset_hashes"][path.name] = digest(path)
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError, match="Private model/array"):
        verify_release(release)


def test_external_source_hashes_are_provenance_only_but_syntax_is_checked(release):
    manifest = read_manifest(release)
    manifest["post_run_working_source_hashes"] = {"../outside.py": "a" * 64}
    write_manifest(release, manifest)
    with pytest.raises(ReleaseIntegrityError, match="Unsafe"):
        verify_release(release)


@pytest.mark.parametrize("location", ["missing_root", "root_file", "manifest_missing"])
def test_missing_or_wrong_root(release, tmp_path, location):
    target = tmp_path / "missing"
    if location == "root_file":
        target.write_text("not a directory")
    elif location == "manifest_missing":
        (release / "release_manifest.json").unlink()
        target = release
    with pytest.raises(ReleaseIntegrityError):
        verify_release(target)


def test_bounded_manifest_size(release):
    with (release / "release_manifest.json").open("wb") as handle:
        handle.truncate(8_000_001)
    with pytest.raises(ReleaseIntegrityError, match="bounded JSON size"):
        verify_release(release)


def test_cli_success_and_failure_are_json(release, capsys):
    assert main(["--release", str(release)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    (release / "report.html").write_text("tampered")
    assert main(["--release", str(release)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_relative_explicit_release_is_accepted(release, monkeypatch):
    monkeypatch.chdir(release.parent)
    assert verify_release(Path(release.name))["status"] == "verified"
