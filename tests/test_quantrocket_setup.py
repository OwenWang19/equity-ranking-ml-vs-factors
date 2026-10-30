"""Offline safety checks for the optional QuantRocket learning-bundle setup."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call

import pytest

BUNDLE = "usstock-learn-1d"
SCRIPT = Path(__file__).resolve().parents[1] / "infra/quantrocket/research/ingest_learning.py"


@pytest.fixture
def learning_setup(monkeypatch):
    """Import the real script against fake modules, never the installed API."""
    api = Mock(spec=["create_usstock_bundle", "ingest_bundle", "list_bundles"])
    quantrocket = ModuleType("quantrocket")
    quantrocket.__path__ = []
    zipline = ModuleType("quantrocket.zipline")
    for name in ("create_usstock_bundle", "ingest_bundle", "list_bundles"):
        setattr(zipline, name, getattr(api, name))
    quantrocket.zipline = zipline
    monkeypatch.setitem(sys.modules, "quantrocket", quantrocket)
    monkeypatch.setitem(sys.modules, "quantrocket.zipline", zipline)
    return runpy.run_path(str(SCRIPT)), api


def json_messages(output):
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def test_import_does_not_create_or_ingest_a_bundle(learning_setup, capsys):
    _, api = learning_setup
    assert api.mock_calls == []
    assert capsys.readouterr().out == ""


def test_new_bundle_is_created_once_in_learning_mode_and_not_reported_complete(learning_setup, capsys):
    script, api = learning_setup
    created = {"status": "success"}
    submitted = {"status": "success", "message": "Ingestion will run asynchronously"}
    snapshot = {BUNDLE: {"ingestions": []}}
    api.list_bundles.side_effect = [{"other-bundle": {}}, snapshot]
    api.create_usstock_bundle.return_value = created
    api.ingest_bundle.return_value = submitted

    script["main"]()

    assert api.mock_calls == [
        call.list_bundles(),
        call.create_usstock_bundle(BUNDLE, learn=True),
        call.ingest_bundle(BUNDLE),
        call.list_bundles(),
    ]
    output = capsys.readouterr().out
    assert json_messages(output) == [
        {"bundle_created": created},
        {"ingestion_requested": submitted, "completed": False},
        {"bundle_status_snapshot": snapshot},
    ]
    assert "Monitor zipline service logs" in output
    assert "verify this bundle's completion message and data coverage" in output


@pytest.mark.parametrize("ingestions", [[], ["2026-10-30T00:00:00Z"]])
def test_existing_bundle_is_never_overwritten_or_reingested(learning_setup, capsys, ingestions):
    script, api = learning_setup
    api.list_bundles.return_value = {BUNDLE: {"ingestions": ingestions}}

    with pytest.raises(RuntimeError, match="already exists"):
        script["main"]()

    assert api.mock_calls == [call.list_bundles()]
    api.create_usstock_bundle.assert_not_called()
    api.ingest_bundle.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failing_step", ["create_usstock_bundle", "ingest_bundle"])
def test_creation_or_ingestion_error_propagates_without_false_success(learning_setup, capsys, failing_step):
    script, api = learning_setup
    failure = RuntimeError(f"{failing_step} failed")
    api.list_bundles.return_value = {}
    api.create_usstock_bundle.return_value = {"status": "success"}
    getattr(api, failing_step).side_effect = failure

    with pytest.raises(RuntimeError) as caught:
        script["main"]()

    assert caught.value is failure
    expected_calls = [call.list_bundles(), call.create_usstock_bundle(BUNDLE, learn=True)]
    if failing_step == "ingest_bundle":
        expected_calls.append(call.ingest_bundle(BUNDLE))
    else:
        api.ingest_bundle.assert_not_called()
    assert api.mock_calls == expected_calls
    output = capsys.readouterr().out
    expected_messages = [{"bundle_created": {"status": "success"}}] if failing_step == "ingest_bundle" else []
    assert json_messages(output) == expected_messages
    assert "ingestion_requested" not in output
    assert "completed" not in output
    assert "bundle_status_snapshot" not in output
    assert "Monitor" not in output


def test_listing_error_propagates_before_any_mutation(learning_setup, capsys):
    script, api = learning_setup
    failure = ConnectionError("QuantRocket service unavailable")
    api.list_bundles.side_effect = failure

    with pytest.raises(ConnectionError) as caught:
        script["main"]()

    assert caught.value is failure
    assert api.mock_calls == [call.list_bundles()]
    assert capsys.readouterr().out == ""
