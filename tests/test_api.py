"""Page services, the internal FastAPI service and the static export."""

import json

import pytest
from fastapi.testclient import TestClient

from quantos_platform.api import schemas
from quantos_platform.api.app import app
from quantos_platform.services import common, today
from tests.conftest import has_local_data

client = TestClient(app)


def test_level_word_and_ordinal():
    assert common.level_word(None) is None
    assert [common.level_word(p) for p in (0.05, 0.2, 0.5, 0.8, 0.95)] == \
        ["very low", "low", "normal", "high", "very high"]
    assert common.ordinal(0.81) == "81st percentile"
    assert common.ordinal(0.12) == "12th percentile"
    assert common.ordinal(0.02) == "2nd percentile"
    assert common.ordinal(0.23) == "23rd percentile"


def test_unusual_rule():
    syms = {"X": {"name": "X", "events": [
        {"id": "a", "name": "A", "date": "2026-01-02", "p_value": 0.01, "n": 30},
        {"id": "b", "name": "B", "date": "2026-01-03", "p_value": 0.01, "n": 5},      # too few cases
        {"id": "c", "name": "C", "date": "2026-01-01", "p_value": 0.40, "n": 300},    # like a normal day
    ]}}
    rows = today._events(syms)
    assert [r["id"] for r in rows] == ["b", "a", "c"]                                 # newest first
    assert {r["id"]: r["unusual"] for r in rows} == {"a": True, "b": False, "c": False}


def test_today_empty_without_artifacts(tmp_path, monkeypatch, isolated_system):
    from quantos_platform.core import settings
    monkeypatch.setattr(settings, "ANALYTICS_DIR", tmp_path / "none")
    payload = schemas.Today.model_validate(today.build())
    assert payload.indices == [] and payload.readings == [] and payload.events == []


def test_healthz_and_meta(isolated_system):
    assert client.get("/internal/healthz").json() == {"ok": True}
    m = client.get("/internal/v1/meta").json()
    assert m["version"] == "1" and isinstance(m["market_open"], bool) and m["freshness"] == []


def test_openapi_has_page_routes():
    spec = client.get("/internal/openapi.json").json()
    assert {"/internal/v1/today", "/internal/v1/meta"} <= set(spec["paths"])
    assert "Today" in spec["components"]["schemas"]


@pytest.mark.slow
@pytest.mark.skipif(not has_local_data(), reason="no local data directory")
def test_today_endpoint_and_etag():
    r = client.get("/internal/v1/today")
    assert r.status_code == 200
    body = r.json()
    assert {i["symbol"] for i in body["indices"]} >= {"NIFTY_50", "NIFTY_BANK"}
    for i in body["indices"]:
        assert i["history"] and i["history"][-1]["time"] <= i["market"]["as_of"]
        assert i["market"]["close"] == pytest.approx(i["history"][-1]["value"], rel=1e-6)
    again = client.get("/internal/v1/today", headers={"If-None-Match": r.headers["etag"]})
    assert again.status_code == 304


@pytest.mark.slow
@pytest.mark.skipif(not has_local_data(), reason="no local data directory")
def test_static_export_matches_api(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("export_static", Path(__file__).parents[1] / "scripts" / "export_static.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = {p.name for p in mod.export(tmp_path)}
    assert {"today.json", "meta.json", "us.json", "NIFTY_50.json", "AAPL.json"} <= files
    exported = json.loads((tmp_path / "today.json").read_text())
    assert exported == client.get("/internal/v1/today").json()
