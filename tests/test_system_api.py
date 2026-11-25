"""Data status page and the owner's live prices."""

import pytest
from fastapi.testclient import TestClient

from quantos_platform.api import app as app_module
from quantos_platform.core import settings, status
from quantos_platform.services import system, ticks

client = TestClient(app_module.app)


@pytest.fixture
def temp_status(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STATUS_DB", tmp_path / "status.sqlite")
    return tmp_path


def test_system_reports_jobs_runs_and_hides_tracebacks(temp_status):
    rid = status.start_run("india")
    status.finish_run(rid, True, "ok in 9.1s")
    rid = status.start_run("vol")
    status.finish_run(rid, False, "UpstoxError: 429\nTraceback (most recent call last):\n  File \"/home/x.py\"")
    status.set_freshness("india_daily", "2026-11-25")
    out = system.build()
    by_job = {j["job"]: j for j in out["jobs"]}
    assert by_job["india"]["last_ok"] is True and by_job["india"]["last_success"]
    assert by_job["vol"]["last_ok"] is False and by_job["vol"]["last_success"] is None
    assert by_job["vol"]["message"] == "UpstoxError: 429"          # first line only, no file paths
    assert by_job["us_equities"]["last_run"] is None                # never ran: still listed
    assert [r["job"] for r in out["runs"]] == ["vol", "india"]      # newest first
    assert out["runs"][1]["duration_s"] is not None
    assert out["freshness"][0]["dataset"] == "india_daily"


def test_system_route_validates(temp_status):
    r = client.get("/internal/v1/system")
    assert r.status_code == 200 and {"jobs", "runs", "freshness"} <= r.json().keys()


def test_ticks_are_cached_between_calls(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return {"ts": "t", "market_open": True, "quotes": []}

    monkeypatch.setattr(ticks, "_fetch", fake)
    monkeypatch.setattr(ticks, "_cache", {"at": 0.0, "payload": None})
    ticks.quotes()
    ticks.quotes()
    assert len(calls) == 1


def test_ticks_off_in_public_mode(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC", True)
    assert client.get("/internal/v1/ticks").status_code == 403


def test_ticks_provider_failure_is_503(monkeypatch):
    def boom():
        raise RuntimeError("down")

    monkeypatch.setattr(settings, "PUBLIC", False)
    monkeypatch.setattr(ticks, "quotes", boom)
    r = client.get("/internal/v1/ticks")
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
