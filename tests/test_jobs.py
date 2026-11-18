"""Job runner: failures are recorded (never raised) and overlapping runs are skipped."""

from quantos_platform import jobs
from quantos_platform.core import status


def test_failed_job_is_recorded_not_raised(isolated_system, monkeypatch):
    def boom(force=False):
        raise RuntimeError("upstream down")
    monkeypatch.setitem(jobs.JOBS, "india", boom)
    res = jobs.run("india")
    assert res["ok"] is False and "upstream down" in res["message"]
    last = status.last_run("india")
    assert last["ok"] is False


def test_successful_job_and_freshness(isolated_system, monkeypatch):
    def fine(force=False):
        status.set_freshness("india_daily", "2026-11-18")
        return {"updated": {}}
    monkeypatch.setitem(jobs.JOBS, "india", fine)
    assert jobs.run("india")["ok"] is True
    assert status.freshness("india_daily")["as_of"] == "2026-11-18"
    assert status.last_run("india", ok_only=True) is not None


def test_overlapping_run_is_skipped(isolated_system, monkeypatch):
    calls = []
    monkeypatch.setitem(jobs.JOBS, "vol", lambda force=False: calls.append(1) or {})
    (isolated_system / "locks" / "vol.lock").write_text("123")
    res = jobs.run("vol")
    assert res.get("skipped") and not calls


def test_public_mode_never_runs_jobs(isolated_system, monkeypatch):
    from quantos_platform.core import settings
    monkeypatch.setattr(settings, "PUBLIC", True)
    monkeypatch.setitem(jobs.JOBS, "india", lambda force=False: (_ for _ in ()).throw(AssertionError("ran")))
    assert jobs.ensure_fresh("india") is None
