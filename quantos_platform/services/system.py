"""System status: when each job last ran, whether it worked, and how fresh each dataset is (the Data status page)."""

from __future__ import annotations

import pandas as pd

from quantos_platform.core import calendar, settings, status
from quantos_platform.services.common import freshness

# what each job does and when the cloud schedule runs it (.github/workflows/daily-data.yml), in plain words
JOBS = {
    "india": ("Index prices and regime models", "Weekdays, 18:00 IST"),
    "vol": ("Option chain and volatility surfaces", "Weekdays, 14:45 IST"),
    "analytics": ("Volatility, events, states, risk and the Today brief", "After each India or option run"),
    "us_equities": ("US stock prices (fundamentals weekly)", "Tuesday to Saturday, 07:00 IST"),
}
# local research jobs: named in the run history, but not scheduled in the cloud, so not listed as datasets
LOCAL_ONLY = {"playbook": "Expiry playbook (local research)"}


def _label(job: str) -> str:
    return JOBS[job][0] if job in JOBS else LOCAL_ONLY.get(job, job)


def _duration(started, finished) -> float | None:
    if not started or not finished:
        return None
    return round((pd.Timestamp(finished) - pd.Timestamp(started)).total_seconds(), 1)


def _message(msg: str | None) -> str:
    # first line only: tracebacks and file paths stay in the logs, not on a public page
    return (msg or "").splitlines()[0][:160] if msg else ""


def build(limit: int = 30) -> dict:
    runs = status.recent_runs(limit) if settings.STATUS_DB.exists() else pd.DataFrame()
    recent = [
        {"job": r.job, "label": _label(r.job), "started": r.started, "finished": r.finished,
         "ok": None if pd.isna(r.ok) else bool(r.ok), "duration_s": _duration(r.started, r.finished),
         "message": _message(r.message)}
        for r in runs.itertuples()
    ]
    jobs = []
    for key, (label, schedule) in JOBS.items():
        last = status.last_run(key) if settings.STATUS_DB.exists() else None
        good = status.last_run(key, ok_only=True) if settings.STATUS_DB.exists() else None
        jobs.append({"job": key, "label": label, "schedule": schedule,
                     "last_run": last["started"] if last else None,
                     "last_ok": None if not last else last["ok"],
                     "last_success": good["started"] if good else None,
                     "message": _message(last["message"]) if last else ""})
    fresh = [f for f in freshness() if f["dataset"] not in LOCAL_ONLY]
    return {"now": calendar.now_ist().isoformat(), "jobs": jobs, "freshness": fresh, "runs": recent}
