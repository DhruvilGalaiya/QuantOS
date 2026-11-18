"""
Job status store (SQLite): what ran, when, whether it worked, and how fresh the data is.

Pages read this to show "data as of ..." and stale-data warnings; the uptime /
alerting layer (deployment step 5) will read it too.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from quantos_platform.core import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    ok INTEGER,
    message TEXT,
    details TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_runs_job ON job_runs(job, id);
CREATE TABLE IF NOT EXISTS freshness (
    dataset TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    updated TEXT NOT NULL
);
"""


@contextmanager
def _conn():
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.STATUS_DB, timeout=30)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return pd.Timestamp.now(tz=settings.IST).isoformat()


def start_run(job: str) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO job_runs(job, started) VALUES (?, ?)", (job, _now())).lastrowid


def finish_run(run_id: int, ok: bool, message: str = "", details: dict | None = None) -> None:
    with _conn() as c:
        c.execute("UPDATE job_runs SET finished=?, ok=?, message=?, details=? WHERE id=?",
                  (_now(), int(ok), message[:2000], json.dumps(details or {}, default=str), run_id))


def set_freshness(dataset: str, as_of) -> None:
    """Record the market time a dataset now covers (e.g. last session, snapshot time)."""
    with _conn() as c:
        c.execute("INSERT INTO freshness(dataset, as_of, updated) VALUES (?, ?, ?) "
                  "ON CONFLICT(dataset) DO UPDATE SET as_of=excluded.as_of, updated=excluded.updated",
                  (dataset, str(as_of), _now()))


def freshness(dataset: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT as_of, updated FROM freshness WHERE dataset=?", (dataset,)).fetchone()
    return {"as_of": row[0], "updated": row[1]} if row else None


def last_run(job: str, ok_only: bool = False) -> dict | None:
    q = "SELECT job, started, finished, ok, message, details FROM job_runs WHERE job=?"
    q += " AND ok=1" if ok_only else ""
    with _conn() as c:
        row = c.execute(q + " ORDER BY id DESC LIMIT 1", (job,)).fetchone()
    if not row:
        return None
    return {"job": row[0], "started": row[1], "finished": row[2], "ok": None if row[3] is None else bool(row[3]),
            "message": row[4], "details": json.loads(row[5] or "{}")}


def recent_runs(limit: int = 50) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT job, started, finished, ok, message FROM job_runs ORDER BY id DESC LIMIT ?", c, params=(limit,))
