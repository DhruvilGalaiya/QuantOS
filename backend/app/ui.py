"""
Shared dashboard helpers: data-freshness display and the local/public refresh policy.

    local mode  (default)  stale data is refreshed when a page opens; refresh buttons shown
    public mode (QUANTOS_MODE=public)  pages only read; scheduled jobs keep data current
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantos_platform import jobs
from quantos_platform.core import calendar, settings, status

PUBLIC = settings.PUBLIC


@st.cache_data(ttl=300, show_spinner=False)
def refresh_on_launch(job: str) -> dict | None:
    """Local mode: run `job` if its data is stale (cached 5 min). Public mode: no-op."""
    return jobs.ensure_fresh(job)


def force_refresh(job: str) -> dict:
    refresh_on_launch.clear()
    return jobs.run(job, force=True)


def _fmt(ts: str | None) -> str:
    if not ts:
        return "never"
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None and len(ts) <= 10:
            return t.strftime("%d %b %Y")
        return t.tz_convert(settings.IST).strftime("%d %b %Y, %H:%M IST") if t.tzinfo else t.strftime("%d %b %Y")
    except Exception:
        return str(ts)


def freshness_caption(dataset: str, label: str = "Data") -> str:
    f = status.freshness(dataset)
    return f"{label} as of **{_fmt(f['as_of']) if f else 'unknown'}**"


def job_health(job: str, dataset: str | None = None) -> None:
    """Warn when the last run of `job` failed, or daily data is behind the market calendar."""
    last = status.last_run(job)
    if last and last.get("ok") is False:
        ok = status.last_run(job, ok_only=True)
        st.warning(f"The last **{job}** update failed ({last['message'][:160]}). "
                   f"Showing data from the last successful run ({_fmt(ok['finished']) if ok else 'none'}).",
                   icon=":material/warning:")
    if dataset == "india_daily":
        f = status.freshness(dataset)
        try:
            behind = f and pd.Timestamp(f["as_of"]) < calendar.expected_latest_session()
        except Exception:
            behind = False
        if behind:
            st.info(f"Daily data is behind the market calendar (latest stored session {_fmt(f['as_of'])}). "
                    "It will catch up on the next scheduled update.", icon=":material/schedule:")


def refresh_button(job: str, label: str = "Pull latest data", key: str | None = None) -> bool:
    """Shown only in local mode. Returns True when a refresh ran (caller should clear caches)."""
    if PUBLIC:
        return False
    if st.button(label, width="stretch", key=key or f"refresh_{job}"):
        with st.spinner(f"Running {job} update…"):
            res = force_refresh(job)
        (st.toast if res.get("ok") else st.error)(f"{job}: {res.get('message')}")
        return True
    return False
