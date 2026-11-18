"""
Indian market calendar (IST): trading sessions, holidays, session state.

Holidays come from the Upstox holiday API (cached for a day); if that fails we
fall back to weekdays only, so nothing hard-fails on a calendar lookup.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from quantos_platform.core import settings

IST = settings.IST
OPEN, CLOSE = "09:15", "15:30"
EOD_READY = "15:45"   # daily candle considered final after this


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def today_ist() -> pd.Timestamp:
    """Today's date in IST as a tz-naive midnight timestamp."""
    return now_ist().normalize().tz_localize(None)


def _cached_json(name: str, fetch, max_age_h: float):
    settings.ensure_dirs()
    path = settings.CACHE_DIR / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_h * 3600:
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    return data


def trading_holidays() -> set[pd.Timestamp]:
    """NSE F&O trading holidays (this year and next)."""
    def fetch():
        from quantos_platform.core.upstox import get

        out = []
        year = today_ist().year
        for y in (year, year + 1):
            try:
                rows = get("/v2/market/holidays").get("data", []) if y == year \
                    else get(f"/v2/market/holidays/{y}-01-01").get("data", [])
            except Exception:
                rows = []
            out += [r["date"] for r in rows
                    if r.get("holiday_type") == "TRADING_HOLIDAY" and "NFO" in r.get("closed_exchanges", [])]
        return sorted(set(out))
    try:
        return set(pd.to_datetime(_cached_json("holidays", fetch, max_age_h=24)))
    except Exception:
        return set()


def is_session(day: pd.Timestamp, holidays: set | None = None) -> bool:
    d = day.tz_localize(None) if getattr(day, "tzinfo", None) else day
    d = d.normalize()
    hol = trading_holidays() if holidays is None else holidays
    return d.dayofweek < 5 and d not in hol


def market_open_now() -> bool:
    now = now_ist()
    return is_session(now) and OPEN <= now.strftime("%H:%M") < CLOSE


def future_sessions(after: pd.Timestamp, until: pd.Timestamp) -> list[pd.Timestamp]:
    """Trading sessions in (after, until]: weekdays minus NSE F&O holidays."""
    hol = trading_holidays()
    return [d for d in pd.bdate_range(after + pd.Timedelta(days=1), until) if d not in hol]


def last_close() -> pd.Timestamp:
    """15:30 IST of the most recent session that has closed (tz-aware)."""
    hol = trading_holidays()
    now = now_ist()
    day = now.normalize()
    if not (is_session(day, hol) and now.strftime("%H:%M") >= CLOSE):
        day -= pd.Timedelta(days=1)
        while not is_session(day, hol):
            day -= pd.Timedelta(days=1)
    return day + pd.Timedelta(hours=15, minutes=30)


def last_completed_session_date():
    """Today once the daily candle is final (15:45 IST), otherwise yesterday (date)."""
    now = now_ist()
    ready = now.normalize() + pd.Timedelta(hours=15, minutes=45)
    return now.date() if now >= ready else (now - pd.Timedelta(days=1)).date()


def expected_latest_session() -> pd.Timestamp:
    """The latest session whose daily data should exist by now (tz-naive date)."""
    hol = trading_holidays()
    now = now_ist()
    day = now.normalize().tz_localize(None)
    if not (is_session(day, hol) and now.strftime("%H:%M") >= EOD_READY):
        day -= pd.Timedelta(days=1)
    while not is_session(day, hol):
        day -= pd.Timedelta(days=1)
    return day


def sessions_to_expiry(day: pd.Timestamp, expiry: pd.Timestamp, known: pd.DatetimeIndex) -> int:
    """Sessions in (day, expiry]; 1 = the next session is expiry day."""
    past = known[(known > day) & (known <= expiry)]
    last_known = known.max()
    extra = future_sessions(max(day, last_known), expiry) if expiry > last_known else []
    return len(past) + len(extra)
