"""Live index prices for the owner's ticker strip (local or signed-in owner only; the public site is end-of-day).

The one place outside the jobs that calls a data provider: one Upstox LTP request for three instruments, cached
for a few seconds so any number of viewers costs at most one call per interval.
"""

from __future__ import annotations

import threading
import time

from quantos_platform.core import calendar

SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK", "INDIA_VIX": "India VIX"}
TTL_OPEN_S = 2.0
TTL_CLOSED_S = 300.0

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "payload": None}


def _fetch() -> dict:
    from quantos_platform.data.daily import fetch_live_quotes

    market_open = calendar.market_open_now()
    raw = fetch_live_quotes(list(SYMBOLS))
    quotes = []
    for sym, name in SYMBOLS.items():
        q = raw.get(sym)
        if not q:
            continue
        prev = q.get("prev_close")
        quotes.append({"symbol": sym, "name": name, "ltp": q["ltp"], "prev_close": prev,
                       "change_pct": (q["ltp"] / prev - 1) if prev else None})
    return {"ts": calendar.now_ist().isoformat(), "market_open": market_open, "quotes": quotes}


def quotes() -> dict:
    """Latest prices, at most one provider call per TTL (shorter while NSE is open)."""
    with _lock:
        p = _cache["payload"]
        ttl = TTL_OPEN_S if p and p["market_open"] else TTL_CLOSED_S
        if p is None or time.monotonic() - _cache["at"] > ttl:
            _cache["payload"] = _fetch()
            _cache["at"] = time.monotonic()
        return _cache["payload"]
