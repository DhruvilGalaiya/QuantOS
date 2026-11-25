"""
Event Lab page service: what followed market events (52-week highs, losing streaks, VIX spikes, ...) at four
horizons, each against ordinary days with a bootstrap confidence interval and p-value, plus every occurrence.
Reads analytics/events_summary_<sym>.json and events_<sym>.parquet (built by the analytics job).
"""

from __future__ import annotations

import pandas as pd

from quantos_platform.analytics import INDICES, read_frame, read_json

ALPHA = 0.05
MIN_N = 10


def _r(x, d=5):
    return None if x is None or pd.isna(x) else round(float(x), d)


def build(symbol: str) -> dict:
    if symbol not in INDICES:
        raise KeyError(symbol)
    M = read_json(f"events_summary_{symbol.lower()}.json")
    base = {"symbol": symbol, "name": INDICES[symbol], "symbols": [{"symbol": k, "name": v} for k, v in INDICES.items()]}
    if not M:
        return {**base, "as_of": None}
    rows = read_frame(f"events_{symbol.lower()}.parquet")
    events = []
    for e in M["events"].values():
        stats = {}
        for h, s in e["stats"].items():
            s = dict(s)
            s["unusual"] = bool(s.get("p_value") is not None and s["p_value"] < ALPHA and (s.get("n") or 0) >= MIN_N)
            stats[h] = s
        occ = rows[rows["event"] == e["id"]].sort_values("date", ascending=False)
        events.append({
            "id": e["id"], "name": e["name"], "description": e["description"], "group": e["group"], "min_gap": e["min_gap"],
            "count": e["count"], "last": e.get("last"), "stats": stats, "path": e["path"],
            "occurrences": [{"date": pd.Timestamp(r.date).strftime("%Y-%m-%d"), "close": _r(r.close, 2),
                             "h1": _r(r.fwd_1), "h5": _r(r.fwd_5), "h21": _r(r.fwd_21), "h63": _r(r.fwd_63)} for r in occ.itertuples()],
        })
    return {**base, "as_of": M["as_of"], "since": M["since"], "sessions": M["sessions"], "horizons": M["horizons"],
            "n_boot": M["n_boot"], "recent": M.get("recent", []), "events": events, "alpha": ALPHA, "min_n": MIN_N}
