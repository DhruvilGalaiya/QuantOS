"""
Volatility & Edge page service: realized vs implied volatility, the volatility risk premium (NIFTY 50, from
India VIX), the volatility cone, GARCH(1,1)-t, what an overnight at-the-money straddle seller kept by days to
expiry, and the rough-volatility Hurst estimate. Reads analytics/volatility_<sym>.json, its series parquet and
straddle_decay_<sym>.parquet (built by the analytics job). Series are thinned to keep the payload small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.analytics import INDICES, read_frame, read_json
from quantos_platform.services.common import ordinal

LOT = {"NIFTY_50": 65, "NIFTY_BANK": 30}       # current exchange lot sizes, for rupee figures per lot
MAX_TDTE = 5                                     # longer gaps (holiday weeks) are too rare to measure
HIST_EDGES = list(range(-300, 111, 10))          # seller result per ₹100 of premium, losses beyond -300 grouped


def _r(x, d: int = 5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def _vrp(V: dict, series: pd.DataFrame) -> dict | None:
    v = V.get("vrp")
    if not v or series.empty:
        return None
    s = series.dropna(subset=["vix"]).copy()
    s["gap_avg"] = s["spread_realized"].rolling(63, min_periods=40).mean()
    # daily for the last three years, weekly before that (the long view does not need every session)
    s = pd.concat([s.iloc[:-756].iloc[::5], s.iloc[-756:]])
    pts = [{"time": d.strftime("%Y-%m-%d"), "vix": _r(r.vix), "rv": _r(r.rv_forward), "gap_avg": _r(r.gap_avg)}
           for d, r in s[["vix", "rv_forward", "gap_avg"]].iterrows()]
    return {**{k: v.get(k) for k in ("since", "n_complete", "share_implied_above", "mean_spread", "median_spread",
                                      "worst_spread", "worst_date", "vix_now", "vix_percentile_1y")},
            "premium_now": v.get("spread_trailing_now"), "premium_percentile": v.get("spread_trailing_percentile"),
            "premium_ordinal": ordinal(v.get("spread_trailing_percentile")),
            "by_vix_level": v.get("by_vix_level", []), "yearly": v.get("yearly", []), "series": pts}


def _garch(V: dict, series: pd.DataFrame) -> dict:
    g = dict(V.get("garch") or {})
    hist = series.dropna(subset=["garch_vol"]).iloc[-5 * 252:]
    hist = hist.iloc[::2]  # every other session is plenty for a 5-year line
    g["history"] = [{"time": d.strftime("%Y-%m-%d"), "garch": _r(r.garch_vol), "realized": _r(r.rv21_close)}
                    for d, r in hist[["garch_vol", "rv21_close"]].iterrows()]
    return g


def _decay(symbol: str, decay: pd.DataFrame, V: dict) -> dict | None:
    sd = V.get("straddle_decay") or {}
    if decay.empty or not sd.get("summary"):
        return None
    lot = LOT.get(symbol, 1)
    d = decay.assign(per100=decay["seller_pnl_pct"] * 100, rupees=(decay["entry"] - decay["exit"]) * lot)
    d = d[d["tdte"] <= MAX_TDTE]
    rows = []
    for t, g in d.groupby("tdte"):
        avg, worst = float(g["rupees"].mean()), float(g["rupees"].min())
        verdict = "loss" if avg <= 0 else ("fragile" if abs(worst) > 20 * avg else "steady")
        counts, _ = np.histogram(g["per100"].clip(lower=-299), bins=HIST_EDGES)
        rows.append({"tdte": int(t), "nights": int(len(g)), "win_rate": float((g["per100"] > 0).mean()),
                     "avg_rupees": avg, "median_per100": float(g["per100"].median()), "worst_rupees": worst,
                     "verdict": verdict, "hist": counts.tolist()})
    return {"lot": lot, "n": sd.get("n"), "start": sd.get("start"), "end": sd.get("end"), "edges": HIST_EDGES, "rows": rows}


def _hurst(V: dict) -> dict | None:
    h = V.get("hurst")
    if not h:
        return None
    return {"H": h.get("H"), "zeta": {str(k): v for k, v in (h.get("zeta") or {}).items()}, "rolling": h.get("rolling", []),
            "proxy": h.get("proxy")}


def build(symbol: str) -> dict:
    """Payload for one index. Raises KeyError for an unknown symbol."""
    if symbol not in INDICES:
        raise KeyError(symbol)
    s = symbol.lower()
    V = read_json(f"volatility_{s}.json")
    if not V:
        return {"symbol": symbol, "name": INDICES[symbol], "indices": [{"symbol": k, "name": v} for k, v in INDICES.items()],
                "as_of": None}
    series = read_frame(f"volatility_{s}_series.parquet")
    decay = read_frame(f"straddle_decay_{s}.parquet")
    implied = ((read_json("today.json").get("symbols") or {}).get(symbol) or {}).get("implied") or {}
    yz, yz_p = V["realized_now"].get("yang_zhang"), V["realized_percentile"].get("yang_zhang")
    return {
        "symbol": symbol, "name": INDICES[symbol], "as_of": V.get("as_of"), "close": V.get("close"),
        "indices": [{"symbol": k, "name": v} for k, v in INDICES.items()],
        "realized": {"yang_zhang": yz, "percentile": yz_p, "ordinal": ordinal(yz_p), "estimators": V["realized_now"]},
        "implied": {"atm_iv": implied.get("atm_iv"), "expiry": implied.get("expiry")} if implied.get("atm_iv") else None,
        "vrp": _vrp(V, series),
        "cone": V.get("cone", []),
        "garch": _garch(V, series),
        "decay": _decay(symbol, decay, V),
        "hurst": _hurst(V),
    }
