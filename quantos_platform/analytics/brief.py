"""
Today brief: one artifact (analytics/today.json) that the Today page renders. Built from the volatility and
event artifacts plus the existing regime, surface and playbook outputs. Missing inputs degrade to None.
"""

from __future__ import annotations

import json

import numpy as np

from quantos_platform.analytics import TRADING_DAYS, load_daily, read_json, write_json
from quantos_platform.core import calendar, settings

BRIEF_SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK"}
HEADLINE_HORIZON = "h21"


def _regime(symbol: str) -> dict:
    path = settings.PREDICTION_DIR / "latest_predictions.json"
    try:
        s = json.loads(path.read_text()).get(symbol, {}).get("summary", {})
    except (OSError, ValueError):
        return {}
    return {"regime": s.get("current_regime"), "streak": s.get("current_regime_streak"),
            "stickiness": s.get("stickiness_level"), "as_of": s.get("latest_date")}


def _implied_move(symbol: str) -> dict:
    """Front-expiry implied move = forward · ATM IV · √T: what option prices currently charge, not a forecast."""
    path = settings.SURFACE_DIR / f"{symbol.lower()}_surface_v2.json"
    try:
        s = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    exps = s.get("expiries") or []
    if not exps:
        return {}
    e = exps[0]
    iv, T, fwd = e.get("atm_iv"), e.get("T"), e.get("forward") or s.get("spot")
    if not (iv and T and fwd):
        return {}
    move = fwd * iv * np.sqrt(T)
    spot = s.get("spot") or fwd
    out = {"expiry": e.get("expiry"), "days": e.get("days"), "atm_iv": iv, "move_points": float(move),
           "move_pct": float(iv * np.sqrt(T)), "snapshot_ts": s.get("snapshot_ts"),
           "rr25": e.get("rr25"), "bf25": e.get("bf25"), "spot": float(spot), "forward": float(fwd),
           "lower": float(spot - move), "upper": float(spot + move)}
    # the options behind the number: the call and put at the listed strike nearest the forward, valued at the
    # front at-the-money implied volatility (Black-76), i.e. roughly what the at-the-money straddle costs now
    strikes = sorted({p["strike"] for p in s.get("points", []) if p.get("expiry") == e.get("expiry")})
    if strikes:
        from scipy.stats import norm

        k = min(strikes, key=lambda x: abs(x - fwd))
        sd = iv * np.sqrt(T)
        d1 = (np.log(fwd / k) + 0.5 * sd ** 2) / sd
        df_ = e.get("discount_factor") or 1.0
        call = df_ * (fwd * norm.cdf(d1) - k * norm.cdf(d1 - sd))
        put = df_ * (k * norm.cdf(sd - d1) - fwd * norm.cdf(-d1))
        out.update(atm_strike=float(k), atm_call=float(call), atm_put=float(put), straddle=float(call + put))
    return out


def _playbook(symbol: str) -> dict:
    from quantos_platform import jobs

    sig = jobs.load_tonight(symbol) or {}
    return {k: sig.get(k) for k in ("phase", "active", "tdte", "expiry", "side", "strike", "next_window", "as_of")}


def _market(symbol: str) -> dict:
    df = load_daily(symbol)
    c = df["close"]
    peak = c.cummax()
    hi52 = c.iloc[-TRADING_DAYS:].max()
    lo52 = c.iloc[-TRADING_DAYS:].min()
    return {"as_of": df.index[-1].strftime("%Y-%m-%d"), "close": float(c.iloc[-1]),
            "change_pct": float(c.iloc[-1] / c.iloc[-2] - 1),
            "week_pct": float(c.iloc[-1] / c.iloc[-6] - 1), "month_pct": float(c.iloc[-1] / c.iloc[-22] - 1),
            "drawdown": float(c.iloc[-1] / peak.iloc[-1] - 1), "ath": float(peak.iloc[-1]),
            "hi52": float(hi52), "lo52": float(lo52),
            "range52_pos": float((c.iloc[-1] - lo52) / (hi52 - lo52)) if hi52 > lo52 else None}


def _events(symbol: str) -> list[dict]:
    meta = read_json(f"events_summary_{symbol.lower()}.json")
    out = []
    for r in meta.get("recent", []):
        e = meta["events"].get(r["id"], {})
        st = e.get("stats", {}).get(HEADLINE_HORIZON, {})
        out.append({**r, "n": st.get("n"), "mean": st.get("mean"), "hit_rate": st.get("hit_rate"),
                    "baseline_mean": st.get("baseline_mean"), "p_value": st.get("p_value"),
                    "ci_low": st.get("ci_low"), "ci_high": st.get("ci_high")})
    return out


def build() -> dict:
    out = {"built": calendar.now_ist().isoformat(), "symbols": {}}
    for sym, name in BRIEF_SYMBOLS.items():
        vol = read_json(f"volatility_{sym.lower()}.json")
        entry = {"name": name, "market": _market(sym), "regime": _regime(sym), "implied": _implied_move(sym),
                 "playbook": _playbook(sym), "events": _events(sym)}
        if vol:
            cone21 = next((c for c in vol.get("cone", []) if c["horizon"] == 21), {})
            entry["vol"] = {"rv21": vol["realized_now"].get("yang_zhang"),
                            "rv21_percentile": cone21.get("percentile"),
                            "garch_21d": vol["garch"].get("vol_21d"),
                            "garch_long_run": vol["garch"].get("long_run_vol"),
                            "hurst": vol["hurst"].get("H")}
            if "vrp" in vol:
                v = vol["vrp"]
                entry["vol"].update({"vix": v["vix_now"], "vix_percentile_1y": v["vix_percentile_1y"],
                                     "vrp_now": v["spread_trailing_now"],
                                     "vrp_percentile": v["spread_trailing_percentile"],
                                     "vrp_share_positive": v["share_implied_above"],
                                     "vrp_vs_garch": v.get("spread_vs_garch_now")})
            elif entry["implied"].get("atm_iv") and vol["garch"].get("vol_21d"):
                entry["vol"]["iv_vs_garch"] = entry["implied"]["atm_iv"] - vol["garch"]["vol_21d"]
        out["symbols"][sym] = entry
    write_json("today.json", out)
    return out
