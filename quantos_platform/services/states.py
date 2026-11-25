"""
Market States page service: the volatility-named state (Calm / Normal / Stressed) from the 3-state HMM, how the
market behaved in each state, how long states last, the state timeline and every major fall since 2007.
Reads analytics/states_<sym>.json and its series parquet (built by the analytics job).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.analytics import read_frame, read_json

SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK"}
STATES = ["CALM", "NORMAL", "STRESSED"]
LABEL = {"CALM": "Calm", "NORMAL": "Normal", "STRESSED": "Stressed"}


def _thin(df: pd.DataFrame, recent: int = 756, step: int = 3) -> pd.DataFrame:
    """Daily for the last three years, every `step` sessions before that."""
    return pd.concat([df.iloc[:-recent].iloc[::step], df.iloc[-recent:]])


def build(symbol: str) -> dict:
    if symbol not in SYMBOLS:
        raise KeyError(symbol)
    S = read_json(f"states_{symbol.lower()}.json")
    base = {"symbol": symbol, "name": SYMBOLS[symbol], "symbols": [{"symbol": k, "name": v} for k, v in SYMBOLS.items()],
            "labels": LABEL, "states": STATES}
    if not S:
        return {**base, "as_of": None}
    ser = read_frame(f"states_{symbol.lower()}_series.parquet")
    sv = _thin(ser.dropna(subset=["state"]))
    dd = _thin(ser.dropna(subset=["drawdown"]))
    cond = S["conditional"]
    behaviour = []
    for s in [*STATES, "ALL"]:
        c = cond[s]
        behaviour.append({"state": s, "share": c.get("share_of_days", 1.0), "typical_day": c["vol_annual"] / np.sqrt(252),
                          "vol_annual": c["vol_annual"], "big_down_share": c["big_down_share"],
                          "next_day_up": c.get("next_day_up"), "fwd21_mean": c["fwd21_mean"], "fwd21_p10": c["fwd21_p10"],
                          "fwd21_p90": c["fwd21_p90"], "fwd21_up": c.get("fwd21_up"), "worst_day": c.get("worst_day")})
    T = S["transition"]
    episodes = []
    for s in STATES:
        e = S["episodes"][s]
        total = sum(e["next"].values())
        episodes.append({"state": s, "n": e["n"], "median_days": e["median_days"], "longest": e["longest"],
                         "next": {k: (v / total if total else None) for k, v in e["next"].items() if k != s}})
    D = S["drawdown"]
    falls = [{"peak": e["peak"], "trough": e["trough"], "depth": e["depth"], "days_down": e["days_down"],
              "days_to_recover": e["days_to_recover"], "ongoing": bool(e["ongoing"])}
             for e in D["episodes"] if e["depth"] <= -0.10]
    falls.sort(key=lambda x: x["peak"], reverse=True)
    return {
        **base, "as_of": S["as_of"], "switch_prob": S.get("switch_prob"),
        "current": S["current"],
        "behaviour": behaviour,
        "transition": {"p": T["p"], "lo": T["lo"], "hi": T["hi"], "duration": T["duration"], "dur_lo": T["dur_lo"], "dur_hi": T["dur_hi"]},
        "episodes": episodes,
        "survival": {s: [round(float(x), 4) for x in S["survival"][s][:120]] for s in STATES},
        "timeline": [{"time": d.strftime("%Y-%m-%d"), "close": round(float(r.close), 2), "state": r.state} for d, r in sv[["close", "state"]].iterrows()],
        "drawdown": {"now": D["now"], "stage": D["stage"], "peak": D["peak"], "base_rates": D["base_rates"], "falls": falls,
                     "series": [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 5)} for d, v in dd["drawdown"].items()]},
    }
