"""
Market Regimes page service: the three-state HMM trend regime for one instrument, how sticky it is, how regimes
hand over to each other, every tracked instrument at a glance, and the next-session forecast with its
out-of-sample record. Reads live_predictions/latest_predictions.json, the V11 metrics/history and daily closes.

India VIX regimes read as volatility levels: the model's Bear, Sideways and Bull become Low, Mid and High.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantos_platform.analytics import load_daily
from quantos_platform.core import settings

INSTRUMENTS = {
    "NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK", "SENSEX": "SENSEX", "NIFTY_IT": "NIFTY IT",
    "NIFTY_PHARMA": "NIFTY PHARMA", "NIFTY_AUTO": "NIFTY AUTO", "NIFTY_FIN_SERVICE": "NIFTY FIN SERVICE",
    "INDIA_VIX": "INDIA VIX",
}
V11_SYMBOLS = ("NIFTY_50", "NIFTY_BANK")
MARKET = ["BULL", "SIDE", "BEAR"]
VIX_MAP = {"BEAR": "LOW", "SIDE": "MID", "BULL": "HIGH"}
LABEL = {"BULL": "Bull", "SIDE": "Sideways", "BEAR": "Bear", "LOW": "Low vol", "MID": "Mid vol", "HIGH": "High vol"}
BAND = 0.25          # V11: Bull above +0.25%, Bear below -0.25%, otherwise Sideways
LOG_ROWS = 250       # prediction log rows sent to the page (newest first)


def _latest() -> dict:
    try:
        return json.loads((settings.PREDICTION_DIR / "latest_predictions.json").read_text())
    except (OSError, ValueError):
        return {}


def _disp(symbol: str, regime: str | None) -> str | None:
    if regime is None:
        return None
    r = str(regime).upper()
    return VIX_MAP.get(r, r) if symbol == "INDIA_VIX" else r


def states_for(symbol: str) -> list[str]:
    return ["LOW", "MID", "HIGH"] if symbol == "INDIA_VIX" else MARKET


def _closes(symbol: str) -> pd.Series:
    try:
        return load_daily(symbol)["close"]
    except (OSError, KeyError, ValueError):
        return pd.Series(dtype=float)


def _v11_metrics(symbol: str) -> dict:
    p = settings.V11_DIR / f"metrics_v11_{symbol.lower()}.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _matrix(symbol: str, d: dict) -> list[list[float]] | None:
    tm = d.get("transition_matrix")
    if not tm:
        return None
    m = pd.DataFrame(tm["data"], index=tm["index"], columns=tm["columns"]).reindex(index=MARKET, columns=MARKET)
    if symbol == "INDIA_VIX":
        order = ["BEAR", "SIDE", "BULL"]
        m = m.loc[order, order]
    return [[float(x) for x in row] for row in m.to_numpy()]


def _probs(symbol: str, probs: dict | None) -> dict:
    probs = probs or {}
    return {_disp(symbol, k): float(v) for k, v in probs.items()}


def _forecast(symbol: str, d: dict) -> dict | None:
    m = _v11_metrics(symbol).get("latest") if symbol in V11_SYMBOLS else None
    if m:
        return {"model": "V11", "target_date": m["target_trading_date"], "status": str(m.get("status", "")).title(),
                "pick": m["next_regime"], "confidence": m["next_day_probability"],
                "probs": {"BULL": m["bull_probability"], "SIDE": m["side_probability"], "BEAR": m["bear_probability"]}}
    if d.get("next_regime"):
        return {"model": "V10", "target_date": None, "status": "Trend model", "pick": _disp(symbol, d["next_regime"]),
                "confidence": d.get("next_day_probability"), "probs": _probs(symbol, d.get("probabilities"))}
    return None


def _scorecard(symbol: str) -> dict | None:
    """Out-of-sample record of the next-session forecast (V11 walk-forward test set; V10 from its log)."""
    met = _v11_metrics(symbol)
    test = met.get("test")
    if isinstance(test, dict) and test.get("n"):
        return {"model": "V11", "start": test.get("start"), "end": test.get("end"), "n": test["n"],
                "direction_accuracy": test.get("direction_accuracy"),
                "three_class_accuracy": test.get("three_class_accuracy"),
                "always_up": test.get("always_up_accuracy"), "walk_forward": True}
    if symbol == "INDIA_VIX":
        return None
    log = _log_frame(symbol, "V10")
    log = log.dropna(subset=["move"])
    if log.empty:
        return None
    calls = log[log["next_regime"].isin(["BULL", "BEAR"])]
    hits = ((calls["next_regime"] == "BULL") & (calls["move"] > 0)) | ((calls["next_regime"] == "BEAR") & (calls["move"] < 0))
    actual3 = np.where(log["move"] > BAND, "BULL", np.where(log["move"] < -BAND, "BEAR", "SIDE"))
    return {"model": "V10", "start": str(log["prediction_date"].min().date()), "end": str(log["prediction_date"].max().date()),
            "n": int(len(log)), "direction_accuracy": float(hits.mean()) if len(calls) else None,
            "three_class_accuracy": float((log["next_regime"].to_numpy() == actual3).mean()),
            "always_up": float((log["move"] > 0).mean()), "walk_forward": False}


def _log_frame(symbol: str, model: str) -> pd.DataFrame:
    if model == "V11":
        p = settings.V11_DIR / f"prediction_history_v11_{symbol.lower()}.csv"
    else:
        p = settings.PREDICTION_DIR / f"prediction_history_{symbol.lower()}.csv"
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_csv(p, parse_dates=["prediction_date", "target_trading_date"])
    c = _closes(symbol)
    moves = c.pct_change() * 100
    h["move"] = h["target_trading_date"].dt.normalize().map(moves)
    return h.sort_values("prediction_date")


def _log(symbol: str) -> dict:
    model = "V11" if symbol in V11_SYMBOLS else "V10"
    h = _log_frame(symbol, model)
    if h.empty:
        return {"model": model, "rows": []}
    rows = []
    for r in h.iloc[::-1].head(LOG_ROWS).itertuples():
        mv = None if pd.isna(r.move) else float(r.move)
        pick = _disp(symbol, r.next_regime)
        hit = None
        if mv is not None and pick in ("BULL", "BEAR"):
            hit = (pick == "BULL" and mv > 0) or (pick == "BEAR" and mv < 0)
        rows.append({"date": r.prediction_date.strftime("%Y-%m-%d"), "target": r.target_trading_date.strftime("%Y-%m-%d"),
                     "regime_then": _disp(symbol, r.current_regime), "forecast": pick,
                     "confidence": float(r.next_day_probability), "move": mv, "hit": hit})
    return {"model": model, "rows": rows}


def _timeline(symbol: str, d: dict) -> list[dict]:
    c = _closes(symbol)
    out = []
    for r in d.get("daily_output", []):
        day = pd.Timestamp(r["date"])
        close = c.get(day)
        if close is None or pd.isna(close):
            continue
        out.append({"time": r["date"], "close": round(float(close), 2), "regime": _disp(symbol, r["current_regime"])})
    return out


def _overview(latest: dict) -> list[dict]:
    rows = []
    for sym, name in INSTRUMENTS.items():
        d = latest.get(sym)
        if not d:
            continue
        c = _closes(sym)
        chg = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else None
        f = _forecast(sym, d) or {}
        rows.append({"symbol": sym, "name": name, "close": float(c.iloc[-1]) if len(c) else d.get("latest_close"),
                     "change_pct": chg, "regime": _disp(sym, d.get("current_regime")), "streak": d.get("current_regime_streak"),
                     "persistence": d.get("stickiness_score"), "next": f.get("pick"), "next_model": f.get("model")})
    return rows


def build(symbol: str) -> dict:
    """Payload for one instrument. Raises KeyError for an unknown symbol."""
    if symbol not in INSTRUMENTS:
        raise KeyError(symbol)
    latest = _latest()
    d = latest.get(symbol) or {}
    states = states_for(symbol)
    matrix = _matrix(symbol, d)
    stationary = _probs(symbol, d.get("stationary_distribution"))
    dynamics = []
    if matrix:
        for i, s in enumerate(states):
            stay = matrix[i][i]
            exits = {states[j]: matrix[i][j] for j in range(3) if j != i}
            to = max(exits, key=exits.get) if sum(exits.values()) > 0 else None
            dynamics.append({"regime": s, "label": LABEL[s], "stay": stay, "lasts": 1 / max(1 - stay, 1e-6),
                             "exits_to": to, "exits_share": exits[to] / sum(exits.values()) if to else None,
                             "long_run": stationary.get(s)})
    cur = _disp(symbol, d.get("current_regime"))
    c = _closes(symbol)
    chg = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else None
    return {
        "symbol": symbol, "name": INSTRUMENTS[symbol], "is_vix": symbol == "INDIA_VIX", "states": states,
        "labels": {s: LABEL[s] for s in states},
        "instruments": [{"symbol": k, "name": v} for k, v in INSTRUMENTS.items() if k in latest],
        "as_of": d.get("latest_date"),
        "current": {"regime": cur, "label": LABEL.get(cur or "", cur), "streak": d.get("current_regime_streak"),
                    "persistence": d.get("stickiness_score"), "persistence_level": d.get("stickiness_level"),
                    "stay_probability": d.get("stay_probability"), "consistency": d.get("recent_consistency"),
                    "close": d.get("latest_close"), "change_pct": chg} if d else None,
        "forecast": _forecast(symbol, d) if d else None,
        "scorecard": _scorecard(symbol),
        "matrix": matrix,
        "dynamics": dynamics,
        "timeline": _timeline(symbol, d),
        "overview": _overview(latest),
        "log": _log(symbol),
    }
