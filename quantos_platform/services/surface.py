"""
Volatility Surface page service: the implied-volatility surface fitted with SVI and SABR across strikes and
expiries, each expiry's smile against market quotes, the at-the-money and skew term structures and the
calibration and no-arbitrage checks. Reads volatility_surface/<sym>_surface_v2.json (built by the vol job).
"""

from __future__ import annotations

import json

import numpy as np

from quantos_platform.core import settings

SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK", "SENSEX": "SENSEX"}
REF_MIN_DAYS = 7          # headline numbers use the first expiry at least a week out (front weeklies are noisy)


def _r(x, d=5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def build(symbol: str) -> dict:
    if symbol not in SYMBOLS:
        raise KeyError(symbol)
    base = {"symbol": symbol, "name": SYMBOLS[symbol], "symbols": [{"symbol": k, "name": v} for k, v in SYMBOLS.items()]}
    p = settings.SURFACE_DIR / f"{symbol.lower()}_surface_v2.json"
    try:
        s = json.loads(p.read_text())
    except (OSError, ValueError):
        return {**base, "snapshot_ts": None}
    exps = sorted(s.get("expiries", []), key=lambda e: e["T"])
    if not exps:
        return {**base, "snapshot_ts": None}
    ref = next((e for e in exps if e["days"] >= REF_MIN_DAYS), exps[-1])
    cal = {(c["near"], c["far"]): c["ok"] for c in s.get("calendar", [])}
    expiries = []
    for i, e in enumerate(exps):
        cal_ok = cal.get((e["expiry"], exps[i + 1]["expiry"])) if i + 1 < len(exps) else None
        expiries.append({k: e.get(k) for k in ("expiry", "days", "T", "forward", "n_points", "atm_iv", "rr25", "bf25", "svi_rmse", "sabr_rmse",
                                               "butterfly_ok", "k_min", "k_max")}
                        | {"calendar_ok": cal_ok, "sabr_rho": (e.get("sabr") or {}).get("rho"), "sabr_nu": (e.get("sabr") or {}).get("nu"),
                           "svi_rho": (e.get("svi") or {}).get("rho")})
    points = [{"expiry": q["expiry"], "strike": q["strike"], "k": _r(q["k"]), "iv": _r(q["iv"]), "iv_bid": _r(q.get("iv_bid")),
               "iv_ask": _r(q.get("iv_ask")), "svi": _r(q.get("svi_iv")), "sabr": _r(q.get("sabr_iv")), "type": q.get("type"),
               "used": bool(q.get("used_in_fit")), "oi": _r(q.get("oi"), 0)} for q in s.get("points", []) if q.get("iv")]
    g = s.get("grid") or {}
    n_bad = sum(not e["butterfly_ok"] for e in exps) + sum(not c["ok"] for c in s.get("calendar", []))
    return {
        **base, "snapshot_ts": s.get("snapshot_ts"), "spot": s.get("spot"), "realized_20d": s.get("realized_vol_20d"),
        "ref": {k: ref.get(k) for k in ("expiry", "days", "forward", "atm_iv", "rr25", "bf25")},
        "arb_warnings": n_bad, "quotes": sum(e["n_points"] for e in exps),
        "expiries": expiries, "points": points,
        "grid": {"k": [_r(x) for x in g.get("k", [])], "moneyness": [_r(x) for x in g.get("moneyness_fwd", [])],
                 "days": [_r(x, 2) for x in g.get("days", [])],
                 "svi": [[_r(v, 4) for v in row] for row in g.get("svi_iv", [])],
                 "sabr": [[_r(v, 4) for v in row] for row in g.get("sabr_iv", [])]},
    }
