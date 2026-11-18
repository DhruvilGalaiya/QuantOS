"""
QuantOS | Volatility Surface Engine v2

From the latest option snapshot (options_live.sqlite) build, per underlying:

  1. Exact time to expiry   T = (expiry 15:30 IST - snapshot time) / 365 days
  2. Forward              put-call parity  F = K + (C - P) / DF  (DF from a fixed 6.5% rate)
  3. Own implied vols       Black-76 on bid / mid / ask prices (European index options)
  4. OTM-only smile         puts below the forward, calls above (ITM quotes are noisy)
  5. SVI fit per expiry     Gatheral's raw SVI in total variance w(k) = sigma^2 * T
  6. SABR fit per expiry    Hagan lognormal (beta = 1), same clean inputs, for comparison
  7. No-arbitrage checks    butterfly (Durrleman g(k) >= 0) and calendar (w rising in T)
  8. Skew metrics           ATM vol, 25-delta risk reversal and butterfly
  9. Smooth surface grid    total variance interpolated linearly in T at fixed log-moneyness

Output: data/regime/volatility_surface/<symbol>_surface_v2.json

    python -m quantos_platform.vol.surface
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from quantos_platform.core import settings
from quantos_platform.core.validation import check_surface
from quantos_platform.vol.pricing import black76, implied_vol  # noqa: F401  (re-exported)
from quantos_platform.vol.smile_models import (  # noqa: F401
    durrleman_g,
    fit_sabr,
    fit_svi,
    sabr_curve,
    svi_derivs,
    svi_w,
)

IST = settings.IST
DB_PATH = settings.OPTIONS_DB
OUT_DIR = settings.SURFACE_DIR

SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK", "SENSEX": "SENSEX"}

RISK_FREE = 0.065            # annual risk-free rate for discounting (~ 91-day T-bill)
TICK = 0.05
MIN_PREMIUM = 0.50           # ignore options quoted below 50 paise (tick noise)
MAX_REL_SPREAD = 0.30        # ignore quotes whose spread is > 30% of mid
MIN_T_YEARS = 4 / (365 * 24) # skip expiries less than ~4 hours away
MAX_ABS_K = 0.12             # |ln(K/F)| kept at all
MAX_STD_DEVS = 4.0           # fit window: |k| <= 4 * ATM vol * sqrt(T)
MIN_POINTS = 8               # two-sided quotes needed to fit a slice
MIN_PARITY = 3               # call/put pairs needed to trust the forward
GRID_K = 61
GRID_T = 30


UNDERLYING_KEYS = {"NIFTY_50": "NSE_INDEX|Nifty 50", "NIFTY_BANK": "NSE_INDEX|Nifty Bank",
                   "SENSEX": "BSE_INDEX|SENSEX"}


# ============================================================
# DATA
# ============================================================

def spot_price(symbol: str, chain: pd.DataFrame) -> float:
    """Live index level from Upstox LTP; falls back to put-call parity on the nearest expiry."""
    try:
        from quantos_platform.core.upstox import get

        key = UNDERLYING_KEYS[symbol]
        data = get("/v3/market-quote/ltp", {"instrument_key": key}).get("data", {})
        for v in data.values():
            if v.get("instrument_token") == key and v.get("last_price"):
                return float(v["last_price"])
    except Exception:
        pass
    near = chain[chain["expiry"] == chain["expiry"].min()]
    wide = near.pivot_table(index="strike", columns="option_type", values="ltp", aggfunc="first").dropna()
    if {"CE", "PE"} <= set(wide.columns) and len(wide):
        est = wide.index.to_numpy() + wide["CE"].to_numpy() - wide["PE"].to_numpy()
        atm = np.argmin(np.abs(wide["CE"].to_numpy() - wide["PE"].to_numpy()))
        return float(np.median(est[max(0, atm - 2): atm + 3]))
    return float("nan")


def load_snapshot() -> tuple[pd.Timestamp, pd.DataFrame]:
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        ts = conn.execute("SELECT MAX(snapshot_ts) FROM option_snapshots").fetchone()[0]
        df = pd.read_sql_query(
            "SELECT symbol, expiry, strike, option_type, ltp, bid, ask, volume, oi "
            "FROM option_snapshots WHERE snapshot_ts = ?", conn, params=(ts,))
    snap = pd.Timestamp(ts)
    snap = snap.tz_localize(IST) if snap.tzinfo is None else snap.tz_convert(IST)
    for c in ("strike", "ltp", "bid", "ask", "volume", "oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return snap, df


def clean_mid(row_bid, row_ask, row_ltp):
    """Mid price from a sane two-sided quote, else last traded price (flagged)."""
    bid, ask, ltp = row_bid, row_ask, row_ltp
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
        if (ask - bid) <= max(MAX_REL_SPREAD * mid, 2 * TICK):
            return mid, bid, ask, "mid"
    if np.isfinite(ltp) and ltp > 0:
        return ltp, np.nan, np.nan, "ltp"
    return np.nan, np.nan, np.nan, None


def forward_and_df(wide: pd.DataFrame, spot: float, T: float) -> tuple[float, float, int]:
    """Forward from put-call parity, F = K + (C - P) / DF, median over near-the-money strikes.

    DF comes from a fixed risk-free rate: for expiries only days away the C - P slope
    cannot pin the rate down (it implied 12-24%), while F is well identified.
    """
    df_ = float(np.exp(-RISK_FREE * T))
    near = wide.dropna(subset=["c_mid", "p_mid"])
    near = near[(near["strike"] / spot - 1).abs() <= 0.02]
    if len(near):
        F = float(np.median(near["strike"] + (near["c_mid"] - near["p_mid"]) / df_))
        if abs(F / spot - 1) < 0.05:
            return F, df_, len(near)
    return float(spot * np.exp(RISK_FREE * T)), df_, 0


def realized_vol_20d(symbol: str) -> float:
    try:
        from quantos_platform.models.features_v11 import load_daily
        c = load_daily(symbol)["close"]
        r = np.log(c).diff().dropna().tail(20)
        return float(r.std() * np.sqrt(252))
    except Exception:
        return float("nan")


# ============================================================
# ONE EXPIRY
# ============================================================

def build_slice(chain: pd.DataFrame, spot: float, T: float) -> dict | None:
    rows = []
    for _, r in chain.iterrows():
        mid, bid, ask, src = clean_mid(r["bid"], r["ask"], r["ltp"])
        rows.append({"strike": r["strike"], "type": r["option_type"], "mid": mid, "bid": bid, "ask": ask,
                     "src": src, "oi": r["oi"], "volume": r["volume"]})
    q = pd.DataFrame(rows)
    wide = q.pivot_table(index="strike", columns="type", values="mid", aggfunc="first").rename(
        columns={"CE": "c_mid", "PE": "p_mid"}).reset_index()
    for col in ("c_mid", "p_mid"):
        if col not in wide:
            wide[col] = np.nan
    # only two-sided quotes inform the forward
    mids = q[q["src"] == "mid"].pivot_table(index="strike", columns="type", values="mid", aggfunc="first")
    mids = mids.rename(columns={"CE": "c_mid", "PE": "p_mid"}).reset_index()
    for col in ("c_mid", "p_mid"):
        if col not in mids:
            mids[col] = np.nan
    F, DF, n_parity = forward_and_df(mids, spot, T)

    pts = []
    for _, r in q.iterrows():
        is_call = r["type"] == "CE"
        K = float(r["strike"])
        k = float(np.log(K / F))
        otm = (is_call and K >= F) or ((not is_call) and K <= F)
        near_atm = abs(k) < 0.005
        if not (otm or near_atm) or abs(k) > MAX_ABS_K:
            continue
        if not (np.isfinite(r["mid"]) and r["mid"] >= MIN_PREMIUM):
            continue
        iv = implied_vol(r["mid"], F, K, T, DF, is_call)
        if not np.isfinite(iv):
            continue
        ivb = implied_vol(r["bid"], F, K, T, DF, is_call) if np.isfinite(r["bid"]) else np.nan
        iva = implied_vol(r["ask"], F, K, T, DF, is_call) if np.isfinite(r["ask"]) else np.nan
        pts.append({"strike": K, "k": k, "type": r["type"], "iv": iv, "iv_bid": ivb, "iv_ask": iva,
                    "mid": float(r["mid"]), "src": r["src"], "oi": float(r["oi"] or 0),
                    "volume": float(r["volume"] or 0)})
    if not pts:
        return None
    p = pd.DataFrame(pts)
    # near the forward both a call and a put may qualify: average them per strike
    agg = p.groupby("strike", as_index=False).agg(
        k=("k", "first"), iv=("iv", "mean"), iv_bid=("iv_bid", "mean"), iv_ask=("iv_ask", "mean"),
        mid=("mid", "mean"), oi=("oi", "sum"), volume=("volume", "sum"),
        type=("type", lambda s: "/".join(sorted(set(s)))), src=("src", lambda s: "mid" if "mid" in set(s) else "ltp"),
    ).sort_values("k").reset_index(drop=True)
    return {"F": F, "DF": DF, "n_parity": n_parity, "points": agg}


def _weights(pts: pd.DataFrame) -> np.ndarray:
    """Tighter bid/ask (in vol) -> more weight; LTP-only quotes get little weight."""
    spread = (pts["iv_ask"] - pts["iv_bid"]).to_numpy()
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 0.03)
    w = 1.0 / np.clip(spread, 0.002, 0.05)
    w = np.where(pts["src"].to_numpy() == "ltp", w * 0.3, w)
    return w / w.mean()


def skew_metrics(params, T) -> dict:
    """ATM vol, 25-delta risk reversal / butterfly from the SVI slice (forward deltas)."""
    def vol(k):
        return float(np.sqrt(max(svi_w(k, *params), 1e-12) / T))

    def call_delta(k):
        w = max(svi_w(k, *params), 1e-12)
        return norm.cdf((-k + 0.5 * w) / np.sqrt(w))

    out = {"atm_iv": vol(0.0)}
    try:
        kc = brentq(lambda k: call_delta(k) - 0.25, 0.0, 1.5)
        kp = brentq(lambda k: call_delta(k) - 0.75, -1.5, 0.0)
        vc, vp = vol(kc), vol(kp)
        out.update({"k25c": kc, "k25p": kp, "iv25c": vc, "iv25p": vp,
                    "rr25": vc - vp, "bf25": 0.5 * (vc + vp) - out["atm_iv"]})
    except ValueError:
        pass
    return out


# ============================================================
# ONE UNDERLYING
# ============================================================

def build_symbol(symbol: str, snap: pd.Timestamp, df: pd.DataFrame) -> dict | None:
    data = df[df["symbol"] == symbol]
    if data.empty:
        return None
    spot = spot_price(symbol, data)
    if not np.isfinite(spot):
        return None

    slices, points, skipped = [], [], []
    for expiry, chain in sorted(data.groupby("expiry"), key=lambda x: x[0]):
        exp_ts = pd.Timestamp(expiry).tz_localize(IST) + pd.Timedelta(hours=15, minutes=30)
        T = (exp_ts - snap).total_seconds() / (365 * 24 * 3600)
        if T < MIN_T_YEARS:
            continue
        s = build_slice(chain, spot, T)
        n_mid = 0 if s is None else int((s["points"]["src"] == "mid").sum())
        if s is None or n_mid < MIN_POINTS or s["n_parity"] < MIN_PARITY:
            skipped.append({"expiry": str(expiry), "days": T * 365,
                            "reason": f"too thin ({n_mid} two-sided quotes, "
                                      f"{0 if s is None else s['n_parity']} put-call pairs)"})
            continue
        pts = s["points"]
        # Fit only two-sided quotes within +/-N standard deviations: stale last-trade
        # prices and paise-priced far wings (pure tick noise) are shown, not fitted.
        atm_guess = float(pts.loc[pts["k"].abs() < 0.01, "iv"].median()) if (pts["k"].abs() < 0.01).any() \
            else float(pts["iv"].median())
        k_limit = max(MAX_STD_DEVS * atm_guess * np.sqrt(T), 0.02)
        pts["used_in_fit"] = (pts["src"] == "mid") & (pts["k"].abs() <= k_limit)
        fit = pts[pts["used_in_fit"]]
        k, iv, w = fit["k"].to_numpy(), fit["iv"].to_numpy(), _weights(fit)
        svi = fit_svi(k, iv, T, w)
        sabr = fit_sabr(s["F"], fit["strike"].to_numpy(), iv, T, w)
        info = {
            "expiry": str(expiry), "T": T, "days": T * 365, "forward": s["F"], "discount_factor": s["DF"],
            "carry": float(np.log(s["F"] / spot) / T),  # annualised forward premium over spot
            "parity_pairs": s["n_parity"], "n_points": int(len(fit)), "n_ltp_only": int(len(pts) - len(fit)),
            "k_min": float(k.min()), "k_max": float(k.max()), "k_limit": float(k_limit),
        }
        if svi is not None:
            svi_iv = np.sqrt(np.maximum(svi_w(k, *svi), 1e-12) / T)
            kg = np.linspace(min(k.min(), -0.02) - 0.02, max(k.max(), 0.02) + 0.02, 201)
            g = durrleman_g(kg, svi)
            info["svi"] = dict(zip(["a", "b", "rho", "m", "sigma"], map(float, svi)))
            info["svi_rmse"] = float(np.sqrt(np.mean((svi_iv - iv) ** 2)))
            info["butterfly_min_g"] = float(g.min())
            info["butterfly_ok"] = bool(g.min() >= -1e-6)
            info.update(skew_metrics(svi, T))
            pts["svi_iv"] = np.sqrt(np.maximum(svi_w(pts["k"].to_numpy(), *svi), 1e-12) / T)
        if sabr is not None:
            sabr_iv = sabr_curve(s["F"], k, T, sabr)
            info["sabr"] = {"alpha": float(sabr[0]), "beta": 1.0, "rho": float(sabr[1]), "nu": float(sabr[2])}
            info["sabr_rmse"] = float(np.sqrt(np.nanmean((sabr_iv - iv) ** 2)))
            pts["sabr_iv"] = sabr_curve(s["F"], pts["k"].to_numpy(), T, sabr)
        slices.append(info)
        for r in pts.to_dict("records"):
            r.update({"expiry": str(expiry), "days": T * 365, "T": T, "moneyness_fwd": float(np.exp(r["k"]))})
            points.append(r)
    if not slices:
        return None

    fitted = [s for s in slices if "svi" in s]
    calendar, grid = [], None
    if fitted:
        # calendar arbitrage: total variance must not fall as T grows (same k)
        for near, far in zip(fitted, fitted[1:]):
            lo, hi = max(near["k_min"], far["k_min"]), min(near["k_max"], far["k_max"])
            kk = np.linspace(lo, hi, 101) if hi > lo else np.array([0.0])
            gap = svi_w(kk, *far["svi"].values()) - svi_w(kk, *near["svi"].values())
            calendar.append({"near": near["expiry"], "far": far["expiry"], "min_gap": float(gap.min()),
                             "ok": bool(gap.min() >= -1e-6)})
        grid = surface_grid(fitted)

    front = fitted[0] if fitted else slices[0]
    return {
        "symbol": symbol, "display_name": SYMBOLS[symbol], "snapshot_ts": snap.isoformat(),
        "built_at": pd.Timestamp.now(tz=IST).isoformat(), "spot": spot,
        "realized_vol_20d": realized_vol_20d(symbol),
        "front": {k: front.get(k) for k in ("expiry", "days", "forward", "atm_iv", "rr25", "bf25")},
        "expiries": slices, "skipped": skipped, "points": points, "calendar": calendar, "grid": grid,
    }


def surface_grid(fitted: list[dict]) -> dict:
    """Smooth surface: total variance linear in T at fixed k, between SVI slices."""
    Ts = np.array([s["T"] for s in fitted])
    lo_s = np.array([s["k_min"] for s in fitted])
    hi_s = np.array([s["k_max"] for s in fitted])
    # common range covered by fitted quotes at every maturity -> no extrapolated wings
    kg = np.linspace(float(lo_s.max()), float(hi_s.min()), GRID_K)
    Tg = np.linspace(Ts.min(), Ts.max(), GRID_T) if len(Ts) > 1 else Ts
    W = np.array([svi_w(kg, *s["svi"].values()) for s in fitted])           # (n_exp, n_k)
    S = None
    if all("sabr" in s for s in fitted):
        S = np.array([sabr_curve(s["forward"], kg, s["T"],
                                 (s["sabr"]["alpha"], s["sabr"]["rho"], s["sabr"]["nu"])) for s in fitted])
    svi_iv, sabr_iv = [], []
    for T in Tg:
        j = int(np.clip(np.searchsorted(Ts, T) - 1, 0, max(len(Ts) - 2, 0)))
        if len(Ts) == 1:
            wT = W[0]
            sT = S[0] if S is not None else None
        else:
            t = (T - Ts[j]) / (Ts[j + 1] - Ts[j])
            wT = W[j] + t * (W[j + 1] - W[j])
            sT = None
            if S is not None:
                wS = (S[j] ** 2) * Ts[j] + t * ((S[j + 1] ** 2) * Ts[j + 1] - (S[j] ** 2) * Ts[j])
                sT = np.sqrt(np.maximum(wS, 1e-12) / T)
        svi_iv.append(np.sqrt(np.maximum(wT, 1e-12) / T).tolist())
        if sT is not None:
            sabr_iv.append(np.asarray(sT, dtype=float).tolist())
    return {"k": kg.tolist(), "moneyness_fwd": np.exp(kg).tolist(), "days": (Tg * 365).tolist(),
            "svi_iv": svi_iv, "sabr_iv": sabr_iv or None}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(symbols=None, log=print) -> dict[str, str]:
    snap, df = load_snapshot()
    errors = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sym in symbols or SYMBOLS:
        try:
            out = build_symbol(sym, snap, df)
            if out is None:
                errors[sym] = "not enough clean quotes"
                continue
            problems = check_surface(out)
            if problems:  # keep the last good surface rather than publish a broken one
                errors[sym] = "validation: " + "; ".join(problems[:3])
                continue
            tmp = OUT_DIR / f".{sym.lower()}_surface_v2.json.tmp"
            tmp.write_text(json.dumps(_clean(out)))
            tmp.replace(OUT_DIR / f"{sym.lower()}_surface_v2.json")
            log(f"{sym}: {len(out['expiries'])} expiries, {len(out['points'])} points")
        except Exception as exc:
            errors[sym] = f"{type(exc).__name__}: {exc}"
    return errors


def load(symbol: str) -> dict | None:
    p = OUT_DIR / f"{symbol.lower()}_surface_v2.json"
    return json.loads(p.read_text()) if p.exists() else None


if __name__ == "__main__":
    errs = run()
    for k, v in errs.items():
        print(f"ERROR {k}: {v}")
