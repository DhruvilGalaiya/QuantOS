"""
Volatility & the House Edge: measurements, not forecasts of direction.

    realized_vol(df, window, method)   close-to-close, Parkinson, Garman-Klass, Yang-Zhang (annualised)
    vol_cone(close)                    percentiles of realized vol by horizon + today's value
    vrp_history(vix, close)            implied (India VIX) vs the realized vol that followed
    fit_garch(close)                   GARCH(1,1) with Student-t innovations: params + vol term structure
    hurst_log_vol(df)                  rough-volatility scaling of log volatility (Gatheral-Jaisson-Rosenbaum)
    straddle_decay(symbol)             one-session ATM straddle change by sessions to expiry (expired chains)
    build(symbol)                      writes analytics/volatility_<symbol>.json (+ _series parquet)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.analytics import TRADING_DAYS, load_daily, write_frame, write_json
from quantos_platform.core import calendar, settings

CONE_HORIZONS = (10, 21, 63, 126, 252)
CONE_QUANTILES = {"min": 0.0, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "max": 1.0}
METHODS = {"close": "Close-to-close", "parkinson": "Parkinson", "garman_klass": "Garman-Klass",
           "yang_zhang": "Yang-Zhang"}
OPTION_SYMBOLS = {"NIFTY_50": "nifty_50", "NIFTY_BANK": "nifty_bank"}
SESSION_MIN = 375          # 09:15-15:30
ENTRY_TIME = "15:15"


# ============================================================
# REALIZED VOLATILITY ESTIMATORS
# ============================================================

def realized_vol(df: pd.DataFrame, window: int = 21, method: str = "yang_zhang") -> pd.Series:
    """Annualised rolling realized volatility from daily OHLC."""
    o, h, low, c = (np.log(df[k]) for k in ("open", "high", "low", "close"))
    pc = c.shift(1)
    if method == "close":
        return (c - pc).rolling(window).std() * np.sqrt(TRADING_DAYS)
    if method == "parkinson":
        var = ((h - low) ** 2).rolling(window).mean() / (4 * np.log(2))
        return np.sqrt(var * TRADING_DAYS)
    if method == "garman_klass":
        var = (0.5 * (h - low) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2).rolling(window).mean()
        return np.sqrt(var.clip(lower=0) * TRADING_DAYS)
    if method == "yang_zhang":
        overnight = o - pc
        open_close = c - o
        rs = (h - c) * (h - o) + (low - c) * (low - o)
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        var = (overnight.rolling(window).var() + k * open_close.rolling(window).var()
               + (1 - k) * rs.rolling(window).mean())
        return np.sqrt(var.clip(lower=0) * TRADING_DAYS)
    raise ValueError(f"unknown method {method!r}")


def vol_cone(close: pd.Series, horizons=CONE_HORIZONS) -> list[dict]:
    """For each horizon: the distribution of close-to-close realized vol over history, and where today sits."""
    r = np.log(close).diff()
    out = []
    for h in horizons:
        rv = (r.rolling(h).std() * np.sqrt(TRADING_DAYS)).dropna()
        if rv.empty:
            continue
        now = float(rv.iloc[-1])
        row = {"horizon": h, "current": now, "percentile": float((rv <= now).mean())}
        row.update({k: float(rv.quantile(q)) for k, q in CONE_QUANTILES.items()})
        out.append(row)
    return out


# ============================================================
# VOLATILITY RISK PREMIUM (India VIX vs realized that followed)
# ============================================================

def load_vix() -> pd.Series:
    return load_daily("INDIA_VIX")["close"] / 100.0


def forward_realized(close: pd.Series, h: int = 21) -> pd.Series:
    """Close-to-close realized vol over sessions t+1..t+h, aligned at t (known only after t+h)."""
    r = np.log(close).diff()
    return (r.rolling(h).std() * np.sqrt(TRADING_DAYS)).shift(-h)


def vrp_history(vix: pd.Series, close: pd.Series, h: int = 21) -> tuple[pd.DataFrame, dict]:
    """Implied (VIX, 30 calendar days ≈ 21 sessions) vs the realized vol that actually followed."""
    df = pd.DataFrame({"vix": vix, "close": close}).dropna()
    r = np.log(df["close"]).diff()
    df["rv_trailing"] = r.rolling(h).std() * np.sqrt(TRADING_DAYS)
    df["rv_forward"] = forward_realized(df["close"], h)
    df["spread_realized"] = df["vix"] - df["rv_forward"]          # ex-post premium (what sellers earned)
    df["spread_trailing"] = df["vix"] - df["rv_trailing"]         # ex-ante proxy (known today)
    done = df.dropna(subset=["rv_forward"])
    # VIX level terciles (known at t, no look-ahead in the bucket)
    terc = pd.qcut(done["vix"], 3, labels=["Low VIX", "Mid VIX", "High VIX"])
    by_level = []
    for name, g in done.groupby(terc, observed=True):
        by_level.append({"bucket": str(name), "vix_range": [float(g["vix"].min()), float(g["vix"].max())],
                         "n": len(g), "share_implied_above": float((g["spread_realized"] > 0).mean()),
                         "mean_spread": float(g["spread_realized"].mean()),
                         "median_spread": float(g["spread_realized"].median())})
    yearly = (done.groupby(done.index.year)
              .agg(mean_spread=("spread_realized", "mean"),
                   share_implied_above=("spread_realized", lambda s: float((s > 0).mean())),
                   mean_vix=("vix", "mean"), mean_rv=("rv_forward", "mean")))
    hist_tr = df["spread_trailing"].dropna()
    now = df.iloc[-1]
    summary = {
        "since": df.index[0].strftime("%Y-%m-%d"),
        "n_complete": len(done),
        "share_implied_above": float((done["spread_realized"] > 0).mean()),
        "mean_spread": float(done["spread_realized"].mean()),
        "median_spread": float(done["spread_realized"].median()),
        "worst_spread": float(done["spread_realized"].min()),
        "worst_date": done["spread_realized"].idxmin().strftime("%Y-%m-%d"),
        "vix_now": float(now["vix"]),
        "rv_trailing_now": float(now["rv_trailing"]),
        "spread_trailing_now": float(now["spread_trailing"]),
        "spread_trailing_percentile": float((hist_tr <= now["spread_trailing"]).mean()),
        "spread_trailing_z": float((now["spread_trailing"] - hist_tr.mean()) / hist_tr.std()),
        "vix_percentile_1y": float((df["vix"].iloc[-TRADING_DAYS:] <= now["vix"]).mean()),
        "vix_percentile_all": float((df["vix"] <= now["vix"]).mean()),
        "by_vix_level": by_level,
        "yearly": [{"year": int(y), **{k: float(v) for k, v in row.items()}} for y, row in yearly.iterrows()],
    }
    return df, summary


# ============================================================
# GARCH(1,1) STUDENT-t
# ============================================================

def fit_garch(close: pd.Series, years: int = 10, horizon: int = 63) -> tuple[dict, pd.Series]:
    """Fit on the last `years` of daily log returns (in %). Returns summary + conditional vol history."""
    from arch import arch_model

    r = (np.log(close).diff().dropna() * 100).iloc[-years * TRADING_DAYS:]
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    p = res.params
    omega, alpha, beta, nu = (float(p[k]) for k in ("omega", "alpha[1]", "beta[1]", "nu"))
    persistence = alpha + beta
    var_path = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy()
    term = [{"days": h, "vol": float(np.sqrt(var_path[:h].mean() * TRADING_DAYS) / 100)}
            for h in (1, 5, 10, 21, 42, 63) if h <= horizon]
    cond = res.conditional_volatility * np.sqrt(TRADING_DAYS) / 100
    summary = {
        "omega": omega, "alpha": alpha, "beta": beta, "nu": nu, "mu": float(p["mu"]),
        "persistence": persistence,
        "half_life_days": float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else None,
        "long_run_vol": float(np.sqrt(omega / (1 - persistence) * TRADING_DAYS) / 100) if persistence < 1 else None,
        "current_vol": float(cond.iloc[-1]),
        "term_structure": term,
        "vol_21d": next(t["vol"] for t in term if t["days"] == 21),
        "fit_start": r.index[0].strftime("%Y-%m-%d"), "n_obs": len(r),
        "loglik": float(res.loglikelihood),
    }
    return summary, cond


# ============================================================
# ROUGH VOLATILITY: scaling of log volatility
# ============================================================

def hurst_log_vol(df: pd.DataFrame, qs=(0.5, 1.0, 1.5, 2.0, 3.0), lags=range(1, 31), smooth: int = 5) -> dict:
    """m(q, Δ) = E|log σ_{t+Δ} - log σ_t|^q ~ Δ^{ζ_q}, ζ_q ≈ q·H. Volatility proxy: `smooth`-day Parkinson vol.

    Gatheral, Jaisson & Rosenbaum (2018) find H ≈ 0.1 with intraday realized variance. A daily range proxy adds
    measurement noise (which biases H down), so treat the level as indicative and compare across time/indices.
    """
    park = (np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))).rolling(smooth).mean()
    log_vol = 0.5 * np.log(park.replace(0, np.nan)).dropna()
    x = log_vol.to_numpy()
    lags = [d for d in lags if d < len(x) // 4]
    m = {q: [float(np.mean(np.abs(x[d:] - x[:-d]) ** q)) for d in lags] for q in qs}
    log_lag = np.log(lags)
    # use lags beyond the smoothing window, where overlap no longer forces the scaling
    sel = np.array(lags) >= smooth
    zeta = {q: float(np.polyfit(log_lag[sel], np.log(m[q])[sel], 1)[0]) for q in qs}
    H = float(np.polyfit(list(zeta), list(zeta.values()), 1)[0])
    # rolling H over 3-year windows to show stability
    rolling = []
    step, win = TRADING_DAYS // 2, 3 * TRADING_DAYS
    for end in range(win, len(x) + 1, step):
        seg = x[end - win:end]
        lg = [d for d in lags if d >= smooth]
        z1 = np.polyfit(np.log(lg), np.log([np.mean(np.abs(seg[d:] - seg[:-d])) for d in lg]), 1)[0]
        z2 = np.polyfit(np.log(lg), np.log([np.mean(np.abs(seg[d:] - seg[:-d]) ** 2) for d in lg]), 1)[0]
        rolling.append({"end": log_vol.index[end - 1].strftime("%Y-%m-%d"), "H": float(np.polyfit([1, 2], [z1, z2], 1)[0])})
    return {"H": H, "zeta": {str(q): z for q, z in zeta.items()}, "lags": list(lags),
            "m": {str(q): v for q, v in m.items()}, "smooth_days": smooth, "rolling": rolling,
            "proxy": f"{smooth}-day Parkinson volatility"}


# ============================================================
# ONE-SESSION STRADDLE DECAY BY SESSIONS TO EXPIRY (expired chains)
# ============================================================

def _price_at(bars: pd.DataFrame, day: pd.Timestamp, strike: float, typ: str) -> float:
    ts = pd.Timestamp(f"{day:%Y-%m-%d} {ENTRY_TIME}", tz=settings.IST)
    row = bars[(bars["strike"] == strike) & (bars["type"] == typ) & (bars["timestamp"] == ts)]
    return float(row["open"].iloc[0]) if len(row) else np.nan


def straddle_decay(symbol: str, sessions: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Sell the ATM straddle at 15:15, value it at 15:15 next session. One row per entry day.

    The stored chains hold each strike for the entry day and the next session only, so this measures the
    one-session change, not the full path to expiry.
    """
    folder = OPTION_SYMBOLS.get(symbol)
    base = settings.OPTIONS_DIR / folder if folder else None
    if base is None or not (base / "plan.parquet").exists():
        return pd.DataFrame()
    plan = pd.read_parquet(base / "plan.parquet")
    plan = plan[plan["leg"].isin(["atm_ce", "atm_pe"])]
    if sessions is None:
        sessions = load_daily(symbol).index
    rows, cache = [], {}
    for (day, nxt, expiry), g in plan.groupby(["date", "next_session", "expiry"]):
        expiry = pd.Timestamp(expiry)
        f = base / f"{expiry:%Y-%m-%d}.parquet"
        if not f.exists():
            continue
        if f not in cache:
            cache[f] = pd.read_parquet(f)
        bars = cache[f]
        strike = float(g["strike"].iloc[0])
        p0 = _price_at(bars, day, strike, "CE") + _price_at(bars, day, strike, "PE")
        p1 = _price_at(bars, nxt, strike, "CE") + _price_at(bars, nxt, strike, "PE")
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
            continue
        tdte = calendar.sessions_to_expiry(pd.Timestamp(day), expiry, sessions)
        tau0 = tdte + 15 / SESSION_MIN            # option life left at 15:15, in sessions (15 min left today)
        tau1 = tau0 - 1                           # at 15:15 next session
        rows.append({
            "date": pd.Timestamp(day), "next_session": pd.Timestamp(nxt), "expiry": expiry, "tdte": int(tdte),
            "strike": strike, "spot": float(g["spot"].iloc[0]), "entry": p0, "exit": p1,
            "premium_pct_spot": p0 / float(g["spot"].iloc[0]),
            "seller_pnl_pct": (p0 - p1) / p0,                        # share of premium kept by the seller
            "theory_decay_pct": 1 - np.sqrt(max(tau1, 0) / tau0),    # Black-Scholes ATM decay at unchanged IV
        })
    return pd.DataFrame(rows)


def decay_summary(df: pd.DataFrame, max_tdte: int = 5) -> list[dict]:
    if df.empty:
        return []
    out = []
    for t, g in df[df["tdte"] <= max_tdte].groupby("tdte"):
        out.append({"tdte": int(t), "n": len(g),
                    "mean_pnl_pct": float(g["seller_pnl_pct"].mean()),
                    "median_pnl_pct": float(g["seller_pnl_pct"].median()),
                    "win_rate": float((g["seller_pnl_pct"] > 0).mean()),
                    "worst_pnl_pct": float(g["seller_pnl_pct"].min()),
                    "best_pnl_pct": float(g["seller_pnl_pct"].max()),
                    "theory_decay_pct": float(g["theory_decay_pct"].mean()),
                    "premium_pct_spot": float(g["premium_pct_spot"].mean())})
    return out


# ============================================================
# BUILD
# ============================================================

def build(symbol: str) -> dict:
    df = load_daily(symbol)
    close = df["close"]
    windows = {m: realized_vol(df, 21, m) for m in METHODS}
    garch, cond = fit_garch(close)
    payload: dict = {
        "symbol": symbol,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "close": float(close.iloc[-1]),
        "realized_now": {m: float(s.iloc[-1]) for m, s in windows.items()},
        "realized_percentile": {m: float((s.dropna() <= s.iloc[-1]).mean()) for m, s in windows.items()},
        "cone": vol_cone(close),
        "garch": garch,
        "hurst": hurst_log_vol(df),
    }
    series = pd.DataFrame({f"rv21_{m}": s for m, s in windows.items()})
    series["garch_vol"] = cond.reindex(series.index)
    series["close"] = close

    if symbol == "NIFTY_50":   # India VIX is computed from NIFTY options
        vrp_df, vrp = vrp_history(load_vix(), close)
        vrp["spread_vs_garch_now"] = vrp["vix_now"] - garch["vol_21d"]
        payload["vrp"] = vrp
        for col in ("vix", "rv_forward", "spread_realized", "spread_trailing"):
            series[col] = vrp_df[col].reindex(series.index)

    decay = straddle_decay(symbol, df.index)
    payload["straddle_decay"] = {"summary": decay_summary(decay), "n": len(decay),
                                 "start": decay["date"].min().strftime("%Y-%m-%d") if len(decay) else None,
                                 "end": decay["date"].max().strftime("%Y-%m-%d") if len(decay) else None}
    write_json(f"volatility_{symbol.lower()}.json", payload)
    write_frame(f"volatility_{symbol.lower()}_series.parquet", series.iloc[-15 * TRADING_DAYS:])
    if len(decay):
        write_frame(f"straddle_decay_{symbol.lower()}.parquet", decay)
    return payload
