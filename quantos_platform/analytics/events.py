"""
Event Study Lab: what historically happened after a market event. Base rates, not forecasts.

For each event (defined only from data up to and including the event-day close) we measure the index return
from that close to 1/5/21/63 sessions later, compare it with the same-horizon return of *all* days
(the baseline), and attach a bootstrap 95% confidence interval and a bootstrap p-value.

    detect(symbol)          event dates per event id (de-clustered)
    study(symbol)           per-event forward returns + summary stats + average paths
    build(symbol)           writes analytics/events_<symbol>.parquet and events_summary_<symbol>.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quantos_platform.analytics import TRADING_DAYS, load_daily, read_frame, write_frame, write_json

HORIZONS = (1, 5, 21, 63)
PATH_BEFORE, PATH_AFTER = 10, 63
N_BOOT = 4000
SEED = 7
CALENDAR_FILE = Path(__file__).with_name("event_calendar.csv")


@dataclass(frozen=True)
class Event:
    id: str
    name: str
    description: str
    group: str
    rule: Callable[[pd.DataFrame], pd.Series]   # boolean Series on the daily index (uses data <= t only)
    min_gap: int = 21                           # sessions between counted occurrences (de-clustering)


# ============================================================
# RULES (each uses information available at the event-day close)
# ============================================================

def _ret(df):
    return df["close"].pct_change()


def _new_high(df):
    prior = df["close"].shift(1).rolling(TRADING_DAYS, min_periods=TRADING_DAYS).max()
    return df["close"] > prior


def _new_low(df):
    prior = df["close"].shift(1).rolling(TRADING_DAYS, min_periods=TRADING_DAYS).min()
    return df["close"] < prior


def _weekly_streak(df, n: int, up: bool) -> pd.Series:
    """True on the last session of the week that completes exactly n consecutive down (or up) weeks."""
    week = df["close"].groupby(df.index.to_period("W-FRI"))
    last_day = week.apply(lambda s: s.index[-1])
    wret = week.last().pct_change()
    if df.index[-1].weekday() < 4:       # the latest week is still in progress: don't judge it yet
        wret = wret.iloc[:-1]
    sign = (wret > 0) if up else (wret < 0)
    run = sign.groupby((~sign).cumsum()).cumsum()
    hit_weeks = run[run == n].index
    out = pd.Series(False, index=df.index)
    out.loc[last_day.loc[hit_weeks].values] = True
    return out


def _drawdown_cross(df, level: float) -> pd.Series:
    peak = df["close"].cummax()
    dd = df["close"] / peak - 1
    return (dd <= level) & (dd.shift(1) > level)


def _vix_spike(df):
    vix = df.get("vix")
    if vix is None:
        return pd.Series(False, index=df.index)
    return vix.pct_change() >= 0.20


def _vix_high_cross(df):
    vix = df.get("vix")
    if vix is None:
        return pd.Series(False, index=df.index)
    p90 = vix.shift(1).rolling(TRADING_DAYS, min_periods=TRADING_DAYS // 2).quantile(0.90)
    return (vix > p90) & (vix.shift(1) <= p90)


def _down_days(df, n: int = 5):
    neg = (_ret(df) < 0).astype(int)
    return neg.rolling(n).sum() == n


def _state_turn(df, state: str) -> pd.Series:
    """First day of a new market state (walk-forward HMM labels; no future data)."""
    if "state" not in df:
        return pd.Series(False, index=df.index)
    st = df["state"]
    return (st == state) & (st.shift(1) != state) & st.shift(1).notna()


def _calendar(df, kind: str) -> pd.Series:
    if not CALENDAR_FILE.exists():
        return pd.Series(False, index=df.index)
    cal = pd.read_csv(CALENDAR_FILE, parse_dates=["date"])
    dates = set(cal.loc[cal["event"] == kind, "date"].dt.normalize())
    return pd.Series(df.index.isin(dates), index=df.index)


EVENTS: list[Event] = [
    Event("new_52w_high", "New 52-week high", "Close above every close of the previous 252 sessions.",
          "Trend", _new_high),
    Event("new_52w_low", "New 52-week low", "Close below every close of the previous 252 sessions.",
          "Trend", _new_low),
    Event("down_weeks_4", "4 straight down weeks", "The week closes lower for the fourth week in a row.",
          "Streaks", lambda d: _weekly_streak(d, 4, up=False)),
    Event("up_weeks_4", "4 straight up weeks", "The week closes higher for the fourth week in a row.",
          "Streaks", lambda d: _weekly_streak(d, 4, up=True)),
    Event("down_days_5", "5 straight down days", "Five consecutive sessions closing lower.",
          "Streaks", _down_days, min_gap=10),
    Event("big_down_day", "Fall of 2% or more", "The index closes 2% or more below the previous close.",
          "Shocks", lambda d: _ret(d) <= -0.02, min_gap=5),
    Event("big_up_day", "Rise of 2% or more", "The index closes 2% or more above the previous close.",
          "Shocks", lambda d: _ret(d) >= 0.02, min_gap=5),
    Event("gap_down", "Gap down of 1% or more", "The session opens 1% or more below the previous close.",
          "Shocks", lambda d: d["open"] / d["close"].shift(1) - 1 <= -0.01, min_gap=5),
    Event("gap_up", "Gap up of 1% or more", "The session opens 1% or more above the previous close.",
          "Shocks", lambda d: d["open"] / d["close"].shift(1) - 1 >= 0.01, min_gap=5),
    Event("vix_spike", "India VIX jumps 20%+", "India VIX closes 20% or more above its previous close.",
          "Volatility", _vix_spike, min_gap=10),
    Event("vix_above_p90", "India VIX enters its top 10%", "India VIX crosses above its trailing one-year "
          "90th percentile.", "Volatility", _vix_high_cross),
    Event("correction_10", "Falls into a 10% correction", "The close first sits 10% or more below its "
          "all-time high (after being above that level).", "Drawdowns", lambda d: _drawdown_cross(d, -0.10),
          min_gap=63),
    Event("bear_20", "Falls into a 20% bear market", "The close first sits 20% or more below its all-time high.",
          "Drawdowns", lambda d: _drawdown_cross(d, -0.20), min_gap=126),
    Event("turns_stressed", "Market turns Stressed", "The market-state model (see Market States) switches to "
          "its Stressed, high-volatility state. Labels are fitted walk-forward, so no future data is used.",
          "Market state", lambda d: _state_turn(d, "STRESSED")),
    Event("turns_calm", "Market turns Calm", "The market-state model switches to its Calm, low-volatility state.",
          "Market state", lambda d: _state_turn(d, "CALM")),
    Event("budget_day", "Union Budget day", "The session on which the Union Budget is presented "
          "(dates in quantos_platform/analytics/event_calendar.csv).", "Calendar",
          lambda d: _calendar(d, "budget"), min_gap=1),
]
EVENT_BY_ID = {e.id: e for e in EVENTS}


# ============================================================
# MECHANICS
# ============================================================

def decluster(flags: pd.Series, min_gap: int) -> pd.DatetimeIndex:
    """Keep the first event, then skip any event within `min_gap` sessions of the last kept one."""
    pos = np.flatnonzero(flags.fillna(False).to_numpy())
    kept, last = [], -10**9
    for p in pos:
        if p - last >= min_gap:
            kept.append(p)
            last = p
    return flags.index[kept]


def forward_returns(close: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Return from the close of t to the close of t+h (NaN when t+h is not yet known)."""
    return pd.DataFrame({f"fwd_{h}": close.shift(-h) / close - 1 for h in horizons})


def bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_pvalue(x: np.ndarray, baseline: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> float:
    """Two-sided: how often a random sample of len(x) ordinary days has a mean at least as far from
    the baseline mean as the event mean is."""
    x, baseline = x[np.isfinite(x)], baseline[np.isfinite(baseline)]
    if len(x) < 2 or len(baseline) < 10:
        return np.nan
    rng = np.random.default_rng(seed + 1)
    mu = baseline.mean()
    sims = rng.choice(baseline, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float((np.abs(sims - mu) >= abs(x.mean() - mu)).mean())


def _stats(x: np.ndarray, baseline: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(x)
    b = baseline[np.isfinite(baseline)]
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "hit_rate": float((x > 0).mean()),
            "worst": float(x.min()), "best": float(x.max()), "ci_low": lo, "ci_high": hi,
            "baseline_mean": float(b.mean()), "baseline_hit_rate": float((b > 0).mean()),
            "excess": float(x.mean() - b.mean()), "p_value": bootstrap_pvalue(x, baseline)}


def _paths(close: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Matrix (events x offsets) of close[t+k]/close[t]-1 for k in -PATH_BEFORE..PATH_AFTER (NaN if unknown)."""
    c = close.to_numpy()
    idx = close.index.get_indexer(dates)
    offs = np.arange(-PATH_BEFORE, PATH_AFTER + 1)
    out = np.full((len(idx), len(offs)), np.nan)
    for i, p in enumerate(idx):
        for j, k in enumerate(offs):
            if 0 <= p + k < len(c):
                out[i, j] = c[p + k] / c[p] - 1
    return out


def _path_summary(paths: np.ndarray, base_drift: float) -> dict:
    rng = np.random.default_rng(SEED + 2)
    mean, lo, hi = [], [], []
    for j in range(paths.shape[1]):
        col = paths[:, j][np.isfinite(paths[:, j])]
        if len(col) < 2:
            mean.append(np.nan), lo.append(np.nan), hi.append(np.nan)
            continue
        boots = rng.choice(col, size=(1000, len(col)), replace=True).mean(axis=1)
        mean.append(col.mean()), lo.append(np.quantile(boots, 0.025)), hi.append(np.quantile(boots, 0.975))
    offs = list(range(-PATH_BEFORE, PATH_AFTER + 1))
    return {"offsets": offs, "mean": mean, "ci_low": lo, "ci_high": hi,
            "baseline": [base_drift * k if k > 0 else 0.0 for k in offs]}


# ============================================================
# STUDY
# ============================================================

def frame(symbol: str) -> pd.DataFrame:
    df = load_daily(symbol)
    try:
        vix = load_daily("INDIA_VIX")["close"]
        df["vix"] = vix.reindex(df.index)
    except FileNotFoundError:
        pass
    states = read_frame(f"states_{symbol.lower()}_series.parquet")
    if not states.empty and "state" in states:
        df["state"] = states["state"].reindex(df.index)
    return df


def detect(df: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    return {e.id: decluster(e.rule(df).reindex(df.index).fillna(False).astype(bool), e.min_gap) for e in EVENTS}


def study(symbol: str) -> tuple[pd.DataFrame, dict]:
    df = frame(symbol)
    close = df["close"]
    fwd = forward_returns(close)
    base_drift = float(np.log(close).diff().mean())
    found = detect(df)
    rows, summary = [], {}
    for e in EVENTS:
        dates = found[e.id]
        ev = fwd.loc[dates].copy()
        ev["event"], ev["close"] = e.id, close.loc[dates]
        ev["date"] = dates
        rows.append(ev.reset_index(drop=True))
        stats = {f"h{h}": _stats(ev[f"fwd_{h}"].to_numpy(), fwd[f"fwd_{h}"].to_numpy()) for h in HORIZONS}
        summary[e.id] = {
            "id": e.id, "name": e.name, "description": e.description, "group": e.group, "min_gap": e.min_gap,
            "count": len(dates), "last": dates[-1].strftime("%Y-%m-%d") if len(dates) else None,
            "stats": stats, "path": _path_summary(_paths(close, dates), base_drift),
        }
    per_event = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    last = df.index[-1]
    recent_window = df.index[-5:]
    # raw triggers (not de-clustered) so a repeat event this week still shows; stats use the de-clustered sample
    recent = []
    for e in EVENTS:
        raw = e.rule(df).reindex(df.index).fillna(False).astype(bool)
        hits = raw.loc[recent_window]
        if hits.any():
            d = hits[hits].index[-1]
            recent.append({"id": e.id, "name": e.name, "date": d.strftime("%Y-%m-%d"), "today": bool(d == last),
                           "times_this_week": int(hits.sum())})
    meta = {"symbol": symbol, "as_of": last.strftime("%Y-%m-%d"), "since": df.index[0].strftime("%Y-%m-%d"),
            "sessions": len(df), "horizons": list(HORIZONS), "n_boot": N_BOOT, "recent": recent,
            "events": summary}
    return per_event, meta


def build(symbol: str) -> dict:
    per_event, meta = study(symbol)
    write_frame(f"events_{symbol.lower()}.parquet", per_event)
    write_json(f"events_summary_{symbol.lower()}.json", meta)
    return meta
