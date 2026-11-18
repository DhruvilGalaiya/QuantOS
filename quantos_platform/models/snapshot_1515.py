"""
QuantOS V12 | 15:15 IST snapshot features (Stage 1: pre-close forecaster)

For every session T we take the market AS IT STANDS AT 15:15 IST
(close of the 15:00-15:15 bar) and build features that use nothing after
that instant. Daily indicators (EMA, SMA, RSI, Bollinger, Supertrend,
Donchian, ...) are evaluated on true daily bars up to T-1 plus a partial
bar for T = (open, high-so-far, low-so-far, price@15:15).

Targets (log %):
    y_gap  : price@15:15(T) -> official open(T+1)      (exit 9:15)
    y_c2c  : price@15:15(T) -> official close(T+1)     (hold the day)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.data.intraday import load_intraday  # noqa: E402
from quantos_platform.models.features_v11 import load_daily  # noqa: E402
from quantos_platform.models.features_v11 import load_daily as _load_daily  # noqa: E402

SNAP = "15:00"  # bar START; its close is the price at 15:15
SECTORS = [
    "NIFTY_IT", "NIFTY_PHARMA", "NIFTY_AUTO", "NIFTY_FIN_SERVICE", "NIFTY_METAL",
    "NIFTY_FMCG", "NIFTY_REALTY", "NIFTY_ENERGY", "NIFTY_PSU_BANK",
    "NIFTY_MIDCAP_100", "NIFTY_SMALLCAP_100",
]


# ============================================================
# INTRADAY STATE AT 15:15
# ============================================================

def _today_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)


def _bars(symbol: str, partial_today: bool = False) -> pd.DataFrame:
    b = load_intraday(symbol).copy()
    b["date"] = b["timestamp"].dt.tz_localize(None).dt.normalize()
    b["hm"] = b["timestamp"].dt.strftime("%H:%M")
    b = b[b["hm"] <= SNAP]  # nothing after the 15:00-15:15 bar
    keep = b.groupby("date")["hm"].transform("max") == SNAP
    if partial_today:
        # live use: today's session so far (latest bar stands in for 15:15)
        keep |= b["date"] == _today_ist()
    return b[keep]


def _price_at(b: pd.DataFrame, bar_start: str) -> pd.Series:
    """Price at bar_start + 15 min (close of that bar)."""
    return b[b["hm"] == bar_start].set_index("date")["close"]


def intraday_state(symbol: str, partial_today: bool = False) -> pd.DataFrame:
    b = _bars(symbol, partial_today)
    g = b.groupby("date")
    s = pd.DataFrame(index=g.size().index)
    s["p1515"] = g["close"].last()  # == close of the 15:00 bar on full days
    s["open"] = g["open"].first()
    s["high_sofar"] = g["high"].max()
    s["low_sofar"] = g["low"].min()
    s["p1115"] = _price_at(b, "11:00")
    s["p1315"] = _price_at(b, "13:00")
    s["p1415"] = _price_at(b, "14:00")
    s["twap"] = g["close"].mean()
    b = b.assign(r=np.log(b["close"]).groupby(b["date"]).diff())
    s["ivol"] = b.groupby("date")["r"].std() * np.sqrt(25)
    s["r_series"] = b.groupby("date")["r"].apply(lambda x: x.dropna().to_numpy())
    return s


# ============================================================
# DAILY INDICATORS WITH A PARTIAL BAR FOR TODAY
# ============================================================

def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _supertrend_states(d: pd.DataFrame, n: int = 10, m: float = 3.0):
    """True-bar Supertrend states (final bands, trend, ATR) through each day."""
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    mid = (d["high"] + d["low"]) / 2
    bu, bl = (mid + m * atr).to_numpy(), (mid - m * atr).to_numpy()
    c = d["close"].to_numpy()
    fu, fl, tr_dir = bu.copy(), bl.copy(), np.ones(len(d))
    for i in range(1, len(d)):
        if np.isnan(bu[i]) or np.isnan(fu[i - 1]):
            continue
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if tr_dir[i - 1] < 0 and c[i] > fu[i]:
            tr_dir[i] = 1
        elif tr_dir[i - 1] > 0 and c[i] < fl[i]:
            tr_dir[i] = -1
        else:
            tr_dir[i] = tr_dir[i - 1]
    return pd.DataFrame({"fu": fu, "fl": fl, "dir": tr_dir, "atr": atr.to_numpy()}, index=d.index)


def partial_bar_indicators(daily: pd.DataFrame, snap: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    daily : true daily OHLC (index = date)
    snap  : per-date p1515/open/high_sofar/low_sofar (dates subset of daily)
    Returns indicators for each snap date using bars < T plus the partial bar T.
    """
    d = daily.copy()
    c = d["close"]
    prev = d.shift(1)  # state through T-1, aligned on T
    out = pd.DataFrame(index=snap.index)
    p = snap["p1515"]
    P = lambda s: s.reindex(snap.index)  # noqa: E731

    pc = P(prev["close"])
    out[f"{prefix}_r_day"] = np.log(p / pc)  # prev close -> 15:15

    # EMAs (one recursive step from yesterday's EMA)
    for n in (9, 21, 50):
        a = 2 / (n + 1)
        ema_prev = P(c.ewm(span=n, adjust=False).mean().shift(1))
        ema_now = a * p + (1 - a) * ema_prev
        out[f"{prefix}_ema{n}_dist"] = p / ema_now - 1
        out[f"{prefix}_ema{n}_slope"] = ema_now / ema_prev - 1
    out[f"{prefix}_ema9_21"] = out[f"{prefix}_ema9_dist"] - out[f"{prefix}_ema21_dist"]

    # SMAs / Bollinger / z-score (window = 19 true closes + today's price)
    for n in (20, 50, 200):
        s_prev = P(c.rolling(n - 1).sum().shift(1))
        out[f"{prefix}_sma{n}_dist"] = p / ((s_prev + p) / n) - 1
    sum19 = P(c.rolling(19).sum().shift(1))
    sq19 = P((c ** 2).rolling(19).sum().shift(1))
    mean20 = (sum19 + p) / 20
    std20 = np.sqrt(np.maximum((sq19 + p ** 2) / 20 - mean20 ** 2, 1e-12))
    out[f"{prefix}_z20"] = (p - mean20) / std20
    out[f"{prefix}_bb_pctb"] = (p - (mean20 - 2 * std20)) / (4 * std20)

    # Wilder RSI(2), RSI(14): one step from yesterday's averages
    delta = c.diff()
    for n in (2, 14):
        ag = P(_wilder(delta.clip(lower=0), n).shift(1))
        al = P(_wilder((-delta).clip(lower=0), n).shift(1))
        dd = p - pc
        ag_now = (ag * (n - 1) + dd.clip(lower=0)) / n
        al_now = (al * (n - 1) + (-dd).clip(lower=0)) / n
        out[f"{prefix}_rsi{n}"] = 100 - 100 / (1 + ag_now / al_now.replace(0, np.nan))

    # Supertrend(10,3) with the partial bar
    st = _supertrend_states(d).shift(1).reindex(snap.index)
    hi, lo = snap["high_sofar"], snap["low_sofar"]
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = (st["atr"] * 9 + tr) / 10
    mid = (hi + lo) / 2
    bu, bl = mid + 3 * atr, mid - 3 * atr
    fu = np.where((bu < st["fu"]) | (pc > st["fu"]), bu, st["fu"])
    fl = np.where((bl > st["fl"]) | (pc < st["fl"]), bl, st["fl"])
    sdir = np.where((st["dir"] < 0) & (p > fu), 1, np.where((st["dir"] > 0) & (p < fl), -1, st["dir"]))
    out[f"{prefix}_st_dir"] = sdir
    out[f"{prefix}_st_dist"] = np.where(sdir > 0, p / fl - 1, p / fu - 1)
    out[f"{prefix}_st_flip"] = (sdir != st["dir"]).astype(float)

    # Donchian 20 breakout, drawdown, ADX-lite (trend strength)
    hh = P(d["high"].rolling(20).max().shift(1))
    ll = P(d["low"].rolling(20).min().shift(1))
    out[f"{prefix}_donch"] = (p - ll) / (hh - ll) * 2 - 1
    out[f"{prefix}_dd252"] = p / P(d["high"].rolling(252, min_periods=60).max().shift(1)) - 1
    out[f"{prefix}_trend_eff"] = (np.log(p / P(c.shift(10)))).abs() / P(np.log(c).diff().abs().rolling(10).sum())

    # Daily returns history & streak
    out[f"{prefix}_r_prev"] = P(np.log(c).diff())  # yesterday's full-day move
    out[f"{prefix}_vol20"] = P(np.log(c).diff().rolling(20).std() * np.sqrt(252))
    return out


# ============================================================
# FULL SNAPSHOT FRAME
# ============================================================

def daily_state(symbol: str) -> pd.DataFrame:
    """Daily-bar stand-in for intraday_state (close plays the 15:15 price).

    Used to train on 2010+ history where no intraday candles exist.
    """
    d = load_daily(symbol)
    s = pd.DataFrame(index=d.index)
    s["p1515"] = d["close"]
    s["open"] = d["open"]
    s["high_sofar"] = d["high"]
    s["low_sofar"] = d["low"]
    for col in ("p1115", "p1315", "p1415", "twap", "ivol"):
        s[col] = np.nan
    s["r_series"] = None
    return s


def build_snapshot_frame(target: str = "NIFTY_50", asia: pd.DataFrame | None = None,
                         us: pd.DataFrame | None = None, source: str = "intraday",
                         partial_today: bool = False) -> pd.DataFrame:
    """partial_today=True (intraday source only) adds a row for the live session,
    using bars so far and a provisional daily bar; its targets are NaN."""
    live = partial_today and source == "intraday"
    _memo: dict[str, pd.DataFrame] = {}

    def state(symbol: str) -> pd.DataFrame:
        if symbol not in _memo:
            _memo[symbol] = intraday_state(symbol, live) if source == "intraday" else daily_state(symbol)
        return _memo[symbol]

    def load_daily(symbol: str) -> pd.DataFrame:  # noqa: F811 - shadows module import on purpose
        d = _load_daily(symbol)
        if live:
            s = state(symbol)
            today = s.index.max()
            if pd.notna(today) and today > d.index.max():
                row = pd.DataFrame({"open": s.at[today, "open"], "high": s.at[today, "high_sofar"],
                                    "low": s.at[today, "low_sofar"], "close": s.at[today, "p1515"]},
                                   index=pd.DatetimeIndex([today], name=d.index.name))
                d = pd.concat([d, row])
        return d

    daily = load_daily(target)
    st = state(target)
    st = st[st.index.isin(daily.index)]
    f = pd.DataFrame(index=st.index)
    p = st["p1515"]
    pc = daily["close"].shift(1).reindex(st.index)

    # Intraday path
    f["i_day"] = np.log(p / pc)
    f["i_gap_today"] = np.log(st["open"] / pc)
    f["i_open_to_now"] = np.log(p / st["open"])
    f["i_morning"] = np.log(st["p1115"] / st["open"])
    f["i_last2h"] = np.log(p / st["p1315"])
    f["i_last1h"] = np.log(p / st["p1415"])
    f["i_range_pos"] = (p - st["low_sofar"]) / (st["high_sofar"] - st["low_sofar"]).replace(0, np.nan)
    f["i_twap_dist"] = p / st["twap"] - 1
    f["i_ivol"] = st["ivol"]
    gap_up = st["open"] > pc
    f["i_gap_filled"] = np.where(gap_up, st["low_sofar"] <= pc, st["high_sofar"] >= pc).astype(float)

    # Daily indicators with a partial bar
    f = f.join(partial_bar_indicators(daily, st, "d"))
    f["i_day_z"] = f["i_day"] / (f["d_vol20"] / np.sqrt(252))

    # India VIX at 15:15 and its relation to price
    vix_d = load_daily("INDIA_VIX")
    vst = state("INDIA_VIX")
    vix1515 = vst["p1515"].reindex(f.index)
    vpc = vix_d["close"].shift(1).reindex(f.index)
    f["vix"] = vix1515
    f["vix_day"] = np.log(vix1515 / vpc)
    f["vix_open_to_now"] = np.log(vix1515 / vst["open"].reindex(f.index))
    f["vix_z252"] = (vix1515 - vix_d["close"].rolling(252, min_periods=60).mean().shift(1).reindex(f.index)) / \
        vix_d["close"].rolling(252, min_periods=60).std().shift(1).reindex(f.index)
    # intraday NIFTY-VIX correlation of 15-minute returns, and 20-day daily correlation
    rr = st["r_series"].reindex(f.index)
    vr = vst["r_series"].reindex(f.index)
    f["vix_corr_intraday"] = [
        np.corrcoef(a[-min(len(a), len(b)):], b[-min(len(a), len(b)):])[0, 1]
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and min(len(a), len(b)) > 5 else np.nan
        for a, b in zip(rr, vr)
    ]
    dr = np.log(daily["close"]).diff()
    dv = np.log(vix_d["close"]).diff()
    f["vix_corr20"] = dr.rolling(20).corr(dv).shift(1).reindex(f.index)
    f["vix_price_same_dir"] = (np.sign(f["vix_day"]) == np.sign(f["i_day"])).astype(float)
    f["vrp"] = f["vix"] / 100 - f["d_vol20"]

    # Breadth at 15:15
    sec = {}
    for s in SECTORS:
        sd = load_daily(s)["close"].shift(1)
        ss = state(s)["p1515"]
        sec[s] = np.log(ss / sd.reindex(ss.index)).reindex(f.index)
    sec = pd.DataFrame(sec)
    f["b_up"] = (sec > 0).sum(axis=1) / sec.notna().sum(axis=1)
    f["b_disp"] = sec.std(axis=1)
    f["b_small_large"] = sec["NIFTY_SMALLCAP_100"] - f["i_day"]
    other = "NIFTY_BANK" if target == "NIFTY_50" else "NIFTY_50"
    od = load_daily(other)["close"].shift(1)
    os_ = state(other)["p1515"]
    f["x_other_day"] = np.log(os_ / od.reindex(os_.index)).reindex(f.index)

    # Calendar: day-of-week, weekend ahead, expiry day (Thu until Aug-2025, Tue after)
    idx = f.index.to_series()
    f["dow"] = idx.dt.dayofweek
    nxt_session = daily.index.to_series().shift(-1).reindex(f.index)
    f["days_to_next"] = (nxt_session - idx).dt.days
    expiry_dow = np.where(idx < "2025-09-01", 3, 1)
    f["expiry_day"] = (idx.dt.dayofweek == expiry_dow).astype(float)

    # Asia (closed before 15:15 IST) and the US session of T-1 (already known)
    for name, frame in (("asia", asia), ("usprev", us)):
        if frame is None:
            continue
        for col in frame.columns:
            r = np.log(frame[col].dropna()).diff()
            if name == "usprev":
                # last US session dated strictly before T (closed ~02:30 IST on T)
                pos = np.searchsorted(r.index.values, f.index.values, side="left") - 1
                vals = r.to_numpy()
                f[f"usprev_{col}"] = np.where(pos >= 1, vals[np.clip(pos, 0, None)], np.nan)
            else:
                # same calendar date; Nikkei/Hang Seng close before 15:15 IST
                f[f"asia_{col}"] = r.reindex(f.index).fillna(0.0)

    # Targets
    nxt_open = daily["open"].shift(-1).reindex(f.index)
    nxt_close = daily["close"].shift(-1).reindex(f.index)
    f["p1515"] = p
    f["y_gap"] = np.log(nxt_open / p) * 100
    f["y_c2c"] = np.log(nxt_close / p) * 100
    f["y_last15"] = np.log(daily["close"].reindex(f.index) / p) * 100
    f["next_date"] = nxt_session
    return f.replace([np.inf, -np.inf], np.nan)
