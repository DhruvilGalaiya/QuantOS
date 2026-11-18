"""
QuantOS V11 | Causal feature engineering for next-day index forecasting

Every feature on row T uses only information available at the close of
session T (Upstox daily candles). The target on row T is the close-to-close
move from T to the next trading session T+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.core import settings  # noqa: E402

PROJECT_ROOT = settings.ROOT
DAILY_DIR = settings.DATA_DIR / "regime" / "daily"

# Indices with reliable OHLC history (used for bar-shape features).
OHLC_SYMBOLS = ["NIFTY_50", "NIFTY_BANK"]

BREADTH_SYMBOLS = [
    "NIFTY_BANK", "NIFTY_IT", "NIFTY_PHARMA", "NIFTY_AUTO",
    "NIFTY_FIN_SERVICE", "NIFTY_METAL", "NIFTY_FMCG", "NIFTY_REALTY",
    "NIFTY_ENERGY", "NIFTY_PSU_BANK", "NIFTY_MEDIA",
    "NIFTY_MIDCAP_100", "NIFTY_SMALLCAP_100", "NIFTY_NEXT_50",
]


def load_daily(symbol: str) -> pd.DataFrame:
    path = DAILY_DIR / f"{symbol.lower()}.parquet"
    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.index = ts.dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    df.index.name = "date"
    df = df.drop(columns="timestamp").apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _streak(sign: pd.Series) -> pd.Series:
    """Signed count of consecutive up (+) / down (-) closes."""
    out = np.zeros(len(sign))
    s = sign.to_numpy()
    for i in range(len(s)):
        if i and s[i] == s[i - 1] and s[i] != 0:
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    return pd.Series(out, index=sign.index)


def price_block(df: pd.DataFrame, prefix: str, ohlc: bool) -> pd.DataFrame:
    c = df["close"]
    r = np.log(c).diff()
    vol20 = r.rolling(20).std()
    f = pd.DataFrame(index=df.index)
    f[f"{prefix}_r1"] = r
    f[f"{prefix}_r2"] = np.log(c / c.shift(2))
    f[f"{prefix}_r5"] = np.log(c / c.shift(5))
    f[f"{prefix}_r20"] = np.log(c / c.shift(20))
    f[f"{prefix}_z1"] = r / vol20
    f[f"{prefix}_vol20"] = vol20 * np.sqrt(252)
    f[f"{prefix}_vol_ratio"] = r.rolling(5).std() / vol20
    f[f"{prefix}_rsi2"] = _rsi(c, 2)
    f[f"{prefix}_rsi14"] = _rsi(c, 14)
    for n in (20, 50, 200):
        f[f"{prefix}_sma{n}"] = c / c.rolling(n).mean() - 1
    f[f"{prefix}_dd252"] = c / c.rolling(252, min_periods=60).max() - 1
    f[f"{prefix}_streak"] = _streak(np.sign(r).fillna(0))
    if ohlc:
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        f[f"{prefix}_ibs"] = (c - df["low"]) / rng
        f[f"{prefix}_range"] = np.log(df["high"] / df["low"]) / vol20
        f[f"{prefix}_gap"] = np.log(df["open"] / c.shift(1))
        f[f"{prefix}_intraday"] = np.log(c / df["open"])
    return f


def build_feature_frame(target: str = "NIFTY_50") -> pd.DataFrame:
    """Feature matrix + next-day target for one index."""
    base = load_daily(target)
    feats = [price_block(base, "tgt", ohlc=target in OHLC_SYMBOLS)]

    # Cross-index context.
    nifty = load_daily("NIFTY_50") if target != "NIFTY_50" else base
    bank = load_daily("NIFTY_BANK")
    feats.append(price_block(nifty, "nifty", ohlc=True)[["nifty_r1", "nifty_z1", "nifty_ibs", "nifty_sma20"]])
    feats.append(price_block(bank, "bank", ohlc=True)[["bank_r1", "bank_z1", "bank_ibs"]])

    # India VIX: level, change, and implied-vs-realised (variance risk premium proxy).
    vix = load_daily("INDIA_VIX")["close"]
    v = pd.DataFrame(index=vix.index)
    v["vix"] = vix
    v["vix_chg1"] = np.log(vix).diff()
    v["vix_chg5"] = np.log(vix / vix.shift(5))
    v["vix_z252"] = (vix - vix.rolling(252, min_periods=60).mean()) / vix.rolling(252, min_periods=60).std()
    feats.append(v)

    # Breadth / dispersion across sector & size indices.
    rets = pd.concat(
        {s: np.log(load_daily(s)["close"]).diff() for s in BREADTH_SYMBOLS}, axis=1
    )
    b = pd.DataFrame(index=rets.index)
    b["breadth_up"] = (rets > 0).sum(axis=1) / rets.notna().sum(axis=1)
    b["breadth_disp"] = rets.std(axis=1)
    b["breadth_ma5"] = b["breadth_up"].rolling(5).mean()
    b["small_minus_large"] = rets["NIFTY_SMALLCAP_100"] - np.log(nifty["close"]).diff().reindex(rets.index)
    b["mid_minus_large"] = rets["NIFTY_MIDCAP_100"] - np.log(nifty["close"]).diff().reindex(rets.index)
    feats.append(b)

    out = pd.concat(feats, axis=1).reindex(base.index)

    out["vrp"] = out["vix"] / 100 - out["tgt_vol20"]
    out["dow"] = out.index.dayofweek
    nxt = out.index.to_series().shift(-1)
    out["gap_days"] = (nxt - out.index.to_series()).dt.days  # weekend / holiday ahead
    month = out.index.to_period("M")
    out["tom"] = (month != pd.Series(month, index=out.index).shift(-1)).astype(float)

    # Target: next-session close-to-close move (percent).
    close = base["close"]
    out["close"] = close
    out["next_ret_pct"] = (close.shift(-1) / close - 1) * 100
    out["next_date"] = nxt

    return out.replace([np.inf, -np.inf], np.nan)
