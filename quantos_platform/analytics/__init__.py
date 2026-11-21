"""
Descriptive analytics ("be the house, not the player"): measurements of the current market state
and historical base rates with uncertainty. Nothing here forecasts direction.

    volatility.py  realized-vol estimators, vol cone, volatility risk premium, GARCH(1,1)-t, rough-vol Hurst,
                   one-session straddle decay by days to expiry
    events.py      event studies: what happened after 52-week highs, losing streaks, VIX spikes, ...
    brief.py       the Today page summary built from the artifacts above

Jobs call build_* functions (python -m quantos_platform.jobs analytics); pages only read the artifacts
in settings.ANALYTICS_DIR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantos_platform.core import settings

INDICES = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK", "SENSEX": "SENSEX"}
TRADING_DAYS = 252


def load_daily(symbol: str) -> pd.DataFrame:
    """Daily OHLC indexed by session date (tz-naive), positive prices only."""
    path = settings.DAILY_DIR / f"{symbol.lower()}.parquet"
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(settings.IST).dt.tz_localize(None)
    df = df.assign(date=ts.dt.normalize()).drop_duplicates("date", keep="last").set_index("date").sort_index()
    df = df[["open", "high", "low", "close"]].astype(float)
    return df[(df > 0).all(axis=1)]


def _clean(obj):
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def write_json(name: str, payload: dict) -> Path:
    settings.ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.ANALYTICS_DIR / name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_clean(payload), indent=1))
    tmp.replace(path)
    return path


def read_json(name: str) -> dict:
    path = settings.ANALYTICS_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def write_frame(name: str, df: pd.DataFrame) -> Path:
    settings.ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.ANALYTICS_DIR / name
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return path


def read_frame(name: str) -> pd.DataFrame:
    path = settings.ANALYTICS_DIR / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
