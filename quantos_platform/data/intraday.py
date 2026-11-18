"""
QuantOS | Upstox 15-minute intraday candles

Upstox serves minute candles from January 2022, one calendar month per request.
Files: data/regime/intraday/15m/<symbol>.parquet  (timestamp = bar START, IST).

    python src/intraday_data.py          # backfill / top up all instruments
"""

from __future__ import annotations

import time

import pandas as pd

from quantos_platform.core import settings  # noqa: E402
from quantos_platform.core.instruments import BREADTH_INSTRUMENTS, REGIME_INSTRUMENTS  # noqa: E402
from quantos_platform.core.upstox import get  # noqa: E402

PROJECT_ROOT = settings.ROOT
INTRADAY_DIR = settings.DATA_DIR / "regime" / "intraday" / "15m"
IST = "Asia/Kolkata"
START = pd.Timestamp("2022-01-01")
THROTTLE_S = 0.3

INTRADAY_SYMBOLS = [
    "NIFTY_50", "NIFTY_BANK", "INDIA_VIX",
    "NIFTY_IT", "NIFTY_PHARMA", "NIFTY_AUTO", "NIFTY_FIN_SERVICE",
    "NIFTY_METAL", "NIFTY_FMCG", "NIFTY_REALTY", "NIFTY_ENERGY",
    "NIFTY_PSU_BANK", "NIFTY_MIDCAP_100", "NIFTY_SMALLCAP_100",
]
KEYS = {**REGIME_INSTRUMENTS, **BREADTH_INSTRUMENTS}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


def _parse(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=COLUMNS[:5])
    rows = [(list(c) + [None] * 7)[:7] for c in candles]
    df = pd.DataFrame(rows, columns=COLUMNS)[COLUMNS[:5]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    for col in COLUMNS[1:5]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def fetch_month(key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    resp = get(f"/v3/historical-candle/{key}/minutes/15/{end.date()}/{start.date()}")
    return _parse(resp.get("data", {}).get("candles", []))


def fetch_today(key: str) -> pd.DataFrame:
    """Live session candles (Upstox intraday endpoint)."""
    resp = get(f"/v3/historical-candle/intraday/{key}/minutes/15")
    return _parse(resp.get("data", {}).get("candles", []))


def update_symbol(symbol: str, include_today: bool = True) -> int:
    key = KEYS[symbol]
    path = INTRADAY_DIR / f"{symbol.lower()}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS[:5])
    today = pd.Timestamp.now(tz=IST).normalize().tz_localize(None)

    if len(old):
        last_day = old["timestamp"].max().tz_localize(None).normalize()
        cursor = last_day  # re-fetch the last stored day (may have been partial)
    else:
        cursor = START

    frames = []
    while cursor < today:
        month_end = min((cursor + pd.offsets.MonthEnd(0)), today - pd.Timedelta(days=1))
        frames.append(fetch_month(key, cursor, month_end))
        time.sleep(THROTTLE_S)
        cursor = month_end + pd.Timedelta(days=1)
    if include_today:
        try:
            frames.append(fetch_today(key))
        except Exception:
            pass

    frames = [f for f in frames if len(f)]
    if not frames:
        return 0
    new = pd.concat(frames)
    merged = (
        pd.concat([old, new]) if len(old) else new
    ).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(merged) - len(old)


def load_intraday(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(INTRADAY_DIR / f"{symbol.lower()}.parquet")


def main():
    for symbol in INTRADAY_SYMBOLS:
        t = time.time()
        try:
            added = update_symbol(symbol)
            df = load_intraday(symbol)
            print(f"{symbol:<20} +{added:>6} bars  total {len(df):>6}  "
                  f"{df['timestamp'].min()} .. {df['timestamp'].max()}  ({time.time()-t:.0f}s)")
        except Exception as exc:
            print(f"{symbol:<20} ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
