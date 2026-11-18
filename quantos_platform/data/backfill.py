"""
QuantOS | Backfill long daily history from Upstox

Extends every daily parquet in data/regime/daily BACKWARDS to START_DATE
(and creates files for instruments that do not have one yet, e.g. the
V11 breadth indices). Existing rows are kept; one row per IST calendar
day; timestamps follow the existing 05:30 IST convention.

    python src/backfill_daily_history.py

Run once. After that, src/live_pipeline.py keeps the files current.
"""

from __future__ import annotations

import pandas as pd

from quantos_platform.data import daily as lp  # noqa: E402

START_DATE = pd.Timestamp("2007-01-01")
CHUNK_YEARS = 5  # Upstox caps a daily-candle request at ~10 years
TIME_OF_DAY = pd.Timedelta(hours=5, minutes=30)


def fetch_range(key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=CHUNK_YEARS) - pd.Timedelta(days=1), end)
        frames.append(lp._fetch_candles(key, cursor.date().isoformat(), chunk_end.date().isoformat()))
        cursor = chunk_end + pd.Timedelta(days=1)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=lp.CANDLE_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = lp._to_ist(df["timestamp"]).dt.normalize() + TIME_OF_DAY
    return df


def backfill(symbol: str, key: str) -> str:
    path = lp.DAILY_DIR / f"{symbol.lower()}.parquet"
    today = pd.Timestamp(lp.last_completed_session_date())

    if path.exists():
        old = pd.read_parquet(path)
        old["timestamp"] = lp._to_ist(old["timestamp"])
        first = old["timestamp"].min().tz_localize(None).normalize()
        if first <= START_DATE + pd.Timedelta(days=10):
            return f"{symbol:<20} already from {first.date()}"
        fresh = fetch_range(key, START_DATE, first - pd.Timedelta(days=1))
    else:
        old = pd.DataFrame(columns=lp.CANDLE_COLUMNS + ["instrument"])
        fresh = fetch_range(key, START_DATE, today)

    fresh["instrument"] = symbol
    for col in old.columns:
        if col not in fresh.columns:
            fresh[col] = pd.NA
    cols = list(old.columns) if len(old.columns) else list(fresh.columns)
    fresh = fresh[cols]

    merged = pd.concat([fresh, old], ignore_index=True) if len(old) else fresh
    merged["timestamp"] = lp._to_ist(merged["timestamp"])
    merged = merged.sort_values("timestamp")
    merged = merged[~merged["timestamp"].dt.date.duplicated(keep="last")].reset_index(drop=True)

    if merged["close"].isna().any():
        raise ValueError(f"{symbol}: missing closes after merge")

    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    return (
        f"{symbol:<20} {len(old):>5} -> {len(merged):>5} rows  "
        f"{merged['timestamp'].min().date()} .. {merged['timestamp'].max().date()}"
    )


def main():
    lp.DAILY_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, key in lp.INSTRUMENTS.items():
        try:
            print(backfill(symbol, key))
        except Exception as exc:
            print(f"{symbol:<20} ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
