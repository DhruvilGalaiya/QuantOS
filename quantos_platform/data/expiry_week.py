"""
QuantOS | Expiry-week option chains for the strategy builder (15-minute candles)

For every expired contract cycle, fetch a strike grid wide enough to price any strike from at-the-money
to ~2% out of the money, at any 15-minute bar of the last WEEK_SESSIONS sessions up to expiry:
    calls: lowest index price of the week x (1 - 0.5%)  ..  highest x (1 + 2.2%)
    puts:  lowest x (1 - 2.2%)  ..  highest x (1 + 0.5%)
Files: data/derivatives/expiry_week/<sym>/<expiry>.parquet (one per expiry, resumable).

    python -m quantos_platform.data.expiry_week            # both indices, only missing expiries
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quantos_platform.core import settings  # noqa: E402
from quantos_platform.data.derivatives import UNDERLYINGS, _get, candles, expiries  # noqa: E402
from quantos_platform.data.intraday import load_intraday  # noqa: E402

OUT_DIR = settings.DATA_DIR / "derivatives" / "expiry_week"
WEEK_SESSIONS = 5
NEAR, FAR = 0.005, 0.022


def _sessions(symbol: str) -> pd.DataFrame:
    b = load_intraday(symbol)
    b["day"] = b["timestamp"].dt.tz_localize(None).dt.normalize()
    return b.groupby("day").agg(low=("low", "min"), high=("high", "max"))


def grid(lo: float, hi: float, step: int) -> list[tuple[float, str]]:
    def rng(a, b):
        return np.arange(np.floor(a / step) * step, np.ceil(b / step) * step + step / 2, step)
    return ([(float(k), "CE") for k in rng(lo * (1 - NEAR), hi * (1 + FAR))]
            + [(float(k), "PE") for k in rng(lo * (1 - FAR), hi * (1 + NEAR))])


def fetch(symbol: str, limit: int | None = None) -> None:
    key, step = UNDERLYINGS[symbol]["key"], UNDERLYINGS[symbol]["step"]
    out = OUT_DIR / symbol.lower()
    out.mkdir(parents=True, exist_ok=True)
    days = _sessions(symbol)
    done = 0
    for exp in sorted(expiries(symbol)):
        path = out / f"{exp}.parquet"
        e = pd.Timestamp(exp)
        week = days[days.index <= e].tail(WEEK_SESSIONS)
        if path.exists() or len(week) < WEEK_SESSIONS or week.index[-1] != e:
            continue
        contracts = pd.DataFrame(_get("/v2/expired-instruments/option/contract",
                                      {"instrument_key": key, "expiry_date": exp})["data"])
        if contracts.empty:
            continue
        contracts["strike"] = contracts["strike_price"].astype(float)
        frames = []
        for strike, typ in grid(week["low"].min(), week["high"].max(), step):
            m = contracts[(contracts["strike"] == strike) & (contracts["instrument_type"] == typ)]
            if m.empty:
                continue
            df = candles(m.iloc[0]["instrument_key"], str(week.index[0].date()), exp)
            if len(df):
                df["strike"], df["type"], df["lot_size"] = strike, typ, int(m.iloc[0]["lot_size"])
                frames.append(df)
        if frames:
            pd.concat(frames).assign(expiry=e).to_parquet(path, index=False)
        print(f"{symbol} {exp}: {len(frames)} contracts", flush=True)
        done += 1
        if limit and done >= limit:
            return


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for sym in UNDERLYINGS:
        fetch(sym, lim)
