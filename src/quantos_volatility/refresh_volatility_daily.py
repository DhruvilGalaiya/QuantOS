
"""
QuantOS | Volatility Daily Data Refresh
---------------------------------------
Refreshes ONLY the three volatility underlyings:
    NIFTY 50, NIFTY BANK, SENSEX

Uses Upstox Historical Candle V3.

The volatility pipeline is intentionally separate from the HMM data pipeline,
so changing this file cannot accidentally alter the regime engine.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

OUTPUT_DIR = ROOT / "data" / "regime" / "volatility_daily"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Current completed market date requested for the volatility baseline.
FROM_DATE = "2022-01-01"
TO_DATE = "2026-09-04"

INSTRUMENTS = {
    "NIFTY_50": "NSE_INDEX|Nifty 50",
    "NIFTY_BANK": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
}


def get_token() -> str:
    load_dotenv(ENV_FILE)

    token = os.getenv("UPSTOX_ANALYTICS_TOKEN")

    if not token:
        raise RuntimeError(
            "UPSTOX_ANALYTICS_TOKEN not found in .env"
        )

    return token


def fetch_daily(instrument_key: str) -> pd.DataFrame:
    token = get_token()

    url = (
        "https://api.upstox.com/v3/historical-candle/"
        f"{instrument_key}/days/1/{TO_DATE}/{FROM_DATE}"
    )

    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    payload = response.json()

    candles = (
        payload.get("data", {})
        .get("candles", [])
    )

    if not candles:
        raise RuntimeError(
            f"No daily candles returned for "
            f"{instrument_key}"
        )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def main():
    print("=" * 70)
    print("QuantOS VOLATILITY DAILY DATA REFRESH")
    print("=" * 70)
    print(f"Requested range: {FROM_DATE} -> {TO_DATE}")
    print("Instruments: NIFTY 50, NIFTY BANK, SENSEX")

    success = 0

    for symbol, key in INSTRUMENTS.items():
        print("\n" + "-" * 70)
        print(symbol)
        print("-" * 70)

        try:
            df = fetch_daily(key)

            output = OUTPUT_DIR / (
                f"{symbol.lower()}.parquet"
            )

            df.to_parquet(
                output,
                index=False,
            )

            success += 1

            print(f"Rows:        {len(df):,}")
            print(
                f"Period:      "
                f"{df['timestamp'].min()} -> "
                f"{df['timestamp'].max()}"
            )
            print(
                f"Latest close: "
                f"{df['close'].iloc[-1]:,.2f}"
            )
            print(f"Saved:       {output}")

        except Exception as exc:
            print(
                f"ERROR: {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 70)
    print("DAILY DATA REFRESH COMPLETE")
    print("=" * 70)
    print(
        f"Successful symbols: "
        f"{success}/{len(INSTRUMENTS)}"
    )


if __name__ == "__main__":
    main()
