from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "regime" / "portfolio_us"

US_ASSETS = [
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "PLTR",
    "NFLX",
]

START_DATE = "2016-01-01"
END_DATE = "2026-09-05"


def download_us_asset(symbol: str) -> pd.DataFrame:

    print(f"\nDownloading {symbol}...")

    df = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Handle yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError(
            f"{symbol}: Close column not found. "
            f"Columns: {list(df.columns)}"
        )

    df = df[["Close"]].copy()

    # Normalize index
    df.index = pd.to_datetime(df.index)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "timestamp"

    df["close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df[["close"]]

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if (df["close"] <= 0).any():
        raise ValueError(f"{symbol}: non-positive prices detected")

    return df


def save_us_asset(symbol: str, df: pd.DataFrame):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / f"{symbol}.parquet"

    output = df.reset_index()

    output["symbol"] = symbol

    output.to_parquet(
        path,
        index=False,
    )

    print(
        f"Saved {symbol}: "
        f"{len(output)} rows | "
        f"{output['timestamp'].min().date()} → "
        f"{output['timestamp'].max().date()} | "
        f"{path}"
    )


def main():

    print("=" * 70)
    print("QUANTOS US EQUITY DATA DOWNLOADER")
    print("=" * 70)

    print(f"Assets: {len(US_ASSETS)}")
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Output: {OUTPUT_DIR}")

    results = []

    for symbol in US_ASSETS:

        try:
            df = download_us_asset(symbol)

            save_us_asset(symbol, df)

            results.append(
                {
                    "symbol": symbol,
                    "rows": len(df),
                    "start": df.index.min(),
                    "end": df.index.max(),
                    "status": "OK",
                }
            )

        except Exception as e:

            print(f"ERROR {symbol}: {e}")

            results.append(
                {
                    "symbol": symbol,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "status": f"ERROR: {e}",
                }
            )

    summary = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\nSuccessful assets:", (summary["status"] == "OK").sum())
    print("Failed assets:", (summary["status"] != "OK").sum())


if __name__ == "__main__":
    main()