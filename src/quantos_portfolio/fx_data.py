from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "regime" / "portfolio_fx"

START_DATE = "2016-01-01"
END_DATE = "2026-09-05"

SYMBOL = "USDINR=X"


def download_usdinr():

    print("=" * 70)
    print("QUANTOS USD/INR DATA")
    print("=" * 70)

    df = yf.download(
        SYMBOL,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if df.empty:
        raise RuntimeError("No USD/INR data returned")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise RuntimeError(
            f"Close column missing. Columns: {list(df.columns)}"
        )

    df = df[["Close"]].copy()

    df.index = pd.to_datetime(df.index)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "timestamp"

    df["close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df[["close"]]

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if (df["close"] <= 0).any():
        raise RuntimeError("Invalid USD/INR values detected")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "USDINR.parquet"

    output = df.reset_index()
    output["symbol"] = "USDINR"

    output.to_parquet(
        output_path,
        index=False
    )

    print(f"Rows: {len(output)}")
    print(
        f"Period: "
        f"{output['timestamp'].min().date()} → "
        f"{output['timestamp'].max().date()}"
    )
    print(
        f"Latest USD/INR: "
        f"{output['close'].iloc[-1]:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    download_usdinr()