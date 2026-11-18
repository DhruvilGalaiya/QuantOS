"""
QuantOS | US equity daily closes (Yahoo Finance), incremental and split-aware.

    load_us_asset(symbol)    stored closes (data/regime/portfolio_us/<symbol>.parquet)
    update_us_asset(symbol)  top up to the latest session; full re-download on split/dividend revisions
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from quantos_platform.core import settings

OUTPUT_DIR = settings.US_EQUITY_DIR
START_DATE = "2016-01-01"
END_DATE = (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def download_us_asset(symbol: str) -> pd.DataFrame:

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


def load_us_asset(symbol: str) -> pd.DataFrame:
    """Stored daily closes for one ticker (empty frame if none)."""
    path = OUTPUT_DIR / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "close", "symbol"])
    return pd.read_parquet(path)


def update_us_asset(symbol: str) -> pd.DataFrame:
    """
    Bring one ticker up to the latest completed session.

    New tickers get full history from START_DATE; existing files only
    re-download the last few weeks and merge (Yahoo can revise recent bars).
    """
    symbol = symbol.upper().strip()
    old = load_us_asset(symbol)
    start = START_DATE
    if len(old):
        start = (pd.to_datetime(old["timestamp"]).max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    end = (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, actions=False)
    if raw.empty:
        if len(old):
            return old
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    new = pd.DataFrame({
        "timestamp": pd.to_datetime(raw.index).tz_localize(None).normalize(),
        "close": pd.to_numeric(raw["Close"], errors="coerce").to_numpy(),
    }).dropna()
    new["symbol"] = symbol

    # A split/dividend re-adjusts Yahoo's whole history: if the overlap no
    # longer agrees with what is stored, replace the file with a full download.
    if len(old):
        overlap = old.assign(timestamp=pd.to_datetime(old["timestamp"])).merge(
            new, on="timestamp", suffixes=("_old", "_new"))
        if len(overlap) and ((overlap["close_new"] / overlap["close_old"] - 1).abs() > 0.005).any():
            old = pd.DataFrame(columns=new.columns)
            full = download_us_asset(symbol).reset_index()
            full["symbol"] = symbol
            new = full[["timestamp", "close", "symbol"]]

    merged = pd.concat([old, new]) if len(old) else new
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = (
        merged.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_DIR / f"{symbol}.parquet", index=False)
    return merged
