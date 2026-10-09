from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_fx_series(path: str | Path) -> pd.Series:
    """
    Load an FX close series from Parquet.
    Expected columns: timestamp, close
    """

    path = Path(path)

    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    ).dt.tz_localize(None)

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
    )

    series = df.set_index("timestamp")["close"]

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if (series <= 0).any():
        raise ValueError("FX series contains non-positive values.")

    series.name = "USDINR"

    return series


def convert_usd_to_inr(
    usd_prices: pd.DataFrame,
    usdinr: pd.Series,
) -> pd.DataFrame:
    """
    Convert USD-denominated asset prices into INR.

    INR price = USD price × USDINR
    """

    usd_prices = usd_prices.copy()
    usdinr = usdinr.copy()

    usd_prices.index = pd.to_datetime(
        usd_prices.index
    )

    usdinr.index = pd.to_datetime(
        usdinr.index
    )

    # Align FX to US trading dates only.
    aligned_fx = (
    usdinr
    .reindex(usd_prices.index)
    .ffill()
)

    converted = usd_prices.multiply(
        aligned_fx,
        axis=0
    )

    return converted