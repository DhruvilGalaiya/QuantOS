from pathlib import Path

import pandas as pd

from src.quantos_portfolio.currency import (
    load_fx_series,
    convert_usd_to_inr,
)


ROOT = Path(__file__).resolve().parents[2]

USD_DATA = (
    ROOT
    / "data"
    / "regime"
    / "portfolio_us"
    / "AAPL.parquet"
)

FX_DATA = (
    ROOT
    / "data"
    / "regime"
    / "portfolio_fx"
    / "USDINR.parquet"
)


# Load AAPL
aapl = pd.read_parquet(USD_DATA)

aapl["timestamp"] = pd.to_datetime(
    aapl["timestamp"]
)

aapl_prices = (
    aapl
    .set_index("timestamp")[["close"]]
    .rename(columns={"close": "AAPL"})
)


# Load USDINR
usdinr = load_fx_series(FX_DATA)


# Convert
aapl_inr = convert_usd_to_inr(
    aapl_prices,
    usdinr,
)


print("=" * 70)
print("QUANTOS CURRENCY NORMALIZATION TEST")
print("=" * 70)

print("\nUSD AAPL:")
print(aapl_prices.tail())

print("\nUSD/INR:")
print(usdinr.tail())

print("\nAAPL INR:")
print(aapl_inr.tail())

print("\nRows:", len(aapl_inr))
print("Missing:", aapl_inr["AAPL"].isna().sum())