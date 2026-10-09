from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.quantos_portfolio.currency import (
    load_fx_series,
    convert_usd_to_inr,
)


ROOT = Path(__file__).resolve().parents[2]

INDIA_DIR = ROOT / "data" / "regime" / "daily"
US_DIR = ROOT / "data" / "regime" / "portfolio_us"
FX_PATH = (
    ROOT
    / "data"
    / "regime"
    / "portfolio_fx"
    / "USDINR.parquet"
)


INDIAN_ASSETS = [
    "NIFTY_50",
    "NIFTY_BANK",
    "NIFTY_MIDCAP_100",
    "NIFTY_NEXT_50",
]


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


def normalize_timestamp_index(values):
    """
    Normalize daily market timestamps.

    Daily market data should have exactly one timestamp per
    calendar trading day. Different providers may represent
    the same daily observation as 00:00, 09:15, or another
    timezone-aware timestamp.

    We remove timezone information and normalize the timestamp
    to the calendar date so that:

        2016-01-04 00:00
        2016-01-04 09:15

    both become:

        2016-01-04 00:00
    """

    converted = pd.to_datetime(values)

    if isinstance(converted, pd.Series):

        if isinstance(
            converted.dtype,
            pd.DatetimeTZDtype,
        ):
            converted = converted.dt.tz_localize(None)

        return converted.dt.normalize()

    if isinstance(converted, pd.DatetimeIndex):

        if converted.tz is not None:
            converted = converted.tz_localize(None)

        return converted.normalize()

    if getattr(converted, "tz", None) is not None:
        converted = converted.tz_localize(None)

    return converted.normalize()

def load_indian_price(symbol: str) -> pd.Series:
    """
    Load one Indian index close-price series.
    """

    path = INDIA_DIR / f"{symbol.lower()}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Indian data not found: {path}"
        )

    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{symbol}: timestamp column missing"
        )

    if "close" not in df.columns:
        raise ValueError(
            f"{symbol}: close column missing"
        )

    df["timestamp"] = normalize_timestamp_index(
        df["timestamp"]
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
    )

    series = df.set_index("timestamp")["close"]

    series.name = symbol

    return series


def load_us_prices() -> pd.DataFrame:
    """
    Load all US equity prices in USD.
    """

    series_list = []

    for symbol in US_ASSETS:

        path = US_DIR / f"{symbol}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"US data not found: {path}"
            )

        df = pd.read_parquet(path)

        if "timestamp" not in df.columns:
            raise ValueError(
                f"{symbol}: timestamp column missing"
            )

        if "close" not in df.columns:
            raise ValueError(
                f"{symbol}: close column missing"
            )

        df["timestamp"] = normalize_timestamp_index(
            df["timestamp"]
        )

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

        df = (
            df
            .dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .drop_duplicates(
                subset="timestamp",
                keep="last",
            )
        )

        s = df.set_index("timestamp")["close"]

        s.name = symbol

        series_list.append(s)

    prices = pd.concat(
        series_list,
        axis=1,
        join="outer",
    ).sort_index()

    prices.index = normalize_timestamp_index(
        prices.index
    )

    return prices


def build_indian_price_matrix() -> pd.DataFrame:
    """
    Build the Indian index price matrix.
    """

    series_list = []

    for symbol in INDIAN_ASSETS:

        series_list.append(
            load_indian_price(symbol)
        )

    india = pd.concat(
        series_list,
        axis=1,
        join="outer",
    ).sort_index()

    india.index = normalize_timestamp_index(
        india.index
    )

    return india


def build_inr_price_matrix() -> pd.DataFrame:
    """
    Build the complete 14-asset QuantOS price matrix.

    Indian assets:
        already denominated in INR.

    US assets:
        USD prices converted into INR using USD/INR.

    The resulting matrix uses the union of trading dates.
    Missing observations are intentionally preserved.
    """

    print("Loading Indian assets...")

    india = build_indian_price_matrix()

    print(
        f"Indian matrix: "
        f"{india.shape[0]} rows × "
        f"{india.shape[1]} assets"
    )

    print("Loading US assets...")

    us_usd = load_us_prices()

    print(
        f"US USD matrix: "
        f"{us_usd.shape[0]} rows × "
        f"{us_usd.shape[1]} assets"
    )

    print("Loading USD/INR...")

    usdinr = load_fx_series(FX_PATH)

    usdinr.index = normalize_timestamp_index(
        usdinr.index
    )

    print(
        f"USD/INR observations: "
        f"{len(usdinr)}"
    )

    print("Converting US prices from USD to INR...")

    us_inr = convert_usd_to_inr(
        usd_prices=us_usd,
        usdinr=usdinr,
    )

    us_inr.index = normalize_timestamp_index(
        us_inr.index
    )

    print("Combining Indian and US assets...")

    prices = pd.concat(
        [
            india,
            us_inr,
        ],
        axis=1,
        join="outer",
    )

    prices.index = normalize_timestamp_index(
        prices.index
    )

    prices = (
        prices
        .sort_index()
        .loc[
            ~prices.index.duplicated(
                keep="last"
            )
        ]
    )

    return prices


if __name__ == "__main__":

    prices = build_inr_price_matrix()

    print()
    print("=" * 80)
    print("QUANTOS 14-ASSET INR PRICE MATRIX")
    print("=" * 80)

    print()
    print("Shape:")
    print(prices.shape)

    print()
    print("Assets:")
    for i, asset in enumerate(
        prices.columns,
        start=1,
    ):
        print(f"{i:2}. {asset}")

    print()
    print("Date range:")
    print(
        f"{prices.index.min().date()} "
        f"→ "
        f"{prices.index.max().date()}"
    )

    print()
    print("Missing observations:")

    print(
        prices.isna().sum()
    )

    print()
    print("Missing percentage:")

    print(
        (prices.isna().mean() * 100)
        .round(2)
    )

    print()
    print("Latest prices:")

    print(
        prices.tail(5)
    )

    print()
    print("All observed prices positive:")

    positive_check = (
        prices
        .stack()
        .gt(0)
        .all()
    )

    print(positive_check)

    print()
    print("=" * 80)