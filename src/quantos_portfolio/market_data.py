from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NATIVE_DATA_DIR = ROOT / "data" / "regime" / "daily"


def _normalize_price_series(
    prices: pd.Series,
    name: str,
) -> pd.Series:
    """
    Normalize a price series into the QuantOS
    standard format.

    Output:
        DatetimeIndex named 'date'
        Series named after the asset
    """
    result = prices.copy()

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result[~result.index.isna()]

    # Remove timezone information so that native
    # and user-provided datasets can be aligned.
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    result = pd.to_numeric(
        result,
        errors="coerce",
    )

    result = result.dropna()

    result = result[~result.index.duplicated(
        keep="last"
    )]

    result = result.sort_index()

    result.name = name
    result.index.name = "date"

    return result


def load_native_prices(
    symbol: str,
) -> pd.Series:
    """
    Load a native QuantOS asset from daily Parquet data.
    """
    symbol = symbol.upper().strip()

    path = NATIVE_DATA_DIR / f"{symbol}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Native data not found for {symbol}: {path}"
        )

    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{symbol} dataset does not contain "
            "'timestamp'."
        )

    if "close" not in df.columns:
        raise ValueError(
            f"{symbol} dataset does not contain "
            "'close'."
        )

    dates = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    prices = pd.Series(
        df["close"].values,
        index=dates,
        name=symbol,
    )

    return _normalize_price_series(
        prices,
        symbol,
    )


def load_csv_prices(
    path: str | Path,
    symbol: str | None = None,
    date_column: str = "Date",
    price_column: str = "Close",
) -> pd.Series:
    """
    Load a single-asset price series from CSV.

    Expected format:

        Date,Close
        2024-01-01,100.25
        2024-01-02,101.10

    symbol:
        Name assigned to the resulting price series.
        If omitted, the CSV filename is used.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    df = pd.read_csv(path)

    if date_column not in df.columns:
        raise ValueError(
            f"CSV does not contain date column "
            f"'{date_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    if price_column not in df.columns:
        raise ValueError(
            f"CSV does not contain price column "
            f"'{price_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    if symbol is None:
        symbol = path.stem.upper()

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    prices = pd.Series(
        df[price_column].values,
        index=dates,
        name=symbol.upper(),
    )

    return _normalize_price_series(
        prices,
        symbol.upper(),
    )


def combine_price_series(
    series_list: list[pd.Series],
) -> pd.DataFrame:
    """
    Combine multiple price series into one aligned
    price matrix.

    No forward filling is performed.
    """
    if not series_list:
        raise ValueError(
            "No price series supplied."
        )

    normalized = []

    for series in series_list:
        normalized.append(
            _normalize_price_series(
                series,
                str(series.name),
            )
        )

    prices = pd.concat(
        normalized,
        axis=1,
        join="outer",
    )

    prices = prices.sort_index()

    prices.index.name = "date"

    return prices


def load_native_price_matrix(
    symbols: list[str],
) -> pd.DataFrame:
    """
    Load multiple native QuantOS assets.
    """
    if not symbols:
        raise ValueError(
            "No symbols supplied."
        )

    series = [
        load_native_prices(symbol)
        for symbol in symbols
    ]

    return combine_price_series(series)


def load_mixed_price_matrix(
    native_symbols: list[str] | None = None,
    csv_assets: dict[str, str | Path] | None = None,
) -> pd.DataFrame:
    """
    Load a mixed portfolio universe.

    native_symbols:
        Example:
            ["NIFTY_50", "NIFTY_BANK"]

    csv_assets:
        Example:
            {
                "AAPL": "/path/to/AAPL.csv",
                "RELIANCE": "/path/to/RELIANCE.csv",
            }
    """
    series = []

    if native_symbols:
        for symbol in native_symbols:
            series.append(
                load_native_prices(symbol)
            )

    if csv_assets:
        for symbol, path in csv_assets.items():
            series.append(
                load_csv_prices(
                    path,
                    symbol=symbol,
                )
            )

    if not series:
        raise ValueError(
            "No native symbols or CSV assets supplied."
        )

    return combine_price_series(series)


def validate_price_matrix(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a price matrix before it enters
    the QuantOS return engine.
    """
    if prices.empty:
        raise ValueError(
            "Price matrix is empty."
        )

    result = prices.copy()

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result[
        ~result.index.isna()
    ]

    result = result.sort_index()

    result = result[
        ~result.index.duplicated(
            keep="last"
        )
    ]

    for column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # Prices cannot be zero or negative.
    result = result.where(result > 0)

    return result 