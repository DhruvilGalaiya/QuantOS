import yfinance as yf
import pandas as pd


def fetch_market_data(
    symbol: str,
    quantos_symbol: str,
    period: str = "max",
    interval: str = "1d"
) -> pd.DataFrame:

    print(f"Downloading {quantos_symbol} from Yahoo Finance...")

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data returned for {quantos_symbol} ({symbol})"
        )

    # Flatten MultiIndex columns returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Standardize timestamp column
    if "Date" in data.columns:
        data = data.rename(columns={"Date": "timestamp"})

    elif "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "timestamp"})

    # Standardize column names
    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume"
        }
    )

    # Add metadata
    data["symbol"] = quantos_symbol
    data["source"] = "Yahoo Finance"

    # Ensure timestamp is datetime
    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    # Select standard schema
    columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source"
    ]

    data = data[columns]

    # Remove rows with missing essential market data
    data = data.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    # Sort chronologically
    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return data