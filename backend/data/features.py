import pandas as pd
import numpy as np


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add quantitative features to clean market data.

    Features are calculated independently for each symbol.
    """

    required_columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = data.copy()

    # ---------------------------------------------------------
    # Basic preparation
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Price returns
    # ---------------------------------------------------------

    df["return_1d"] = (
        df.groupby("symbol")["close"]
        .pct_change()
    )

    df["log_return_1d"] = (
        df.groupby("symbol")["close"]
        .transform(
            lambda x: np.log(x / x.shift(1))
        )
    )

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    grouped_close = df.groupby("symbol")["close"]

    df["sma_10"] = (
        grouped_close
        .transform(
            lambda x: x.rolling(10).mean()
        )
    )

    df["sma_20"] = (
        grouped_close
        .transform(
            lambda x: x.rolling(20).mean()
        )
    )

    df["sma_50"] = (
        grouped_close
        .transform(
            lambda x: x.rolling(50).mean()
        )
    )

    df["ema_20"] = (
        grouped_close
        .transform(
            lambda x: x.ewm(
                span=20,
                adjust=False
            ).mean()
        )
    )

    df["ema_50"] = (
        grouped_close
        .transform(
            lambda x: x.ewm(
                span=50,
                adjust=False
            ).mean()
        )
    )

    # ---------------------------------------------------------
    # Rolling volatility
    # ---------------------------------------------------------

    df["volatility_20"] = (
        df.groupby("symbol")["log_return_1d"]
        .transform(
            lambda x: x.rolling(20).std()
        )
    )

    # ---------------------------------------------------------
    # RSI 14
    # ---------------------------------------------------------

    def calculate_rsi(series, period=14):

        delta = series.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = (
            gain.rolling(period).mean()
        )

        avg_loss = (
            loss.rolling(period).mean()
        )

        rs = avg_gain / avg_loss

        return 100 - (
            100 / (1 + rs)
        )

    df["rsi_14"] = (
        df.groupby("symbol")["close"]
        .transform(calculate_rsi)
    )

    # ---------------------------------------------------------
    # Volume features
    # ---------------------------------------------------------

    grouped_volume = df.groupby("symbol")["volume"]

    df["volume_sma_20"] = (
        grouped_volume
        .transform(
            lambda x: x.rolling(20).mean()
        )
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_sma_20"]
    )

    # ---------------------------------------------------------
    # Price range
    # ---------------------------------------------------------

    df["high_low_range"] = (
        df["high"] - df["low"]
    )

    df["high_low_range_pct"] = (
        df["high_low_range"] /
        df["close"]
    )

    # ---------------------------------------------------------
    # True Range
    # ---------------------------------------------------------

    previous_close = (
        df.groupby("symbol")["close"]
        .shift(1)
    )

    true_range_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1
    )

    df["true_range"] = (
        true_range_components.max(axis=1)
    )

    # ---------------------------------------------------------
    # ATR 14
    # ---------------------------------------------------------

    df["atr_14"] = (
        df.groupby("symbol")["true_range"]
        .transform(
            lambda x: x.rolling(14).mean()
        )
    )

    # ---------------------------------------------------------
    # Distance from moving averages
    # ---------------------------------------------------------

    df["close_vs_sma20"] = (
        df["close"] / df["sma_20"] - 1
    )

    df["close_vs_sma50"] = (
        df["close"] / df["sma_50"] - 1
    )

    # ---------------------------------------------------------
    # Remove temporary calculation column
    # ---------------------------------------------------------

    df = df.drop(
        columns=["true_range"]
    )

    return df