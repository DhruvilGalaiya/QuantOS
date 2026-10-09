from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# QuantOS LIVE FEATURE BUILDER
# ============================================================

INPUT_DIR = Path("data/regime/daily")
OUTPUT_DIR = Path("data/regime/features")

FILES = {
    "NIFTY_50": "nifty_50.parquet",
    "NIFTY_BANK": "nifty_bank.parquet",
    "NIFTY_IT": "nifty_it.parquet",
    "NIFTY_PHARMA": "nifty_pharma.parquet",
    "NIFTY_AUTO": "nifty_auto.parquet",
    "NIFTY_FIN_SERVICE": "nifty_fin_service.parquet",
    "INDIA_VIX": "india_vix.parquet",
    "SENSEX": "sensex.parquet",
}


# EXACT V10 FEATURE NAMES
V10_FEATURES = [
    "log_return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "atr_14",
    "close_vs_sma20",
    "close_vs_sma50",
]


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    return 100 - (100 / (1 + rs))


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    high,
    low,
    close,
    period=14,
):

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return (
        true_range
        .rolling(period)
        .mean()
    )


# ============================================================
# BUILD V10 FEATURES
# ============================================================

def build_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Normalize names
    # --------------------------------------------------------

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Sort / deduplicate
    # --------------------------------------------------------

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ========================================================
    # EXACT V10 FEATURES
    # ========================================================

    # 1. Log return
    df["log_return_1d"] = np.log(
        close / close.shift(1)
    )

    # 2. 20-day volatility
    df["volatility_20"] = (
        df["log_return_1d"]
        .rolling(20)
        .std()
    )

    # 3. RSI
    df["rsi_14"] = calculate_rsi(
        close,
        period=14,
    )

    # 4. Volume ratio
    volume_ma = (
        volume
        .rolling(20)
        .mean()
    )

    valid_volume = (
        volume_ma.notna()
        & (volume_ma > 0)
        & (volume > 0)
    )

    df["volume_ratio"] = np.where(
        valid_volume,
        volume / volume_ma,
        1.0,
    )

    # 5. ATR
    df["atr_14"] = calculate_atr(
        high,
        low,
        close,
        period=14,
    )

    # 6. Close vs SMA20
    sma20 = (
        close
        .rolling(20)
        .mean()
    )

    df["close_vs_sma20"] = (
        close / sma20 - 1
    )

    # 7. Close vs SMA50
    sma50 = (
        close
        .rolling(50)
        .mean()
    )

    df["close_vs_sma50"] = (
        close / sma50 - 1
    )

    # --------------------------------------------------------
    # Remove infinities
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nMissing values before cleaning:")

    for feature in V10_FEATURES:

        count = int(
            df[feature].isna().sum()
        )

        print(
            f"  {feature:<18} {count:>5}"
        )

    # --------------------------------------------------------
    # Remove incomplete warm-up rows
    # --------------------------------------------------------

    rows_before = len(df)

    df = (
        df.dropna(
            subset=V10_FEATURES
        )
        .reset_index(drop=True)
    )

    rows_removed = (
        rows_before - len(df)
    )

    print(
        f"\nRows removed during "
        f"feature cleaning: {rows_removed:,}"
    )

    if df.empty:
        raise ValueError(
            "Feature builder produced ZERO rows."
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    missing_final = int(
        df[V10_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    duplicates = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    if missing_final:
        raise ValueError(
            f"Final missing values: {missing_final}"
        )

    if duplicates:
        raise ValueError(
            f"Final duplicate timestamps: {duplicates}"
        )

    return df


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(symbol, filename):

    input_path = INPUT_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    print("\n" + "=" * 70)
    print(f"PROCESSING {symbol}")
    print("=" * 70)

    raw = pd.read_parquet(
        input_path
    )

    print(
        f"Raw rows:       {len(raw):,}"
    )

    features = build_features(
        raw
    )

    print(
        f"Feature rows:   {len(features):,}"
    )

    print(
        f"Date range:     "
        f"{features['timestamp'].min()} -> "
        f"{features['timestamp'].max()}"
    )

    print(
        f"Duplicates:     "
        f"{features['timestamp'].duplicated().sum()}"
    )

    print(
        f"Missing values: "
        f"{features[V10_FEATURES].isna().sum().sum()}"
    )

    output_path = (
        OUTPUT_DIR / filename
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Saved:          {output_path}"
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("QuantOS LIVE FEATURE BUILDER")
    print("=" * 70)

    results = {}

    for symbol, filename in FILES.items():

        try:

            features = process_symbol(
                symbol,
                filename,
            )

            results[symbol] = len(
                features
            )

        except Exception as error:

            print(
                f"\nERROR: {symbol}: {error}"
            )

    print("\n" + "=" * 70)
    print("FEATURE BUILD COMPLETE")
    print("=" * 70)

    for symbol, rows in results.items():

        print(
            f"{symbol:<22} "
            f"{rows:>6,} rows"
        )

    print(
        f"\nSuccessful datasets: "
        f"{len(results)}/{len(FILES)}"
    )


if __name__ == "__main__":
    main()