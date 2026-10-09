from ingestion import fetch_market_data
from features import add_features


print("Loading market data...")

data = fetch_market_data(
    symbol="^NSEI",
    quantos_symbol="NIFTY50",
    period="5y",
    interval="1d"
)

print(f"Raw rows: {len(data)}")

print("\nAdding features...")

features = add_features(data)

print(f"Feature rows: {len(features)}")

print("\nFeature columns:")

print(
    features.columns.tolist()
)

print("\nLast 5 rows:")

print(
    features[
        [
            "timestamp",
            "symbol",
            "close",
            "return_1d",
            "sma_20",
            "sma_50",
            "rsi_14",
            "volatility_20",
            "atr_14",
            "volume_ratio"
        ]
    ].tail()
)

print("\nFeature null counts:")

print(
    features[
        [
            "return_1d",
            "sma_20",
            "sma_50",
            "rsi_14",
            "volatility_20",
            "atr_14"
        ]
    ].isnull().sum()
)

print("\nFeature engineering completed successfully.")