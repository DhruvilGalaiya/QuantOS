import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from app.database import engine
from features import add_features


def load_features():

    print("=" * 60)
    print("FEATURE ENGINEERING LOAD")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load clean market data
    # ---------------------------------------------------------

    print("\nLoading clean market data...")

    query = """
        SELECT
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume,
            source
        FROM clean_market_data
        ORDER BY symbol, timestamp;
    """

    data = pd.read_sql(
        query,
        engine
    )

    print(
        f"Loaded {len(data)} clean rows."
    )

    if data.empty:
        raise ValueError(
            "clean_market_data is empty."
        )

    # ---------------------------------------------------------
    # Generate features
    # ---------------------------------------------------------

    print("\nGenerating features...")

    features = add_features(data)

    print(
        f"Generated features for "
        f"{len(features)} rows."
    )

    # ---------------------------------------------------------
    # Select database columns
    # ---------------------------------------------------------

    feature_columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",

        "return_1d",
        "log_return_1d",

        "sma_10",
        "sma_20",
        "sma_50",

        "ema_20",
        "ema_50",

        "volatility_20",

        "rsi_14",

        "volume_sma_20",
        "volume_ratio",

        "high_low_range",
        "high_low_range_pct",

        "atr_14",

        "close_vs_sma20",
        "close_vs_sma50",
    ]

    features = features[
        feature_columns
    ].copy()

    # ---------------------------------------------------------
    # Insert into PostgreSQL
    # ---------------------------------------------------------

    print("\nWriting features to PostgreSQL...")

    insert_query = text("""
        INSERT INTO market_features (
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume,
            return_1d,
            log_return_1d,
            sma_10,
            sma_20,
            sma_50,
            ema_20,
            ema_50,
            volatility_20,
            rsi_14,
            volume_sma_20,
            volume_ratio,
            high_low_range,
            high_low_range_pct,
            atr_14,
            close_vs_sma20,
            close_vs_sma50
        )
        VALUES (
            :timestamp,
            :symbol,
            :open,
            :high,
            :low,
            :close,
            :adjusted_close,
            :volume,
            :return_1d,
            :log_return_1d,
            :sma_10,
            :sma_20,
            :sma_50,
            :ema_20,
            :ema_50,
            :volatility_20,
            :rsi_14,
            :volume_sma_20,
            :volume_ratio,
            :high_low_range,
            :high_low_range_pct,
            :atr_14,
            :close_vs_sma20,
            :close_vs_sma50
        )
        ON CONFLICT (symbol, timestamp)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume,
            return_1d = EXCLUDED.return_1d,
            log_return_1d = EXCLUDED.log_return_1d,
            sma_10 = EXCLUDED.sma_10,
            sma_20 = EXCLUDED.sma_20,
            sma_50 = EXCLUDED.sma_50,
            ema_20 = EXCLUDED.ema_20,
            ema_50 = EXCLUDED.ema_50,
            volatility_20 = EXCLUDED.volatility_20,
            rsi_14 = EXCLUDED.rsi_14,
            volume_sma_20 = EXCLUDED.volume_sma_20,
            volume_ratio = EXCLUDED.volume_ratio,
            high_low_range = EXCLUDED.high_low_range,
            high_low_range_pct = EXCLUDED.high_low_range_pct,
            atr_14 = EXCLUDED.atr_14,
            close_vs_sma20 = EXCLUDED.close_vs_sma20,
            close_vs_sma50 = EXCLUDED.close_vs_sma50;
    """)

    records = (
        features
        .where(pd.notnull(features), None)
        .to_dict(orient="records")
    )

    with engine.begin() as connection:

        connection.execute(
            insert_query,
            records
        )

    print(
        f"Inserted/updated {len(records)} rows."
    )

    print("\nFEATURE LOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    load_features()