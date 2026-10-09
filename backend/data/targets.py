"""
Target generation for QuantOS.

Creates forward-looking prediction targets from market_features.

Important:
- Features remain at time t.
- Targets use future prices only as labels.
- Future prices NEVER become model input features.
- target_end_timestamp is stored for leakage-aware splitting.
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text


# -------------------------------------------------------------------
# Make backend importable when running:
# python data/targets.py
# -------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.database import engine


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

TARGET_HORIZONS = [1, 5]


# -------------------------------------------------------------------
# Create target table
# -------------------------------------------------------------------

def create_targets():
    print("=" * 70)
    print("TARGET GENERATION")
    print("=" * 70)

    with engine.begin() as conn:

        print("\nDropping existing market_targets table if it exists...")

        conn.execute(
            text(
                """
                DROP TABLE IF EXISTS market_targets;
                """
            )
        )

        print("Creating market_targets...")

        conn.execute(
            text(
                """
                CREATE TABLE market_targets AS

                WITH future_prices AS (

                    SELECT
                        timestamp,
                        symbol,
                        close,

                        LEAD(close, 1)
                            OVER (
                                PARTITION BY symbol
                                ORDER BY timestamp
                            ) AS close_t1,

                        LEAD(timestamp, 1)
                            OVER (
                                PARTITION BY symbol
                                ORDER BY timestamp
                            ) AS timestamp_t1,

                        LEAD(close, 5)
                            OVER (
                                PARTITION BY symbol
                                ORDER BY timestamp
                            ) AS close_t5,

                        LEAD(timestamp, 5)
                            OVER (
                                PARTITION BY symbol
                                ORDER BY timestamp
                            ) AS timestamp_t5

                    FROM market_features
                )

                SELECT

                    timestamp,
                    symbol,
                    close,

                    close_t1,
                    timestamp_t1,

                    close_t5,
                    timestamp_t5,

                    CASE
                        WHEN close_t1 IS NOT NULL
                        THEN close_t1 / close - 1.0
                        ELSE NULL
                    END AS target_return_1d,

                    CASE
                        WHEN close_t5 IS NOT NULL
                        THEN close_t5 / close - 1.0
                        ELSE NULL
                    END AS target_return_5d,

                    CASE
                        WHEN close_t1 IS NULL
                        THEN NULL
                        WHEN close_t1 > close
                        THEN 1
                        ELSE 0
                    END AS target_direction_1d,

                    CASE
                        WHEN close_t5 IS NULL
                        THEN NULL
                        WHEN close_t5 > close
                        THEN 1
                        ELSE 0
                    END AS target_direction_5d,

                    timestamp_t1 AS target_end_timestamp_1d,
                    timestamp_t5 AS target_end_timestamp_5d

                FROM future_prices;
                """
            )
        )

        print("Target table created.")

        print("\nCreating indexes...")

        conn.execute(
            text(
                """
                CREATE INDEX idx_market_targets_symbol_timestamp
                ON market_targets(symbol, timestamp);
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX idx_market_targets_target_end_1d
                ON market_targets(target_end_timestamp_1d);
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX idx_market_targets_target_end_5d
                ON market_targets(target_end_timestamp_5d);
                """
            )
        )

    print("\nTARGET GENERATION COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------
# Inspect generated targets
# -------------------------------------------------------------------

def inspect_targets():

    print("\n" + "=" * 70)
    print("TARGET INSPECTION")
    print("=" * 70)

    queries = {

        "total_rows": """
            SELECT COUNT(*) AS total_rows
            FROM market_targets;
        """,

        "symbols": """
            SELECT COUNT(DISTINCT symbol) AS symbols
            FROM market_targets;
        """,

        "target_counts": """
            SELECT
                symbol,
                COUNT(*) AS total_rows,
                COUNT(target_return_1d) AS usable_1d,
                COUNT(target_return_5d) AS usable_5d,
                COUNT(target_direction_1d) AS usable_direction_1d,
                COUNT(target_direction_5d) AS usable_direction_5d
            FROM market_targets
            GROUP BY symbol
            ORDER BY symbol;
        """,

        "direction_balance": """
            SELECT
                symbol,

                COUNT(*) FILTER (
                    WHERE target_direction_1d = 1
                ) AS up_1d,

                COUNT(*) FILTER (
                    WHERE target_direction_1d = 0
                ) AS down_1d,

                COUNT(*) FILTER (
                    WHERE target_direction_5d = 1
                ) AS up_5d,

                COUNT(*) FILTER (
                    WHERE target_direction_5d = 0
                ) AS down_5d

            FROM market_targets
            GROUP BY symbol
            ORDER BY symbol;
        """,

        "sample_rows": """
            SELECT
                timestamp,
                symbol,
                close,
                target_return_1d,
                target_direction_1d,
                target_return_5d,
                target_direction_5d,
                target_end_timestamp_1d,
                target_end_timestamp_5d
            FROM market_targets
            WHERE target_return_5d IS NOT NULL
            ORDER BY symbol, timestamp
            LIMIT 20;
        """
    }

    with engine.connect() as conn:

        for name, query in queries.items():

            print(f"\n--- {name.upper()} ---")

            df = pd.read_sql(text(query), conn)

            print(df.to_string(index=False))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    create_targets()
    inspect_targets()