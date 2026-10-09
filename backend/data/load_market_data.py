from data.assets import ASSET_UNIVERSE
from data.ingestion import fetch_market_data

from app.database import SessionLocal
from app.models import MarketData


def load_market_data():

    db = SessionLocal()

    total_inserted = 0
    total_skipped = 0

    try:

        for asset_key, asset in ASSET_UNIVERSE.items():

            print("=" * 60)
            print(f"Processing {asset['name']} ({asset['symbol']})")

            try:

                data = fetch_market_data(
                    symbol=asset["symbol"],
                    quantos_symbol=asset_key,
                    period="max",
                    interval="1d"
                )

                inserted = 0
                skipped = 0

                for _, row in data.iterrows():

                    existing = (
                        db.query(MarketData)
                        .filter(
                            MarketData.symbol == row["symbol"],
                            MarketData.timestamp == row["timestamp"]
                        )
                        .first()
                    )

                    if existing:
                        skipped += 1
                        continue

                    market_record = MarketData(
                        timestamp=row["timestamp"],
                        symbol=row["symbol"],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        adjusted_close=float(
                            row["adjusted_close"]
                        ),
                        volume=int(row["volume"]),
                        source=row["source"]
                    )

                    db.add(market_record)
                    inserted += 1

                db.commit()

                total_inserted += inserted
                total_skipped += skipped

                print(
                    f"✓ {asset_key}: "
                    f"{inserted} inserted, "
                    f"{skipped} skipped"
                )

            except Exception as e:

                db.rollback()

                print(
                    f"✗ {asset_key} failed: {e}"
                )

        print("=" * 60)
        print("DATABASE LOAD COMPLETE")
        print("=" * 60)
        print(f"Total inserted: {total_inserted}")
        print(f"Total skipped:  {total_skipped}")

    finally:

        db.close()


if __name__ == "__main__":
    load_market_data()