from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "regime" / "portfolio_us"

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


print("=" * 70)
print("QUANTOS US DATA VALIDATION")
print("=" * 70)

for symbol in US_ASSETS:

    path = DATA_DIR / f"{symbol}.parquet"

    if not path.exists():
        print(f"{symbol:8} MISSING")
        continue

    df = pd.read_parquet(path)

    missing = df["close"].isna().sum()
    duplicates = df["timestamp"].duplicated().sum()

    print(
        f"{symbol:8} | "
        f"rows={len(df):4} | "
        f"{df['timestamp'].min().date()} → "
        f"{df['timestamp'].max().date()} | "
        f"missing={missing} | "
        f"duplicates={duplicates}"
    )

print("=" * 70)