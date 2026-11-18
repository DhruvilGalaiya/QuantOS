from pathlib import Path
from datetime import date, timedelta
import shutil

import pandas as pd

from src.upstox_client import get
from src.config import REGIME_INSTRUMENTS, VALIDATION_INSTRUMENTS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "regime" / "daily"
BACKUP_DIR = ROOT / "data" / "backups" / "hmm_daily_before_refresh"

INSTRUMENTS = {
    **REGIME_INSTRUMENTS,
    **VALIDATION_INSTRUMENTS,
}

def normalize_timestamp(series):
    values = pd.to_datetime(series, errors="coerce", utc=True)
    return values.dt.tz_convert("Asia/Kolkata")

def main():
    today = date.today()
    print("=" * 70)
    print("QUANTOS INCREMENTAL HMM DATA REFRESH")
    print("=" * 70)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0

    for symbol, instrument_key in INSTRUMENTS.items():
        path = DATA_DIR / f"{symbol.lower()}.parquet"

        if not path.exists():
            print(f"\nSKIP {symbol}: file not found: {path}")
            failures += 1
            continue

        try:
            old = pd.read_parquet(path)

            if "timestamp" not in old.columns or "close" not in old.columns:
                raise ValueError("Expected timestamp and close columns")

            old["timestamp"] = normalize_timestamp(old["timestamp"])
            old = old.dropna(subset=["timestamp", "close"])

            latest = old["timestamp"].max()
            start_date = (latest.date() - timedelta(days=7)).isoformat()
            end_date = today.isoformat()

            print(f"\n{symbol}: existing latest={latest.date()}")
            print(f"Fetching {start_date} -> {end_date}")

            response = get(
                f"/v3/historical-candle/{instrument_key}/days/1/"
                f"{end_date}/{start_date}"
            )
            candles = response.get("data", {}).get("candles", [])

            if not candles:
                print(f"NO NEW CANDLES returned for {symbol}; keeping existing file")
                successes += 1
                continue

            columns = [
                "timestamp", "open", "high", "low",
                "close", "volume", "open_interest"
            ]
            fresh = pd.DataFrame(candles, columns=columns)
            fresh["timestamp"] = normalize_timestamp(fresh["timestamp"])

            for col in ["open", "high", "low", "close", "volume"]:
                fresh[col] = pd.to_numeric(fresh[col], errors="coerce")

            fresh = fresh.dropna(subset=["timestamp", "close"])

            # Keep the existing file's schema and preserve its history.
            for col in old.columns:
                if col not in fresh.columns:
                    fresh[col] = pd.NA
            for col in fresh.columns:
                if col not in old.columns:
                    old[col] = pd.NA

            fresh = fresh[old.columns]
            merged = pd.concat([old, fresh], ignore_index=True)
            merged = (
                merged.sort_values("timestamp")
                .drop_duplicates("timestamp", keep="last")
                .reset_index(drop=True)
            )

            if merged.empty or merged["close"].isna().any():
                raise ValueError("Merged data failed validation")

            if merged["timestamp"].duplicated().any():
                raise ValueError("Duplicate timestamps remain")

            backup = BACKUP_DIR / path.name
            if not backup.exists():
                shutil.copy2(path, backup)

            temp_path = path.with_suffix(".parquet.tmp")
            merged.to_parquet(temp_path, index=False)
            temp_path.replace(path)

            latest_after = merged["timestamp"].max()
            added = len(merged) - len(old)

            print(f"Rows: {len(old)} -> {len(merged)} (net change {added:+})")
            print(f"Latest after refresh: {latest_after}")
            print(f"Saved: {path}")
            successes += 1

        except Exception as exc:
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")
            failures += 1

    print("\n" + "=" * 70)
    print(f"Refresh finished: {successes} successful, {failures} failed/skipped")
    print(f"Backups: {BACKUP_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
