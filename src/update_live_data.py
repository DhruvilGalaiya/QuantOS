"""
QuantOS Incremental Live Data Updater

Purpose:
    Update existing daily market-data Parquet files only with
    dates missing after the latest stored observation.

This script does NOT:
    - retrain the HMM
    - modify HMM parameters
    - rebuild the historical dataset
    - change the V10 model
    - overwrite existing historical observations

It only performs:
    existing parquet
          ↓
    find latest stored date
          ↓
    fetch missing dates from Upstox
          ↓
    merge + deduplicate
          ↓
    save updated parquet
"""

from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import REGIME_INSTRUMENTS, VALIDATION_INSTRUMENTS
from upstox_client import get


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/regime/daily")

IST = ZoneInfo("Asia/Kolkata")

# Combine regime + validation instruments.
INSTRUMENTS = {
    **REGIME_INSTRUMENTS,
    **VALIDATION_INSTRUMENTS,
}


# ============================================================
# HELPERS
# ============================================================

def normalize_timestamps(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to timezone-naive IST timestamps.

    Existing QuantOS files may contain timezone-naive timestamps,
    while Upstox responses can contain timezone-aware timestamps.
    This prevents pandas merge/concat timezone conflicts.
    """

    ts = pd.to_datetime(series, errors="coerce")

    if getattr(ts.dt, "tz", None) is not None:
        ts = (
            ts.dt
            .tz_convert("Asia/Kolkata")
            .dt
            .tz_localize(None)
        )

    return ts


def fetch_daily_data(
    instrument_key: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily candles from Upstox V3.
    """

    endpoint = (
        f"/v3/historical-candle/"
        f"{instrument_key}/days/1/"
        f"{end_date}/{start_date}"
    )

    response = get(endpoint)

    data = response.get("data", {})
    candles = data.get("candles", [])

    if not candles:
        return pd.DataFrame()

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]

    rows = []

    for candle in candles:
        row = candle[:len(columns)]

        # Pad missing columns if necessary.
        row += [None] * (len(columns) - len(row))

        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    df["timestamp"] = normalize_timestamps(df["timestamp"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "close"])

    return df


# ============================================================
# UPDATE ONE INSTRUMENT
# ============================================================

def update_instrument(
    symbol: str,
    instrument_key: str,
    today: pd.Timestamp,
) -> None:

    file_path = DATA_DIR / f"{symbol.lower()}.parquet"

    print()
    print("=" * 70)
    print(f"UPDATING: {symbol}")
    print("=" * 70)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    # --------------------------------------------------------
    # Load existing data
    # --------------------------------------------------------

    existing = pd.read_parquet(file_path)

    if "timestamp" not in existing.columns:
        print("[ERROR] Missing timestamp column.")
        return

    existing["timestamp"] = normalize_timestamps(
        existing["timestamp"]
    )

    existing = (
        existing
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    latest_existing = existing["timestamp"].max()

    print(f"Existing rows : {len(existing):,}")
    print(f"Latest stored : {latest_existing}")

    # --------------------------------------------------------
    # Determine missing date range
    # --------------------------------------------------------

    latest_date = latest_existing.date()
    today_date = today.date()

    if latest_date >= today_date:
        print("Status        : ALREADY UP TO DATE")
        return

    start_date = latest_date + timedelta(days=1)
    end_date = today_date

    print(f"Fetch start   : {start_date}")
    print(f"Fetch end     : {end_date}")
    print(f"Instrument    : {instrument_key}")

    # --------------------------------------------------------
    # Fetch missing data
    # --------------------------------------------------------

    new_data = fetch_daily_data(
        instrument_key=instrument_key,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    if new_data.empty:
        print("Status        : NO NEW TRADING-DAY DATA AVAILABLE")
        return

    print(f"New rows      : {len(new_data):,}")

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    combined = pd.concat(
        [existing, new_data],
        ignore_index=True,
    )

    combined["timestamp"] = normalize_timestamps(
        combined["timestamp"]
    )

    combined = (
        combined
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    duplicate_count = combined["timestamp"].duplicated().sum()

    if duplicate_count != 0:
        raise RuntimeError(
            f"{symbol}: duplicate timestamps remain: "
            f"{duplicate_count}"
        )

    if combined["close"].isna().any():
        raise RuntimeError(
            f"{symbol}: missing close values detected."
        )

    latest_final = combined["timestamp"].max()

    print(f"Final rows    : {len(combined):,}")
    print(f"Latest final  : {latest_final}")
    print(f"Rows added    : {len(combined) - len(existing):,}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_parquet(
        file_path,
        index=False,
    )

    print(f"Saved         : {file_path}")
    print("Status        : UPDATED")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("QUANTOS INCREMENTAL LIVE DATA UPDATE")
    print("=" * 70)

    now_ist = pd.Timestamp.now(tz=IST)
    today = now_ist.normalize()

    print(f"Current IST   : {now_ist}")
    print(f"Target date   : {today.date()}")
    print(f"Data directory: {DATA_DIR}")

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Instruments   : {len(INSTRUMENTS)}")

    # --------------------------------------------------------
    # Update every configured instrument
    # --------------------------------------------------------

    for symbol, instrument_key in INSTRUMENTS.items():

        try:

            update_instrument(
                symbol=symbol,
                instrument_key=instrument_key,
                today=today,
            )

        except Exception as exc:

            print()
            print(f"[ERROR] {symbol}")
            print(f"        {type(exc).__name__}: {exc}")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("UPDATE COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. python src/build_live_features.py")
    print("2. python src/run_live_hmm.py")
    print()


if __name__ == "__main__":
    main()