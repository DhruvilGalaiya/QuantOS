import os
import pandas as pd

from src.upstox_client import get


def fetch_daily_data(
    instrument_name: str,
    instrument_key: str,
    from_date: str,
    to_date: str,
):
    """
    Fetch daily historical candles for one instrument.
    """

    print(
        f"Fetching {instrument_name}: "
        f"{from_date} -> {to_date}"
    )

    data = get(
        f"/v3/historical-candle/"
        f"{instrument_key}/days/1/{to_date}/{from_date}"
    )

    candles = data["data"]["candles"]

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]

    df = pd.DataFrame(candles, columns=columns)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df["instrument"] = instrument_name

    return df


def save_daily_data(
    df: pd.DataFrame,
    instrument_name: str,
):
    """
    Save daily data as Parquet.
    """

    os.makedirs("data/regime/daily", exist_ok=True)

    filename = (
        instrument_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    path = f"data/regime/daily/{filename}.parquet"

    df.to_parquet(path, index=False)

    print(f"Saved: {path}")


def fetch_chunked_daily_data(
    instrument_name: str,
    instrument_key: str,
    chunks: list[tuple[str, str]],
):
    """
    Fetch daily data across multiple date ranges.

    Upstox V3 limits daily historical retrieval
    to approximately one decade per request, so
    longer research histories are downloaded in
    separate chronological chunks.
    """

    all_chunks = []

    for from_date, to_date in chunks:

        df = fetch_daily_data(
            instrument_name=instrument_name,
            instrument_key=instrument_key,
            from_date=from_date,
            to_date=to_date,
        )

        all_chunks.append(df)

        print(
            f"  Retrieved {len(df)} rows"
        )

    if not all_chunks:
        raise RuntimeError(
            f"No data retrieved for {instrument_name}"
        )

    combined = pd.concat(
        all_chunks,
        ignore_index=True,
    )

    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"]
    )

    combined = (
        combined
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return combined


if __name__ == "__main__":

    from src.config import REGIME_INSTRUMENTS

    # -----------------------------------------------------------
    # LONG-HISTORY PERIOD
    # -----------------------------------------------------------

    # We deliberately request more than 10 calendar years
    # because the first ~504 observations will later be used
    # for causal model training.
    #
    # This gives us a much longer out-of-sample evaluation
    # period after the rolling training window.

    DATA_START = "2016-01-01"
    DATA_END = "2026-09-04"

    # -----------------------------------------------------------
    # UPSTOX DAILY API CHUNKS
    # -----------------------------------------------------------

    # Each request remains comfortably inside Upstox's
    # maximum daily retrieval window.

    CHUNKS = [
        ("2016-01-01", "2020-12-31"),
        ("2021-01-01", "2026-09-04"),
    ]

    # -----------------------------------------------------------
    # PORTFOLIO INSTRUMENTS
    # -----------------------------------------------------------

    PORTFOLIO_INSTRUMENTS = {
        "NIFTY_50": REGIME_INSTRUMENTS["NIFTY_50"],
        "NIFTY_BANK": REGIME_INSTRUMENTS["NIFTY_BANK"],
        "NIFTY_MIDCAP_100": REGIME_INSTRUMENTS[
            "NIFTY_MIDCAP_100"
        ],
        "NIFTY_NEXT_50": REGIME_INSTRUMENTS[
            "NIFTY_NEXT_50"
        ],
    }

    print("=" * 70)
    print(
        "QuantOS LONG-HISTORY INDIAN MARKET DATA DOWNLOAD"
    )
    print("=" * 70)

    print(
        f"\nRequested period: "
        f"{DATA_START} -> {DATA_END}"
    )

    print(
        "\nDownload strategy:"
        "\n  Chunk 1: 2016-01-01 -> 2020-12-31"
        "\n  Chunk 2: 2021-01-01 -> 2026-09-04"
    )

    for instrument_name, instrument_key in (
        PORTFOLIO_INSTRUMENTS.items()
    ):

        print("\n" + "-" * 70)
        print(
            f"Downloading {instrument_name}"
        )
        print("-" * 70)

        df = fetch_chunked_daily_data(
            instrument_name=instrument_name,
            instrument_key=instrument_key,
            chunks=CHUNKS,
        )

        save_daily_data(
            df=df,
            instrument_name=instrument_name,
        )

        print(
            f"\nFinal {instrument_name} dataset:"
        )

        print(
            f"  Rows:       {len(df)}"
        )

        print(
            f"  Period:     "
            f"{df['timestamp'].min()} -> "
            f"{df['timestamp'].max()}"
        )

        print(
            f"  Duplicates: "
            f"{df['timestamp'].duplicated().sum()}"
        )

        print(
            f"  Missing close: "
            f"{df['close'].isna().sum()}"
        )

    print("\n" + "=" * 70)
    print("LONG-HISTORY DOWNLOAD COMPLETE")
    print("=" * 70)