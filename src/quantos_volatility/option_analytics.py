from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import Optional
import os
import time
import requests

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "data/regime/options_live/options_live.sqlite"

UNDERLYINGS = [
    "NIFTY_50",
    "NIFTY_BANK",
    "SENSEX",
]


# ============================================================
# DATABASE
# ============================================================

def load_snapshots(
    db_path: Path = DB_PATH,
    snapshot_ts: Optional[str] = None,
) -> pd.DataFrame:

    if not db_path.exists():
        raise FileNotFoundError(
            f"Options database not found: {db_path}"
        )

    query = """
        SELECT
            snapshot_ts,
            received_ts,
            exchange_ts,
            symbol,
            instrument_key,
            expiry,
            strike,
            option_type,
            ltp,
            previous_close,
            bid,
            bid_qty,
            ask,
            ask_qty,
            volume,
            oi,
            iv,
            delta,
            gamma,
            theta,
            vega,
            rho
        FROM option_snapshots
    """

    params = ()

    if snapshot_ts is not None:
        query += """
            WHERE snapshot_ts = ?
        """
        params = (snapshot_ts,)

    query += """
        ORDER BY symbol, expiry, strike, option_type
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=params,
        )

    if df.empty:
        return df

    df["snapshot_ts"] = pd.to_datetime(
        df["snapshot_ts"],
        errors="coerce",
    )

    df["expiry"] = pd.to_datetime(
        df["expiry"],
        errors="coerce",
    )

    numeric_columns = [
        "strike",
        "ltp",
        "previous_close",
        "bid",
        "bid_qty",
        "ask",
        "ask_qty",
        "volume",
        "oi",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# LATEST SNAPSHOT
# ============================================================

def get_latest_snapshot_ts(
    db_path: Path = DB_PATH,
) -> Optional[str]:

    with sqlite3.connect(db_path) as conn:

        result = conn.execute(
            """
            SELECT MAX(snapshot_ts)
            FROM option_snapshots
            """
        ).fetchone()

    if result is None or result[0] is None:
        return None

    return result[0]


def load_latest_snapshot(
    db_path: Path = DB_PATH,
) -> pd.DataFrame:

    latest_ts = get_latest_snapshot_ts(db_path)

    if latest_ts is None:
        return pd.DataFrame()

    return load_snapshots(
        db_path=db_path,
        snapshot_ts=latest_ts,
    )


# ============================================================
# SPOT PRICE
# ============================================================

# Upstox index instrument keys used for the live spot lookup.
UPSTOX_SPOT_KEYS = {
    "NIFTY_50": "NSE_INDEX|Nifty 50",
    "NIFTY_BANK": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
}

_SPOT_CACHE = {}
_SPOT_CACHE_SECONDS = 3.0


def _load_upstox_token() -> Optional[str]:
    """Load the Analytics Token from the environment or project .env."""
    token = os.getenv("UPSTOX_ANALYTICS_TOKEN")
    if token:
        return token.strip()

    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "UPSTOX_ANALYTICS_TOKEN":
                    return value.strip().strip('"').strip("'")
        except Exception:
            pass

    return None


def _get_live_upstox_spot(symbol: str) -> Optional[float]:
    """
    Get the actual live index LTP from Upstox.

    This is intentionally used instead of estimating spot from option
    put-call parity. The previous implementation used C-P+K across the
    whole option chain, which can be badly distorted by stale/illiquid
    option prices and therefore produced the incorrect 23,605 spot.
    """
    instrument_key = UPSTOX_SPOT_KEYS.get(symbol)
    token = _load_upstox_token()

    if not instrument_key or not token:
        return None

    now = time.monotonic()
    cached = _SPOT_CACHE.get(symbol)
    if cached is not None:
        cached_ts, cached_value = cached
        if now - cached_ts <= _SPOT_CACHE_SECONDS:
            return cached_value

    try:
        response = requests.get(
            "https://api.upstox.com/v3/market-quote/ltp",
            params={"instrument_key": instrument_key},
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=3,
        )
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", {})

        if not data:
            return None

        # Upstox may key the response by display symbol rather than the
        # exact requested instrument key, so take the first valid quote.
        for quote in data.values():
            value = quote.get("last_price")
            if value is not None:
                spot = float(value)
                if np.isfinite(spot) and spot > 0:
                    _SPOT_CACHE[symbol] = (now, spot)
                    return spot

    except Exception:
        pass

    return None


def _estimate_spot_from_options(df: pd.DataFrame) -> Optional[float]:
    """
    Conservative fallback when the live Upstox index quote is unavailable.

    Uses put-call parity only around the middle of the available strike
    range and prefers the nearest expiry. This is a fallback, not the
    primary spot source.
    """
    valid = df[
        df["strike"].notna()
        & df["ltp"].notna()
        & df["option_type"].isin(["CE", "PE"])
    ].copy()

    if valid.empty:
        return None

    nearest_expiry = valid["expiry"].min()
    near = valid[valid["expiry"] == nearest_expiry].copy()

    strikes = np.sort(near["strike"].dropna().unique())
    if len(strikes) == 0:
        return None

    # Keep the central portion of the chain. Deep OTM options are much
    # more likely to contain stale prices and distort parity.
    median_strike = float(np.median(strikes))
    band = max(median_strike * 0.04, 500.0)
    near = near[
        near["strike"].between(
            median_strike - band,
            median_strike + band,
        )
    ].copy()

    calls = (
        near[near["option_type"] == "CE"]
        [["strike", "ltp"]]
        .rename(columns={"ltp": "call_ltp"})
    )

    puts = (
        near[near["option_type"] == "PE"]
        [["strike", "ltp"]]
        .rename(columns={"ltp": "put_ltp"})
    )

    paired = calls.merge(
        puts,
        on="strike",
        how="inner",
    )

    if paired.empty:
        return median_strike

    paired["synthetic_price"] = (
        paired["strike"]
        + paired["call_ltp"]
        - paired["put_ltp"]
    )

    paired = paired[
        np.isfinite(paired["synthetic_price"])
        & (paired["synthetic_price"] > 0)
    ]

    if paired.empty:
        return median_strike

    # Median reduces the impact of one stale option quote.
    return float(paired["synthetic_price"].median())


def get_spot_price(
    df: pd.DataFrame,
) -> Optional[float]:
    """
    Return the actual underlying spot.

    Primary source:
        Upstox V3 live LTP for the underlying index.

    Fallback:
        Conservative option-chain parity estimate.
    """
    if df.empty:
        return None

    symbol = str(df["symbol"].iloc[0])

    live_spot = _get_live_upstox_spot(symbol)
    if live_spot is not None:
        return live_spot

    return _estimate_spot_from_options(df)


# ============================================================
# ATM STRIKE
# ============================================================

def get_atm_strike(
    df: pd.DataFrame,
    spot: Optional[float] = None,
) -> Optional[float]:

    if df.empty:
        return None

    strikes = (
        df["strike"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if strikes.empty:
        return None

    if spot is None:
        spot = get_spot_price(df)

    if spot is None:
        return float(strikes.iloc[len(strikes) // 2])

    return float(
        strikes.iloc[
            np.argmin(
                np.abs(
                    strikes.to_numpy()
                    - spot
                )
            )
        ]
    )


# ============================================================
# EXPIRIES
# ============================================================

def get_available_expiries(
    df: pd.DataFrame,
) -> list[str]:

    if df.empty:
        return []

    expiries = (
        df["expiry"]
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return expiries


# ============================================================
# OPTION CHAIN
# ============================================================

def build_option_chain(
    df: pd.DataFrame,
    expiry: Optional[str] = None,
    strike_range: int = 15,
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    data = df.copy()

    if expiry is not None:
        expiry_dt = pd.to_datetime(expiry)

        data = data[
            data["expiry"] == expiry_dt
        ].copy()

    if data.empty:
        return data

    spot = get_spot_price(data)
    atm = get_atm_strike(data, spot)

    if atm is None:
        return data

    strikes = (
        data["strike"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    atm_index = min(
        range(len(strikes)),
        key=lambda i: abs(strikes[i] - atm),
    )

    low = max(
        0,
        atm_index - strike_range,
    )

    high = min(
        len(strikes),
        atm_index + strike_range + 1,
    )

    selected_strikes = strikes[low:high]

    data = data[
        data["strike"].isin(
            selected_strikes
        )
    ].copy()

    data["moneyness"] = (
        data["strike"] / spot
        if spot
        else np.nan
    )

    data["log_moneyness"] = (
        np.log(
            data["strike"] / spot
        )
        if spot
        else np.nan
    )

    today = pd.Timestamp.now().normalize()

    data["days_to_expiry"] = (
        data["expiry"] - today
    ).dt.total_seconds() / 86400.0

    data["is_atm"] = np.isclose(
        data["strike"],
        atm,
    )

    return data.sort_values(
        ["strike", "option_type"]
    ).reset_index(drop=True)


# ============================================================
# PCR
# ============================================================

def calculate_pcr(
    df: pd.DataFrame,
) -> dict:

    if df.empty:
        return {
            "oi_pcr": np.nan,
            "volume_pcr": np.nan,
        }

    calls = df[
        df["option_type"] == "CE"
    ]

    puts = df[
        df["option_type"] == "PE"
    ]

    call_oi = calls["oi"].sum()
    put_oi = puts["oi"].sum()

    call_volume = calls["volume"].sum()
    put_volume = puts["volume"].sum()

    oi_pcr = (
        put_oi / call_oi
        if call_oi > 0
        else np.nan
    )

    volume_pcr = (
        put_volume / call_volume
        if call_volume > 0
        else np.nan
    )

    return {
        "oi_pcr": float(oi_pcr),
        "volume_pcr": float(volume_pcr),
    }


# ============================================================
# ATM IV
# ============================================================

def calculate_atm_iv(
    df: pd.DataFrame,
) -> float:

    if df.empty:
        return np.nan

    data = df[
        df["strike"].notna()
        & df["iv"].notna()
    ].copy()

    if data.empty:
        return np.nan

    # Determine spot from the available option chain
    spot = get_spot_price(data)

    if spot is None:
        return np.nan

    # Find the strike closest to spot
    strikes = (
        data["strike"]
        .dropna()
        .drop_duplicates()
        .to_numpy()
    )

    if len(strikes) == 0:
        return np.nan

    atm_strike = strikes[
        np.argmin(
            np.abs(strikes - spot)
        )
    ]

    atm_rows = data[
        np.isclose(
            data["strike"],
            atm_strike
        )
    ]

    if atm_rows.empty:
        return np.nan

    # Average CE and PE IV at ATM
    atm_iv = atm_rows["iv"].mean()

    return float(atm_iv)


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_data_quality(
    df: pd.DataFrame,
) -> dict:

    if df.empty:
        return {
            "rows": 0,
            "iv_completeness": 0.0,
            "greeks_completeness": 0.0,
        }

    iv_completeness = (
        df["iv"].notna().mean()
    )

    greek_columns = [
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
    ]

    greek_completeness = (
        df[greek_columns]
        .notna()
        .mean()
        .mean()
    )

    return {
        "rows": int(len(df)),
        "iv_completeness": float(
            iv_completeness
        ),
        "greeks_completeness": float(
            greek_completeness
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def get_option_summary(
    df: pd.DataFrame,
) -> dict:

    if df.empty:
        return {}

    symbol = df["symbol"].iloc[0]

    spot = get_spot_price(df)

    atm = get_atm_strike(
        df,
        spot,
    )

    expiries = get_available_expiries(df)

    pcr = calculate_pcr(df)

    atm_iv = calculate_atm_iv(
        df
    )

    quality = calculate_data_quality(
        df
    )

    return {
        "symbol": symbol,
        "snapshot_ts": (
            df["snapshot_ts"]
            .max()
            .isoformat()
        ),
        "spot": spot,
        "atm_strike": atm,
        "expiries": expiries,
        "oi_pcr": pcr["oi_pcr"],
        "volume_pcr": pcr["volume_pcr"],
        "atm_iv": atm_iv,
        "rows": quality["rows"],
        "iv_completeness": quality[
            "iv_completeness"
        ],
        "greeks_completeness": quality[
            "greeks_completeness"
        ],
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("QuantOS OPTION ANALYTICS TEST")
    print("=" * 70)

    print(f"Database: {DB_PATH}")

    df = load_latest_snapshot()

    if df.empty:
        print("No live snapshot available.")
        raise SystemExit(1)

    print(
        f"\nLatest snapshot: "
        f"{df['snapshot_ts'].max()}"
    )

    print(
        f"Rows received: {len(df)}"
    )

    print(
        "\nSymbols:"
    )

    print(
        df["symbol"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "-" * 70)

    for symbol in UNDERLYINGS:

        symbol_df = df[
            df["symbol"] == symbol
        ].copy()

        if symbol_df.empty:
            continue

        summary = get_option_summary(
            symbol_df
        )

        print(
            f"\n{symbol}"
        )

        print(
            f"Spot: "
            f"{summary['spot']:.2f}"
            if summary["spot"] is not None
            else "Spot: N/A"
        )

        print(
            f"ATM Strike: "
            f"{summary['atm_strike']}"
        )

        print(
            f"Expiries: "
            f"{summary['expiries']}"
        )

        print(
            f"OI PCR: "
            f"{summary['oi_pcr']:.4f}"
            if np.isfinite(summary["oi_pcr"])
            else "OI PCR: N/A"
        )

        print(
            f"Volume PCR: "
            f"{summary['volume_pcr']:.4f}"
            if np.isfinite(
                summary["volume_pcr"]
            )
            else "Volume PCR: N/A"
        )

        print(
            f"ATM IV: "
            f"{summary['atm_iv'] * 100:.2f}%"
            if np.isfinite(summary["atm_iv"])
            else "ATM IV: N/A"
        )

        print(
            f"Rows: "
            f"{summary['rows']}"
        )

        print(
            f"IV completeness: "
            f"{summary['iv_completeness'] * 100:.1f}%"
        )

        print(
            f"Greeks completeness: "
            f"{summary['greeks_completeness'] * 100:.1f}%"
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)