from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ============================================================
# QuantOS IMPLIED VOLATILITY SURFACE ENGINE
# ============================================================
#
# Data source:
#     data/regime/options_live/options_live.sqlite
#
# Outputs:
#     data/regime/volatility_surface/
#
# For each underlying:
#     - Live spot
#     - ATM strike
#     - Strike / expiry IV observations
#     - Interactive 3D IV surface
#     - JSON surface artifact
#
# Underlyings:
#     NIFTY 50
#     NIFTY BANK
#     SENSEX
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DB_PATH = (
    ROOT
    / "data"
    / "regime"
    / "options_live"
    / "options_live.sqlite"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "regime"
    / "volatility_surface"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SYMBOLS = {
    "NIFTY_50": "NIFTY 50",
    "NIFTY_BANK": "NIFTY BANK",
    "SENSEX": "SENSEX",
}


# Keep the useful central region of the surface.
# This prevents very far OTM options from dominating
# the visualization.
MAX_MONEYNESS = 0.12


# Valid IV sanity bounds.
MIN_IV = 0.001
MAX_IV = 3.0


# ============================================================
# DATABASE
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Open the QuantOS live option database.
    """

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Options database not found:\n{DB_PATH}"
        )

    return sqlite3.connect(DB_PATH)


def get_latest_snapshot_ts(
    conn: sqlite3.Connection,
) -> Optional[str]:
    """
    Get the latest snapshot timestamp.

    IMPORTANT:
    The actual database column is snapshot_ts,
    not timestamp.
    """

    row = conn.execute(
        """
        SELECT MAX(snapshot_ts)
        FROM option_snapshots
        """
    ).fetchone()

    if row is None:
        return None

    if row[0] is None:
        return None

    return str(row[0])


# ============================================================
# LOAD LATEST OPTION SNAPSHOT
# ============================================================

def load_latest_snapshot(
    symbol: str,
) -> pd.DataFrame:
    """
    Load the latest available option snapshot
    for a specific underlying.
    """

    conn = get_connection()

    try:

        latest_ts = get_latest_snapshot_ts(conn)

        if latest_ts is None:
            return pd.DataFrame()

        query = """
            SELECT
                snapshot_ts,
                received_ts,
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
            WHERE snapshot_ts = ?
              AND symbol = ?
            ORDER BY expiry, strike, option_type
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(
                latest_ts,
                symbol,
            ),
        )

    finally:
        conn.close()

    if df.empty:
        return df

    return normalize_dataframe(df)


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize option data types.

    All timestamps are converted to timezone-naive
    pandas timestamps so that expiry arithmetic cannot
    produce tz-aware / tz-naive comparison errors.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Snapshot timestamp
    # --------------------------------------------------------

    if "snapshot_ts" in df.columns:

        df["snapshot_ts"] = pd.to_datetime(
            df["snapshot_ts"],
            errors="coerce",
            utc=True,
        )

        df["snapshot_ts"] = (
            df["snapshot_ts"]
            .dt
            .tz_convert(None)
        )

    # --------------------------------------------------------
    # Received timestamp
    # --------------------------------------------------------

    if "received_ts" in df.columns:

        df["received_ts"] = pd.to_datetime(
            df["received_ts"],
            errors="coerce",
            utc=True,
        )

        df["received_ts"] = (
            df["received_ts"]
            .dt
            .tz_convert(None)
        )

    # --------------------------------------------------------
    # Expiry
    # --------------------------------------------------------

    if "expiry" in df.columns:

        df["expiry"] = pd.to_datetime(
            df["expiry"],
            errors="coerce",
            utc=True,
        )

        df["expiry"] = (
            df["expiry"]
            .dt
            .tz_convert(None)
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

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

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# CLEAN OPTION DATA
# ============================================================

def clean_option_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only valid option observations.

    Missing IV values are not fabricated.
    """

    if df.empty:
        return df

    df = normalize_dataframe(df)

    # Valid strike
    df = df.dropna(
        subset=[
            "strike",
            "expiry",
        ]
    )

    # Valid option type
    df = df[
        df["option_type"].isin(
            [
                "CE",
                "PE",
            ]
        )
    ]

    # Valid IV
    df = df.dropna(
        subset=[
            "iv",
        ]
    )

    df = df[
        (df["iv"] >= MIN_IV)
        &
        (df["iv"] <= MAX_IV)
    ]

    # Valid expiry
    df = df[
        df["expiry"].notna()
    ]

    return df.reset_index(
        drop=True
    )


# ============================================================
# LIVE SPOT
# ============================================================

def get_live_spot(
    df: pd.DataFrame,
) -> Optional[float]:
    """
    Use the same live Upstox spot logic already implemented
    in option_analytics.py.

    If that fails, fall back to put-call parity.
    """

    if df.empty:
        return None

    # --------------------------------------------------------
    # Primary source: option_analytics.py
    # --------------------------------------------------------

    try:

        from quantos_volatility.option_analytics import (
            get_spot_price,
        )

        spot = get_spot_price(df)

        if spot is not None:

            spot = float(spot)

            if np.isfinite(spot) and spot > 0:
                return spot

    except Exception as exc:

        print(
            "Warning: live spot lookup failed. "
            f"Using option-chain fallback. Reason: {exc}"
        )

    # --------------------------------------------------------
    # Fallback: put-call parity
    # --------------------------------------------------------

    return estimate_spot_from_options(df)


# ============================================================
# OPTION PARITY SPOT FALLBACK
# ============================================================

def estimate_spot_from_options(
    df: pd.DataFrame,
) -> Optional[float]:
    """
    Conservative spot estimate using:

        S ≈ K + Call - Put

    Uses the nearest expiry and the central strike region.
    """

    if df.empty:
        return None

    expiries = sorted(
        df["expiry"]
        .dropna()
        .unique()
    )

    if not expiries:
        return None

    nearest_expiry = expiries[0]

    chain = df[
        df["expiry"] == nearest_expiry
    ].copy()

    if chain.empty:
        return None

    calls = chain[
        chain["option_type"] == "CE"
    ][
        [
            "strike",
            "ltp",
        ]
    ].rename(
        columns={
            "ltp": "call_ltp"
        }
    )

    puts = chain[
        chain["option_type"] == "PE"
    ][
        [
            "strike",
            "ltp",
        ]
    ].rename(
        columns={
            "ltp": "put_ltp"
        }
    )

    pairs = calls.merge(
        puts,
        on="strike",
        how="inner",
    )

    if pairs.empty:
        return None

    pairs = pairs.dropna(
        subset=[
            "call_ltp",
            "put_ltp",
        ]
    )

    if pairs.empty:
        return None

    pairs["synthetic_spot"] = (
        pairs["strike"]
        + pairs["call_ltp"]
        - pairs["put_ltp"]
    )

    values = (
        pairs["synthetic_spot"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if values.empty:
        return None

    # Robust central estimate.
    return float(
        values.median()
    )


# ============================================================
# ATM STRIKE
# ============================================================

def get_atm_strike(
    df: pd.DataFrame,
    spot: float,
) -> Optional[float]:
    """
    Select the strike closest to the live spot.
    """

    if df.empty:
        return None

    strikes = (
        df["strike"]
        .dropna()
        .unique()
    )

    if len(strikes) == 0:
        return None

    atm = min(
        strikes,
        key=lambda x: abs(
            float(x) - spot
        ),
    )

    return float(atm)


# ============================================================
# BUILD SURFACE DATA
# ============================================================

def build_surface_data(
    symbol: str,
) -> tuple[
    pd.DataFrame,
    Optional[float],
    Optional[float],
]:
    """
    Build the clean IV surface dataset.

    Returns:

        surface_df
        spot
        atm_strike
    """

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_latest_snapshot(
        symbol
    )

    if df.empty:
        return (
            pd.DataFrame(),
            None,
            None,
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_option_data(
        df
    )

    if df.empty:
        return (
            pd.DataFrame(),
            None,
            None,
        )

    # --------------------------------------------------------
    # Spot
    # --------------------------------------------------------

    spot = get_live_spot(
        df
    )

    if (
        spot is None
        or not np.isfinite(spot)
        or spot <= 0
    ):
        return (
            pd.DataFrame(),
            None,
            None,
        )

    # --------------------------------------------------------
    # ATM
    # --------------------------------------------------------

    atm_strike = get_atm_strike(
        df,
        spot,
    )

    if atm_strike is None:
        return (
            pd.DataFrame(),
            spot,
            None,
        )

    # --------------------------------------------------------
    # Moneyness
    # --------------------------------------------------------

    df["moneyness"] = (
        df["strike"] / spot
        - 1.0
    )

    df["log_moneyness"] = np.log(
        df["strike"] / spot
    )

    # Central surface only.
    df = df[
        df["moneyness"].abs()
        <= MAX_MONEYNESS
    ]

    if df.empty:
        return (
            pd.DataFrame(),
            spot,
            atm_strike,
        )

    # --------------------------------------------------------
    # Reference timestamp
    # --------------------------------------------------------

    reference_timestamp = (
        df["snapshot_ts"]
        .dropna()
        .max()
    )

    if pd.isna(
        reference_timestamp
    ):

        reference_date = (
            pd.Timestamp.now()
            .normalize()
        )

    else:

        reference_date = (
            pd.Timestamp(
                reference_timestamp
            )
            .tz_localize(None)
            .normalize()
        )

    # --------------------------------------------------------
    # Days to expiry
    # --------------------------------------------------------

    df["days_to_expiry"] = (
        df["expiry"]
        .dt
        .normalize()
        - reference_date
    ).dt.days

    # Only future expiries.
    df = df[
        df["days_to_expiry"] > 0
    ]

    if df.empty:
        return (
            pd.DataFrame(),
            spot,
            atm_strike,
        )

    # --------------------------------------------------------
    # Aggregate CE + PE
    # --------------------------------------------------------
    #
    # For each:
    #
    #     expiry × strike
    #
    # combine available CE/PE IV observations.
    #
    # Median is used so one stale/extreme side does not
    # dominate the surface.
    # --------------------------------------------------------

    grouped = (
        df.groupby(
            [
                "expiry",
                "days_to_expiry",
                "strike",
            ],
            as_index=False,
        )
        .agg(
            implied_volatility=(
                "iv",
                "median",
            ),
            observations=(
                "iv",
                "count",
            ),
            mean_oi=(
                "oi",
                "mean",
            ),
            mean_volume=(
                "volume",
                "mean",
            ),
        )
    )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------

    grouped["moneyness"] = (
        grouped["strike"]
        / spot
        - 1.0
    )

    grouped["log_moneyness"] = np.log(
        grouped["strike"]
        / spot
    )

    grouped["implied_volatility_pct"] = (
        grouped["implied_volatility"]
        * 100.0
    )

    grouped["is_atm"] = np.isclose(
        grouped["strike"],
        atm_strike,
    )

    grouped = grouped.sort_values(
        [
            "days_to_expiry",
            "strike",
        ]
    ).reset_index(
        drop=True
    )

    return (
        grouped,
        float(spot),
        float(atm_strike),
    )


# ============================================================
# CREATE 3D IV SURFACE
# ============================================================

def create_surface_plot(
    symbol: str,
    surface_df: pd.DataFrame,
    spot: float,
    atm_strike: float,
) -> go.Figure:
    """
    Create interactive Plotly 3D IV surface.
    """

    display_name = SYMBOLS[
        symbol
    ]

    fig = go.Figure()

    # --------------------------------------------------------
    # Main surface
    # --------------------------------------------------------

    fig.add_trace(
        go.Mesh3d(
            x=surface_df[
                "strike"
            ],
            y=surface_df[
                "days_to_expiry"
            ],
            z=surface_df[
                "implied_volatility_pct"
            ],
            intensity=surface_df[
                "implied_volatility_pct"
            ],
            colorscale="Viridis",
            opacity=0.88,
            colorbar=dict(
                title="IV (%)"
            ),
            hovertemplate=(
                "<b>Strike:</b> %{x:.0f}"
                "<br>"
                "<b>Days to Expiry:</b> %{y}"
                "<br>"
                "<b>Implied Volatility:</b> %{z:.2f}%"
                "<extra></extra>"
            ),
            name="IV Surface",
        )
    )

    # --------------------------------------------------------
    # ATM observations
    # --------------------------------------------------------

    atm = surface_df[
        surface_df["is_atm"]
    ]

    if not atm.empty:

        fig.add_trace(
            go.Scatter3d(
                x=atm[
                    "strike"
                ],
                y=atm[
                    "days_to_expiry"
                ],
                z=atm[
                    "implied_volatility_pct"
                ],
                mode="markers",
                marker=dict(
                    size=7,
                    color="red",
                    symbol="diamond",
                ),
                name="ATM",
                hovertemplate=(
                    "<b>ATM</b>"
                    "<br>"
                    "Strike: %{x:.0f}"
                    "<br>"
                    "DTE: %{y}"
                    "<br>"
                    "IV: %{z:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        title=(
            f"{display_name} "
            f"Implied Volatility Surface"
        ),

        scene=dict(

            xaxis=dict(
                title="Strike",
                backgroundcolor="white",
            ),

            yaxis=dict(
                title="Days to Expiry",
                backgroundcolor="white",
            ),

            zaxis=dict(
                title="Implied Volatility (%)",
                backgroundcolor="white",
            ),

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.2,
                )
            ),
        ),

        template="plotly_white",

        height=750,

        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0,
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


# ============================================================
# SAVE JSON
# ============================================================

def save_surface_json(
    symbol: str,
    surface_df: pd.DataFrame,
    spot: float,
    atm_strike: float,
) -> Path:
    """
    Save machine-readable surface artifact.
    """

    expiry_strings = sorted(
        surface_df[
            "expiry"
        ]
        .dt
        .strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )

    output = {

        "symbol": symbol,

        "display_name": SYMBOLS[
            symbol
        ],

        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "spot": float(
            spot
        ),

        "atm_strike": float(
            atm_strike
        ),

        "observations": int(
            len(surface_df)
        ),

        "expiries": expiry_strings,

        "surface": [],
    }

    for _, row in surface_df.iterrows():

        output[
            "surface"
        ].append(

            {
                "expiry": row[
                    "expiry"
                ].strftime(
                    "%Y-%m-%d"
                ),

                "days_to_expiry": int(
                    row[
                        "days_to_expiry"
                    ]
                ),

                "strike": float(
                    row[
                        "strike"
                    ]
                ),

                "moneyness": float(
                    row[
                        "moneyness"
                    ]
                ),

                "log_moneyness": float(
                    row[
                        "log_moneyness"
                    ]
                ),

                "implied_volatility": float(
                    row[
                        "implied_volatility"
                    ]
                ),

                "implied_volatility_pct": float(
                    row[
                        "implied_volatility_pct"
                    ]
                ),

                "observations": int(
                    row[
                        "observations"
                    ]
                ),

                "is_atm": bool(
                    row[
                        "is_atm"
                    ]
                ),
            }
        )

    output_path = (
        OUTPUT_DIR
        / f"{symbol.lower()}_iv_surface.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    return output_path


# ============================================================
# RUN ONE SYMBOL
# ============================================================

def run_symbol(
    symbol: str,
) -> None:

    print()
    print(
        "=" * 70
    )

    print(
        f"{SYMBOLS[symbol]} "
        "IMPLIED VOLATILITY SURFACE"
    )

    print(
        "=" * 70
    )

    surface_df, spot, atm_strike = (
        build_surface_data(
            symbol
        )
    )

    if surface_df.empty:

        print(
            "No valid IV observations "
            "available for surface generation."
        )

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Spot: "
        f"{spot:.2f}"
    )

    print(
        f"ATM Strike: "
        f"{atm_strike:.2f}"
    )

    print(
        f"Surface observations: "
        f"{len(surface_df)}"
    )

    expiries = (
        surface_df[
            "expiry"
        ]
        .dt
        .strftime(
            "%Y-%m-%d"
        )
        .unique()
        .tolist()
    )

    print(
        f"Expiries: "
        f"{expiries}"
    )

    print(
        f"Strike range: "
        f"{surface_df['strike'].min():.0f}"
        f" → "
        f"{surface_df['strike'].max():.0f}"
    )

    print(
        f"IV range: "
        f"{surface_df['implied_volatility_pct'].min():.2f}%"
        f" → "
        f"{surface_df['implied_volatility_pct'].max():.2f}%"
    )

    # --------------------------------------------------------
    # Create Plotly figure
    # --------------------------------------------------------

    figure = create_surface_plot(
        symbol=symbol,
        surface_df=surface_df,
        spot=spot,
        atm_strike=atm_strike,
    )

    # --------------------------------------------------------
    # Save HTML
    # --------------------------------------------------------

    html_path = (
        OUTPUT_DIR
        / f"{symbol.lower()}_iv_surface.html"
    )

    figure.write_html(
        html_path,
        include_plotlyjs=True,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = save_surface_json(
        symbol=symbol,
        surface_df=surface_df,
        spot=spot,
        atm_strike=atm_strike,
    )

    print()
    print(
        f"HTML: {html_path}"
    )

    print(
        f"JSON: {json_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "QuantOS IMPLIED VOLATILITY "
        "SURFACE ENGINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Database: {DB_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Verify database before processing
    # --------------------------------------------------------

    if not DB_PATH.exists():

        raise FileNotFoundError(
            f"\nOptions database does not exist:\n"
            f"{DB_PATH}"
        )

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        latest = get_latest_snapshot_ts(
            conn
        )

    print(
        f"Latest option snapshot: "
        f"{latest}"
    )

    print()

    # --------------------------------------------------------
    # Build all three surfaces
    # --------------------------------------------------------

    for symbol in SYMBOLS:

        try:

            run_symbol(
                symbol
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{symbol}:"
            )

            print(
                repr(exc)
            )

            raise

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "IV SURFACE GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()