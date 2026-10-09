from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import least_squares


# ============================================================
# QuantOS SABR IMPLIED VOLATILITY SURFACE
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


# ------------------------------------------------------------
# SABR configuration
# ------------------------------------------------------------

# Standard SABR choice for equity-index volatility:
# beta is fixed and alpha/rho/nu are calibrated.
#
# beta = 1.0 corresponds to lognormal SABR.
# We keep beta fixed to avoid over-parameterizing the
# relatively small option chains available here.
#
BETA = 1.0

MIN_IV = 0.001
MAX_IV = 3.0

MAX_MONEYNESS = 0.12

MIN_CALIBRATION_POINTS = 5


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}"
        )

    return sqlite3.connect(DB_PATH)


def get_latest_snapshot_ts(conn):
    row = conn.execute(
        """
        SELECT MAX(snapshot_ts)
        FROM option_snapshots
        """
    ).fetchone()

    if row is None or row[0] is None:
        return None

    return str(row[0])


def load_latest_snapshot(symbol):
    conn = get_connection()

    try:

        latest_ts = get_latest_snapshot_ts(conn)

        if latest_ts is None:
            return pd.DataFrame()

        query = """
            SELECT
                snapshot_ts,
                expiry,
                strike,
                option_type,
                ltp,
                bid,
                ask,
                volume,
                oi,
                iv
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

    return normalize_data(df)


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_data(df):

    if df.empty:
        return df

    df = df.copy()

    df["snapshot_ts"] = pd.to_datetime(
        df["snapshot_ts"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    df["expiry"] = pd.to_datetime(
        df["expiry"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    numeric_columns = [
        "strike",
        "ltp",
        "bid",
        "ask",
        "volume",
        "oi",
        "iv",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df = df[
        df["option_type"].isin(
            ["CE", "PE"]
        )
    ]

    df = df.dropna(
        subset=[
            "strike",
            "expiry",
            "iv",
        ]
    )

    df = df[
        (df["iv"] >= MIN_IV)
        &
        (df["iv"] <= MAX_IV)
    ]

    return df.reset_index(drop=True)


# ============================================================
# SPOT
# ============================================================

def get_spot(df):

    try:

        from quantos_volatility.option_analytics import (
            get_spot_price,
        )

        spot = get_spot_price(df)

        if spot is not None:
            spot = float(spot)

            if np.isfinite(spot) and spot > 0:
                return spot

    except Exception:
        pass

    # --------------------------------------------------------
    # Put-call parity fallback
    # --------------------------------------------------------

    expiry = sorted(
        df["expiry"].unique()
    )[0]

    chain = df[
        df["expiry"] == expiry
    ]

    calls = chain[
        chain["option_type"] == "CE"
    ][
        ["strike", "ltp"]
    ].rename(
        columns={
            "ltp": "call_ltp"
        }
    )

    puts = chain[
        chain["option_type"] == "PE"
    ][
        ["strike", "ltp"]
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

    pairs["synthetic_spot"] = (
        pairs["strike"]
        + pairs["call_ltp"]
        - pairs["put_ltp"]
    )

    return float(
        pairs["synthetic_spot"]
        .median()
    )


# ============================================================
# SABR FORMULA
# ============================================================

def sabr_implied_volatility(
    F,
    K,
    T,
    alpha,
    beta,
    rho,
    nu,
):
    """
    Hagan SABR implied-volatility approximation.

    F = forward
    K = strike
    T = time to expiry
    alpha = volatility scale
    beta = CEV exponent
    rho = correlation
    nu = volatility-of-volatility
    """

    F = float(F)
    K = float(K)
    T = float(T)

    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    nu = float(nu)

    if (
        F <= 0
        or K <= 0
        or T <= 0
        or alpha <= 0
        or nu <= 0
    ):
        return np.nan

    FK = F * K

    # ATM case
    if abs(F - K) < 1e-10:

        term1 = (
            alpha
            / (F ** (1.0 - beta))
        )

        correction = (
            1.0
            + (
                (
                    ((1.0 - beta) ** 2 / 24.0)
                    * (alpha ** 2)
                    / (F ** (2.0 - 2.0 * beta))
                )
                +
                (
                    rho
                    * beta
                    * nu
                    * alpha
                    / (
                        4.0
                        * F ** (1.0 - beta)
                    )
                )
                +
                (
                    (2.0 - 3.0 * rho ** 2)
                    * nu ** 2
                    / 24.0
                )
            )
            * T
        )

        return term1 * correction

    log_fk = np.log(F / K)

    z = (
        (nu / alpha)
        * (FK ** ((1.0 - beta) / 2.0))
        * log_fk
    )

    sqrt_term = np.sqrt(
        max(
            1e-12,
            1.0
            - 2.0 * rho * z
            + z * z,
        )
    )

    x_z = np.log(
        (
            sqrt_term
            + z
            - rho
        )
        /
        (1.0 - rho)
    )

    if abs(x_z) < 1e-12:
        z_over_xz = 1.0
    else:
        z_over_xz = z / x_z

    numerator = alpha

    denominator = (
        FK ** ((1.0 - beta) / 2.0)
        *
        (
            1.0
            + (
                ((1.0 - beta) ** 2 / 24.0)
                * log_fk ** 2
            )
            + (
                ((1.0 - beta) ** 4 / 1920.0)
                * log_fk ** 4
            )
        )
    )

    correction = (
        1.0
        + (
            (
                ((1.0 - beta) ** 2 / 24.0)
                * alpha ** 2
                / FK ** (1.0 - beta)
            )
            +
            (
                rho
                * beta
                * nu
                * alpha
                / (
                    4.0
                    * FK ** (
                        (1.0 - beta) / 2.0
                    )
                )
            )
            +
            (
                (2.0 - 3.0 * rho ** 2)
                * nu ** 2
                / 24.0
            )
        )
        * T
    )

    return (
        numerator
        / denominator
        * z_over_xz
        * correction
    )


# ============================================================
# FORWARD ESTIMATION
# ============================================================

def estimate_forward(
    df,
    spot,
    expiry,
):
    """
    Estimate forward using put-call parity:

        F ≈ K + C - P

    Median across available strike pairs.
    """

    chain = df[
        df["expiry"] == expiry
    ]

    calls = chain[
        chain["option_type"] == "CE"
    ][
        ["strike", "ltp"]
    ].rename(
        columns={
            "ltp": "call_ltp"
        }
    )

    puts = chain[
        chain["option_type"] == "PE"
    ][
        ["strike", "ltp"]
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
        return spot

    pairs = pairs.dropna(
        subset=[
            "call_ltp",
            "put_ltp",
        ]
    )

    if pairs.empty:
        return spot

    pairs["forward"] = (
        pairs["strike"]
        + pairs["call_ltp"]
        - pairs["put_ltp"]
    )

    values = pairs[
        "forward"
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if values.empty:
        return spot

    # Use a robust median.
    forward = float(
        values.median()
    )

    # Reject obviously pathological estimates.
    if (
        forward <= 0
        or forward < spot * 0.90
        or forward > spot * 1.10
    ):
        return spot

    return forward


# ============================================================
# SABR CALIBRATION
# ============================================================

def calibrate_sabr(
    forward,
    strikes,
    market_iv,
    T,
):
    """
    Calibrate alpha, rho and nu.

    beta is fixed.
    """

    strikes = np.asarray(
        strikes,
        dtype=float,
    )

    market_iv = np.asarray(
        market_iv,
        dtype=float,
    )

    valid = (
        np.isfinite(strikes)
        &
        np.isfinite(market_iv)
        &
        (strikes > 0)
        &
        (market_iv > 0)
    )

    strikes = strikes[valid]
    market_iv = market_iv[valid]

    if len(strikes) < MIN_CALIBRATION_POINTS:
        return None

    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    atm_index = np.argmin(
        np.abs(
            strikes - forward
        )
    )

    atm_iv = float(
        market_iv[atm_index]
    )

    alpha0 = (
        atm_iv
        * forward ** (1.0 - BETA)
    )

    # Parameterization:
    #
    # alpha > 0
    # rho ∈ (-1,1)
    # nu > 0
    #
    # Bounds:
    # alpha
    # rho
    # nu
    # --------------------------------------------------------

    lower = np.array([
        1e-5,
        -0.999,
        1e-5,
    ])

    upper = np.array([
        5.0,
        0.999,
        5.0,
    ])

    x0 = np.array([
        max(
            1e-5,
            alpha0,
        ),
        0.0,
        0.5,
    ])

    def residuals(params):

        alpha, rho, nu = params

        model_iv = np.array(
            [
                sabr_implied_volatility(
                    F=forward,
                    K=K,
                    T=T,
                    alpha=alpha,
                    beta=BETA,
                    rho=rho,
                    nu=nu,
                )
                for K in strikes
            ]
        )

        invalid = (
            ~np.isfinite(
                model_iv
            )
        )

        if invalid.any():

            model_iv[
                invalid
            ] = 10.0

        return (
            model_iv
            - market_iv
        )

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(
            lower,
            upper,
        ),
        max_nfev=3000,
        loss="soft_l1",
    )

    if not result.success:
        return None

    alpha, rho, nu = result.x

    fitted = np.array(
        [
            sabr_implied_volatility(
                F=forward,
                K=K,
                T=T,
                alpha=alpha,
                beta=BETA,
                rho=rho,
                nu=nu,
            )
            for K in strikes
        ]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    fitted
                    - market_iv
                ) ** 2
            )
        )
    )

    return {
        "alpha": float(alpha),
        "beta": float(BETA),
        "rho": float(rho),
        "nu": float(nu),
        "rmse": rmse,
        "points": int(len(strikes)),
    }


# ============================================================
# BUILD SABR SURFACE
# ============================================================

def build_sabr_surface(
    symbol,
):
    df = load_latest_snapshot(
        symbol
    )

    if df.empty:
        return None

    spot = get_spot(
        df
    )

    if spot is None:
        raise RuntimeError(
            f"Unable to determine spot for {symbol}"
        )

    reference_date = (
        df["snapshot_ts"]
        .dropna()
        .max()
        .normalize()
    )

    expiries = sorted(
        df["expiry"].unique()
    )

    calibrated = []

    surface_rows = []

    for expiry in expiries:

        chain = df[
            df["expiry"] == expiry
        ].copy()

        if chain.empty:
            continue

        days = int(
            (
                expiry.normalize()
                - reference_date
            ).days
        )

        if days <= 0:
            continue

        T = days / 365.0

        forward = estimate_forward(
            chain,
            spot,
            expiry,
        )

        # ----------------------------------------------------
        # Aggregate CE + PE IV by strike
        # ----------------------------------------------------

        market = (
            chain.groupby(
                "strike",
                as_index=False,
            )
            .agg(
                market_iv=(
                    "iv",
                    "median",
                )
            )
        )

        market["moneyness"] = (
            market["strike"]
            / spot
            - 1.0
        )

        market = market[
            market["moneyness"].abs()
            <= MAX_MONEYNESS
        ]

        market = market.dropna(
            subset=[
                "strike",
                "market_iv",
            ]
        )

        if len(market) < MIN_CALIBRATION_POINTS:
            continue

        # ----------------------------------------------------
        # Calibrate
        # ----------------------------------------------------

        calibration = calibrate_sabr(
            forward=forward,
            strikes=market[
                "strike"
            ].values,
            market_iv=market[
                "market_iv"
            ].values,
            T=T,
        )

        if calibration is None:
            continue

        calibration.update(
            {
                "expiry": expiry.strftime(
                    "%Y-%m-%d"
                ),
                "days_to_expiry": days,
                "forward": float(forward),
            }
        )

        calibrated.append(
            calibration
        )

        # ----------------------------------------------------
        # Create dense strike grid
        # ----------------------------------------------------

        strike_min = float(
            market["strike"].min()
        )

        strike_max = float(
            market["strike"].max()
        )

        grid = np.linspace(
            strike_min,
            strike_max,
            60,
        )

        sabr_iv = np.array(
            [
                sabr_implied_volatility(
                    F=forward,
                    K=K,
                    T=T,
                    alpha=calibration[
                        "alpha"
                    ],
                    beta=BETA,
                    rho=calibration[
                        "rho"
                    ],
                    nu=calibration[
                        "nu"
                    ],
                )
                for K in grid
            ]
        )

        for K, iv in zip(
            grid,
            sabr_iv,
        ):

            if np.isfinite(iv):

                surface_rows.append(
                    {
                        "expiry": expiry.strftime(
                            "%Y-%m-%d"
                        ),
                        "days_to_expiry": days,
                        "strike": float(K),
                        "sabr_iv": float(iv),
                        "sabr_iv_pct": float(
                            iv * 100.0
                        ),
                        "moneyness": float(
                            K / spot - 1.0
                        ),
                    }
                )

    if not calibrated:
        raise RuntimeError(
            f"No SABR calibration succeeded for {symbol}"
        )

    return {
        "symbol": symbol,
        "display_name": SYMBOLS[
            symbol
        ],
        "spot": float(spot),
        "calibrations": calibrated,
        "surface": surface_rows,
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(result):

    symbol = result[
        "symbol"
    ]

    path = (
        OUTPUT_DIR
        / f"{symbol.lower()}_sabr_surface.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    return path


# ============================================================
# CREATE 3D SABR PLOT
# ============================================================

def create_3d_plot(
    result,
):

    surface = pd.DataFrame(
        result[
            "surface"
        ]
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # SABR surface
    # --------------------------------------------------------

    fig.add_trace(
        go.Mesh3d(
            x=surface[
                "strike"
            ],
            y=surface[
                "days_to_expiry"
            ],
            z=surface[
                "sabr_iv_pct"
            ],
            intensity=surface[
                "sabr_iv_pct"
            ],
            colorscale="Plasma",
            opacity=0.78,
            colorbar=dict(
                title="SABR IV (%)"
            ),
            name="SABR Surface",
            hovertemplate=(
                "<b>Strike:</b> %{x:.0f}"
                "<br>"
                "<b>DTE:</b> %{y}"
                "<br>"
                "<b>SABR IV:</b> %{z:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------------
    # ATM reference
    # --------------------------------------------------------

    spot = result[
        "spot"
    ]

    atm_points = []

    for calibration in result[
        "calibrations"
    ]:

        iv = sabr_implied_volatility(
            F=calibration[
                "forward"
            ],
            K=spot,
            T=calibration[
                "days_to_expiry"
            ] / 365.0,
            alpha=calibration[
                "alpha"
            ],
            beta=BETA,
            rho=calibration[
                "rho"
            ],
            nu=calibration[
                "nu"
            ],
        )

        if np.isfinite(iv):

            atm_points.append(
                {
                    "dte": calibration[
                        "days_to_expiry"
                    ],
                    "iv": iv * 100.0,
                }
            )

    if atm_points:

        atm_df = pd.DataFrame(
            atm_points
        )

        fig.add_trace(
            go.Scatter3d(
                x=[
                    spot
                ]
                * len(atm_df),
                y=atm_df[
                    "dte"
                ],
                z=atm_df[
                    "iv"
                ],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="diamond",
                ),
                name="ATM",
                hovertemplate=(
                    "<b>ATM</b>"
                    "<br>Strike: %{x:.0f}"
                    "<br>DTE: %{y}"
                    "<br>IV: %{z:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        title=(
            f"{result['display_name']} "
            "SABR Implied Volatility Surface"
        ),
        template="plotly_white",
        height=800,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="SABR Implied Volatility (%)",
            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.2,
                )
            ),
        ),
        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0,
        ),
    )

    return fig


# ============================================================
# RUN
# ============================================================

def run_symbol(
    symbol,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{SYMBOLS[symbol]} SABR CALIBRATION"
    )

    print(
        "=" * 70
    )

    result = build_sabr_surface(
        symbol
    )

    print(
        f"Spot: {result['spot']:.2f}"
    )

    print()

    for calibration in result[
        "calibrations"
    ]:

        print(
            f"Expiry: "
            f"{calibration['expiry']}"
        )

        print(
            f"  DTE: "
            f"{calibration['days_to_expiry']}"
        )

        print(
            f"  Forward: "
            f"{calibration['forward']:.2f}"
        )

        print(
            f"  Alpha: "
            f"{calibration['alpha']:.6f}"
        )

        print(
            f"  Beta: "
            f"{calibration['beta']:.2f}"
        )

        print(
            f"  Rho: "
            f"{calibration['rho']:.4f}"
        )

        print(
            f"  Nu: "
            f"{calibration['nu']:.4f}"
        )

        print(
            f"  Calibration RMSE: "
            f"{calibration['rmse'] * 100:.4f}%"
        )

        print(
            f"  Points: "
            f"{calibration['points']}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = save_json(
        result
    )

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    fig = create_3d_plot(
        result
    )

    html_path = (
        OUTPUT_DIR
        / f"{symbol.lower()}_sabr_surface.html"
    )

    fig.write_html(
        html_path,
        include_plotlyjs=True,
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
        "QuantOS SABR VOLATILITY SURFACE ENGINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Database: {DB_PATH}"
    )

    for symbol in SYMBOLS:

        run_symbol(
            symbol
        )

    print()
    print(
        "=" * 70
    )

    print(
        "SABR SURFACE GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()