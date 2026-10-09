from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# QuantOS Volatility Intelligence Dashboard
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SURFACE_DIR = (
    ROOT
    / "data"
    / "regime"
    / "volatility_surface"
)


SYMBOLS = {
    "NIFTY_50": "NIFTY 50",
    "NIFTY_BANK": "NIFTY BANK",
    "SENSEX": "SENSEX",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="QuantOS Volatility Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .quantos-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }

    .quantos-subtitle {
        color: #667085;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 10px;
        padding: 15px 18px;
        min-height: 100px;
    }

    .metric-label {
        color: #667085;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #101828;
        font-size: 1.55rem;
        font-weight: 650;
        margin-top: 5px;
    }

    .metric-small {
        color: #667085;
        font-size: 0.75rem;
        margin-top: 3px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 650;
        color: #101828;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    .info-box {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):

    if not path.exists():
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception as exc:

        st.error(
            f"Unable to read {path.name}: {exc}"
        )

        return None


def load_market_surface(symbol: str):

    path = (
        SURFACE_DIR
        / f"{symbol.lower()}_iv_surface.json"
    )

    return load_json(path)


def load_sabr_surface(symbol: str):

    path = (
        SURFACE_DIR
        / f"{symbol.lower()}_sabr_surface.json"
    )

    return load_json(path)


def market_dataframe(data):

    if data is None:
        return pd.DataFrame()

    rows = data.get(
        "surface",
        [],
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    required = [
        "expiry",
        "days_to_expiry",
        "strike",
        "implied_volatility_pct",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        return pd.DataFrame()

    return df


def sabr_dataframe(data):

    if data is None:
        return pd.DataFrame()

    rows = data.get(
        "surface",
        [],
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    required = [
        "expiry",
        "days_to_expiry",
        "strike",
        "sabr_iv_pct",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        return pd.DataFrame()

    return df


def calibration_dataframe(data):

    if data is None:
        return pd.DataFrame()

    rows = data.get(
        "calibrations",
        [],
    )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# SURFACE PLOT
# ============================================================

def create_market_surface(
    df: pd.DataFrame,
    display_name: str,
    spot: float,
    atm_strike: float,
):

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=df["strike"],
            y=df["days_to_expiry"],
            z=df["implied_volatility_pct"],
            intensity=df["implied_volatility_pct"],
            colorscale="Viridis",
            opacity=0.90,
            colorbar=dict(
                title="IV (%)"
            ),
            name="Market IV",
            hovertemplate=(
                "<b>Strike:</b> %{x:.0f}"
                "<br><b>DTE:</b> %{y}"
                "<br><b>Market IV:</b> %{z:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    atm = df[
        np.isclose(
            df["strike"],
            atm_strike,
        )
    ]

    if not atm.empty:

        fig.add_trace(
            go.Scatter3d(
                x=atm["strike"],
                y=atm["days_to_expiry"],
                z=atm["implied_volatility_pct"],
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

    fig.update_layout(
        title=(
            f"{display_name} "
            "Market Implied Volatility Surface"
        ),
        template="plotly_white",
        height=700,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility (%)",
            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.15,
                )
            ),
        ),
        margin=dict(
            l=0,
            r=0,
            t=65,
            b=0,
        ),
    )

    return fig


def create_sabr_surface(
    df: pd.DataFrame,
    display_name: str,
    spot: float,
):

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=df["strike"],
            y=df["days_to_expiry"],
            z=df["sabr_iv_pct"],
            intensity=df["sabr_iv_pct"],
            colorscale="Plasma",
            opacity=0.90,
            colorbar=dict(
                title="SABR IV (%)"
            ),
            name="SABR",
            hovertemplate=(
                "<b>Strike:</b> %{x:.0f}"
                "<br><b>DTE:</b> %{y}"
                "<br><b>SABR IV:</b> %{z:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    # ATM reference line
    atm_rows = []

    for dte in sorted(
        df["days_to_expiry"].unique()
    ):

        slice_df = df[
            df["days_to_expiry"] == dte
        ]

        if slice_df.empty:
            continue

        nearest = slice_df.iloc[
            (
                slice_df["strike"] - spot
            ).abs().argmin()
        ]

        atm_rows.append(
            nearest
        )

    if atm_rows:

        atm_df = pd.DataFrame(
            atm_rows
        )

        fig.add_trace(
            go.Scatter3d(
                x=[spot] * len(atm_df),
                y=atm_df["days_to_expiry"],
                z=atm_df["sabr_iv_pct"],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="diamond",
                ),
                name="ATM",
            )
        )

    fig.update_layout(
        title=(
            f"{display_name} "
            "SABR-Fitted Volatility Surface"
        ),
        template="plotly_white",
        height=700,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="SABR Implied Volatility (%)",
            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.15,
                )
            ),
        ),
        margin=dict(
            l=0,
            r=0,
            t=65,
            b=0,
        ),
    )

    return fig


# ============================================================
# RESIDUAL SURFACE
# ============================================================

def create_residual_surface(
    market_df: pd.DataFrame,
    sabr_df: pd.DataFrame,
    display_name: str,
):

    rows = []

    common_expiries = sorted(
        set(
            market_df["expiry"]
        )
        &
        set(
            sabr_df["expiry"]
        )
    )

    for expiry in common_expiries:

        market_slice = market_df[
            market_df["expiry"] == expiry
        ].sort_values(
            "strike"
        )

        sabr_slice = sabr_df[
            sabr_df["expiry"] == expiry
        ].sort_values(
            "strike"
        )

        if (
            market_slice.empty
            or sabr_slice.empty
        ):
            continue

        sabr_strikes = (
            sabr_slice["strike"]
            .to_numpy()
        )

        sabr_iv = (
            sabr_slice["sabr_iv_pct"]
            .to_numpy()
        )

        for _, row in market_slice.iterrows():

            strike = float(
                row["strike"]
            )

            fitted = float(
                np.interp(
                    strike,
                    sabr_strikes,
                    sabr_iv,
                )
            )

            residual = (
                float(
                    row[
                        "implied_volatility_pct"
                    ]
                )
                - fitted
            )

            rows.append(
                {
                    "strike": strike,
                    "days_to_expiry": int(
                        row[
                            "days_to_expiry"
                        ]
                    ),
                    "residual": residual,
                }
            )

    residual_df = pd.DataFrame(
        rows
    )

    if residual_df.empty:
        return go.Figure()

    fig = go.Figure()

    max_abs = max(
        abs(
            residual_df[
                "residual"
            ]
        ).max(),
        0.01,
    )

    fig.add_trace(
        go.Mesh3d(
            x=residual_df["strike"],
            y=residual_df["days_to_expiry"],
            z=residual_df["residual"],
            intensity=residual_df["residual"],
            colorscale="RdBu",
            cmin=-max_abs,
            cmax=max_abs,
            cmid=0,
            opacity=0.92,
            colorbar=dict(
                title="Market − SABR (vol pts)"
            ),
            name="Residual",
            hovertemplate=(
                "<b>Strike:</b> %{x:.0f}"
                "<br><b>DTE:</b> %{y}"
                "<br><b>Residual:</b> %{z:.2f} vol pts"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"{display_name} "
            "Market vs SABR Residual Surface"
        ),
        template="plotly_white",
        height=700,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Market IV − SABR IV (vol pts)",
            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.15,
                )
            ),
        ),
        margin=dict(
            l=0,
            r=0,
            t=65,
            b=0,
        ),
    )

    return fig


# ============================================================
# IV SMILE
# ============================================================

def create_smile_chart(
    market_df: pd.DataFrame,
    sabr_df: pd.DataFrame,
    display_name: str,
):

    fig = go.Figure()

    expiries = sorted(
        set(
            market_df["expiry"]
        )
    )

    for expiry in expiries:

        market_slice = market_df[
            market_df["expiry"] == expiry
        ].sort_values(
            "strike"
        )

        if market_slice.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=market_slice["strike"],
                y=market_slice[
                    "implied_volatility_pct"
                ],
                mode="lines+markers",
                name=f"Market {expiry}",
                hovertemplate=(
                    "Strike: %{x:.0f}"
                    "<br>IV: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        sabr_slice = sabr_df[
            sabr_df["expiry"] == expiry
        ].sort_values(
            "strike"
        )

        if not sabr_slice.empty:

            fig.add_trace(
                go.Scatter(
                    x=sabr_slice["strike"],
                    y=sabr_slice[
                        "sabr_iv_pct"
                    ],
                    mode="lines",
                    line=dict(
                        dash="dash",
                    ),
                    name=f"SABR {expiry}",
                    hovertemplate=(
                        "Strike: %{x:.0f}"
                        "<br>SABR IV: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=(
            f"{display_name} "
            "IV Smile / Skew"
        ),
        template="plotly_white",
        height=480,
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# TERM STRUCTURE
# ============================================================

def create_term_structure(
    market_df: pd.DataFrame,
    sabr_df: pd.DataFrame,
    spot: float,
    display_name: str,
):

    market_points = []
    sabr_points = []

    for expiry in sorted(
        market_df["expiry"].unique()
    ):

        m = market_df[
            market_df["expiry"] == expiry
        ].copy()

        if m.empty:
            continue

        nearest = m.iloc[
            (
                m["strike"] - spot
            ).abs().argmin()
        ]

        market_points.append(
            {
                "expiry": expiry,
                "dte": int(
                    nearest[
                        "days_to_expiry"
                    ]
                ),
                "iv": float(
                    nearest[
                        "implied_volatility_pct"
                    ]
                ),
            }
        )

        s = sabr_df[
            sabr_df["expiry"] == expiry
        ].copy()

        if not s.empty:

            nearest_sabr = s.iloc[
                (
                    s["strike"] - spot
                ).abs().argmin()
            ]

            sabr_points.append(
                {
                    "expiry": expiry,
                    "dte": int(
                        nearest_sabr[
                            "days_to_expiry"
                        ]
                    ),
                    "iv": float(
                        nearest_sabr[
                            "sabr_iv_pct"
                        ]
                    ),
                }
            )

    market_term = pd.DataFrame(
        market_points
    )

    sabr_term = pd.DataFrame(
        sabr_points
    )

    fig = go.Figure()

    if not market_term.empty:

        fig.add_trace(
            go.Scatter(
                x=market_term["dte"],
                y=market_term["iv"],
                mode="lines+markers",
                name="Market ATM IV",
                hovertemplate=(
                    "DTE: %{x}"
                    "<br>ATM IV: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    if not sabr_term.empty:

        fig.add_trace(
            go.Scatter(
                x=sabr_term["dte"],
                y=sabr_term["iv"],
                mode="lines+markers",
                line=dict(
                    dash="dash",
                ),
                name="SABR ATM IV",
                hovertemplate=(
                    "DTE: %{x}"
                    "<br>SABR ATM IV: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"{display_name} "
            "ATM IV Term Structure"
        ),
        template="plotly_white",
        height=420,
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# METRICS
# ============================================================

def get_atm_iv(
    market_df,
    spot,
):

    if market_df.empty:
        return np.nan

    values = []

    for expiry in market_df[
        "expiry"
    ].unique():

        slice_df = market_df[
            market_df["expiry"] == expiry
        ]

        if slice_df.empty:
            continue

        nearest = slice_df.iloc[
            (
                slice_df["strike"] - spot
            ).abs().argmin()
        ]

        values.append(
            float(
                nearest[
                    "implied_volatility_pct"
                ]
            )
        )

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def get_calibration_rmse(
    calibration_df,
):

    if calibration_df.empty:
        return np.nan

    if "rmse" not in calibration_df:
        return np.nan

    return float(
        calibration_df["rmse"].mean()
        * 100.0
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "QuantOS Volatility"
)

selected_symbol = st.sidebar.selectbox(
    "Underlying",
    list(SYMBOLS.keys()),
    format_func=lambda x: SYMBOLS[x],
)

view = st.sidebar.radio(
    "3D Surface",
    [
        "Market IV",
        "SABR Fitted",
        "Market − SABR",
    ],
)

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "Interactive quantitative volatility research"
)

st.sidebar.caption(
    "Source: QuantOS live option feed"
)


# ============================================================
# LOAD DATA
# ============================================================

market_data = load_market_surface(
    selected_symbol
)

sabr_data = load_sabr_surface(
    selected_symbol
)

market_df = market_dataframe(
    market_data
)

sabr_df = sabr_dataframe(
    sabr_data
)

calibration_df = calibration_dataframe(
    sabr_data
)


if market_data is None:

    st.error(
        "Market IV surface data not found."
    )

    st.stop()


if market_df.empty:

    st.error(
        "No valid market IV surface observations found."
    )

    st.stop()


display_name = SYMBOLS[
    selected_symbol
]

spot = float(
    market_data.get(
        "spot",
        np.nan,
    )
)

atm_strike = float(
    market_data.get(
        "atm_strike",
        np.nan,
    )
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="quantos-title">'
    'QuantOS Volatility Intelligence'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="quantos-subtitle">'
    'Live implied volatility surface, SABR calibration, '
    'smile/skew and term-structure analytics'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

atm_iv = get_atm_iv(
    market_df,
    spot,
)

avg_rmse = get_calibration_rmse(
    calibration_df
)

expiries = sorted(
    market_df["expiry"].unique()
)

c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Underlying
            </div>
            <div class="metric-value">
                {display_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Spot
            </div>
            <div class="metric-value">
                {spot:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                ATM Strike
            </div>
            <div class="metric-value">
                {atm_strike:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Average ATM IV
            </div>
            <div class="metric-value">
                {atm_iv:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c5:

    rmse_text = (
        f"{avg_rmse:.2f}%"
        if np.isfinite(avg_rmse)
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                SABR Avg RMSE
            </div>
            <div class="metric-value">
                {rmse_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 3D SURFACE
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Interactive 3D Volatility Surface'
    '</div>',
    unsafe_allow_html=True,
)


if view == "Market IV":

    fig = create_market_surface(
        market_df,
        display_name,
        spot,
        atm_strike,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


elif view == "SABR Fitted":

    if sabr_df.empty:

        st.warning(
            "SABR surface data is unavailable."
        )

    else:

        fig = create_sabr_surface(
            sabr_df,
            display_name,
            spot,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )


else:

    if sabr_df.empty:

        st.warning(
            "SABR surface data is unavailable."
        )

    else:

        fig = create_residual_surface(
            market_df,
            sabr_df,
            display_name,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )


# ============================================================
# MODEL SUMMARY
# ============================================================

if not calibration_df.empty:

    st.markdown(
        '<div class="section-title">'
        'SABR Calibration'
        '</div>',
        unsafe_allow_html=True,
    )

    display_calibration = calibration_df.copy()

    display_calibration[
        "rmse_pct"
    ] = (
        display_calibration[
            "rmse"
        ]
        * 100
    )

    display_calibration = (
        display_calibration[
            [
                "expiry",
                "days_to_expiry",
                "forward",
                "alpha",
                "beta",
                "rho",
                "nu",
                "points",
                "rmse_pct",
            ]
        ]
        .rename(
            columns={
                "expiry": "Expiry",
                "days_to_expiry": "DTE",
                "forward": "Forward",
                "alpha": "Alpha",
                "beta": "Beta",
                "rho": "Rho",
                "nu": "Nu",
                "points": "Points",
                "rmse_pct": "RMSE (%)",
            }
        )
    )

    st.dataframe(
        display_calibration,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Forward": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Alpha": st.column_config.NumberColumn(
                format="%.5f"
            ),
            "Beta": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Rho": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Nu": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "RMSE (%)": st.column_config.NumberColumn(
                format="%.3f"
            ),
        },
    )


# ============================================================
# SMILE / SKEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'IV Smile / Skew'
    '</div>',
    unsafe_allow_html=True,
)

if not sabr_df.empty:

    smile_fig = create_smile_chart(
        market_df,
        sabr_df,
        display_name,
    )

else:

    smile_fig = create_smile_chart(
        market_df,
        pd.DataFrame(
            columns=[
                "expiry",
                "strike",
                "sabr_iv_pct",
            ]
        ),
        display_name,
    )

st.plotly_chart(
    smile_fig,
    use_container_width=True,
    config={
        "displaylogo": False,
    },
)


# ============================================================
# TERM STRUCTURE
# ============================================================

st.markdown(
    '<div class="section-title">'
    'ATM IV Term Structure'
    '</div>',
    unsafe_allow_html=True,
)

term_fig = create_term_structure(
    market_df,
    sabr_df,
    spot,
    display_name,
)

st.plotly_chart(
    term_fig,
    use_container_width=True,
    config={
        "displaylogo": False,
    },
)


# ============================================================
# MARKET SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Surface Summary'
    '</div>',
    unsafe_allow_html=True,
)

iv_min = float(
    market_df[
        "implied_volatility_pct"
    ].min()
)

iv_max = float(
    market_df[
        "implied_volatility_pct"
    ].max()
)

strike_min = float(
    market_df[
        "strike"
    ].min()
)

strike_max = float(
    market_df[
        "strike"
    ].max()
)

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.markdown(
        f"""
        <div class="info-box">

        <b>Market Surface</b><br><br>

        Expiries: {len(expiries)}<br>
        Surface observations: {len(market_df)}<br>
        Strike range: {strike_min:,.0f}
        → {strike_max:,.0f}<br>
        IV range: {iv_min:.2f}%
        → {iv_max:.2f}%

        </div>
        """,
        unsafe_allow_html=True,
    )


with summary_col2:

    if not calibration_df.empty:

        st.markdown(
            f"""
            <div class="info-box">

            <b>SABR Model</b><br><br>

            Calibrated expiries:
            {len(calibration_df)}<br>
            Calibration points:
            {int(calibration_df["points"].sum())}<br>
            Average RMSE:
            {avg_rmse:.3f}%<br>
            Fixed β:
            {calibration_df["beta"].iloc[0]:.2f}

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="info-box">
            <b>SABR Model</b><br><br>
            Calibration data unavailable.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "QuantOS Volatility Methodology"
):

    st.markdown(
        """
        **Market Surface**

        The observed volatility surface is constructed from
        live option-chain implied volatilities across strike
        and expiry.

        **SABR Calibration**

        SABR is calibrated separately for each available
        expiry using observed option implied volatilities.

        β is fixed at 1.0 while α, ρ and ν are calibrated.

        **Residual Surface**

        The residual is defined as:

        `Market IV − SABR IV`

        Positive values indicate that observed market
        implied volatility is above the fitted SABR surface,
        while negative values indicate the opposite.

        **Interpretation**

        The surface describes the cross-sectional structure
        of implied volatility. It is a quantitative market
        intelligence layer rather than a trading recommendation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#98A2B3;
        font-size:0.75rem;
        margin-top:30px;
    ">
        QuantOS · Quantitative Market Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)