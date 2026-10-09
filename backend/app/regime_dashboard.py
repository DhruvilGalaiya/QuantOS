"""
QuantOS | Live V10 Regime Dashboard - FINAL

Reads:
    data/regime/live_predictions/latest_predictions.json

It does NOT train the HMM.
It does NOT query PostgreSQL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "regime"
    / "live_predictions"
    / "latest_predictions.json"
)

SYMBOLS = [
    "NIFTY_50",
    "NIFTY_BANK",
    "NIFTY_IT",
    "NIFTY_PHARMA",
    "NIFTY_AUTO",
    "NIFTY_FIN_SERVICE",
    "INDIA_VIX",
    "SENSEX",
]

DISPLAY_NAMES = {
    "NIFTY_50": "NIFTY 50",
    "NIFTY_BANK": "NIFTY BANK",
    "NIFTY_IT": "NIFTY IT",
    "NIFTY_PHARMA": "NIFTY PHARMA",
    "NIFTY_AUTO": "NIFTY AUTO",
    "NIFTY_FIN_SERVICE": "NIFTY FIN SERVICE",
    "INDIA_VIX": "INDIA VIX",
    "SENSEX": "SENSEX",
}

MARKET_STATES = ["BULL", "SIDE", "BEAR"]
VIX_STATES = ["LOW", "MID", "HIGH"]

# Stock layer: deliberately separate from the market-index HMM selector.
STOCK_SYMBOLS = [
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "NFLX",
]

STOCK_DISPLAY_NAMES = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "AVGO": "Broadcom",
    "TSLA": "Tesla",
    "NFLX": "Netflix",
}


st.set_page_config(
    page_title="QuantOS | Market Regime Intelligence",
    page_icon="◈",
    layout="wide",
)

# ============================================================
# ============================================================
# HISTORICAL PREDICTION HISTORY
# ============================================================

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "regime"
    / "live_predictions"
)


# ============================================================
# LOAD
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found: {PREDICTION_FILE}"
        )

    with PREDICTION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "latest_predictions.json is not a dictionary."
        )

    return data


def history_file_for_symbol(symbol: str):
    """
    Find the instrument-specific historical prediction file.

    Example:
        NIFTY_50 -> prediction_history_nifty_50.csv
        INDIA_VIX -> prediction_history_india_vix.csv

    The directory is scanned rather than relying on a hard-coded
    filename so the dashboard cannot accidentally show NIFTY 50
    history while another instrument is selected.
    """

    target = str(symbol).lower().replace("_", "")

    if not HISTORY_DIR.exists():
        return None

    candidates = HISTORY_DIR.glob("prediction_history_*.csv")

    for path in candidates:
        stem = path.stem.removeprefix("prediction_history_")
        normalized = stem.lower().replace("_", "")

        if normalized == target:
            return path

    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_hmm_prediction_history(symbol: str):
    history_file = history_file_for_symbol(symbol)

    if history_file is None:
        return pd.DataFrame()

    df = pd.read_csv(history_file)

    if df.empty:
        return df

    for col in [
        "prediction_date",
        "target_trading_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            )

    # Keep the most recent two years visible in the dashboard.
    # The underlying CSV remains the full available history.
    cutoff = (
        pd.Timestamp.now().normalize()
        - pd.DateOffset(years=2)
    )

    if "prediction_date" in df.columns:
        df = df[
            df["prediction_date"] >= cutoff
        ].copy()

    df = df.sort_values(
        "prediction_date"
    ).reset_index(drop=True)

    return df


# HELPERS
# ============================================================

def first_value(
    data: dict[str, Any],
    keys: list[str],
    default=None,
):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    summary = data.get("summary")

    if isinstance(summary, dict):
        for key in keys:
            if key in summary and summary[key] is not None:
                return summary[key]

    return default


def as_float(value, default=np.nan):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def display_regime(
    symbol: str,
    regime: str,
):
    regime = str(regime).upper()

    if symbol == "INDIA_VIX":
        return {
            "BEAR": "LOW",
            "SIDE": "MID",
            "BULL": "HIGH",
        }.get(regime, regime)

    return regime


def display_states(symbol: str):
    if symbol == "INDIA_VIX":
        return VIX_STATES

    return MARKET_STATES


def get_streak(data):
    # IMPORTANT: V10's actual field is current_regime_streak.
    value = first_value(
        data,
        [
            "current_regime_streak",
            "current_streak",
            "current_streak_days",
            "streak",
            "regime_streak",
        ],
    )

    if value is not None:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            pass

    daily = data.get("daily_output")

    if isinstance(daily, list) and daily:
        row = daily[-1]

        if isinstance(row, dict):
            value = first_value(
                row,
                [
                    "current_regime_streak",
                    "current_streak",
                    "current_streak_days",
                    "streak",
                    "regime_streak",
                ],
            )

            if value is not None:
                try:
                    return int(round(float(value)))
                except (TypeError, ValueError):
                    pass

    return 0


def get_stickiness(data):
    value = first_value(
        data,
        [
            "stickiness_score",
            "stickiness",
            "stickiness_score",
        ],
    )

    if value is not None:
        return as_float(value)

    daily = data.get("daily_output")

    if isinstance(daily, list) and daily:
        row = daily[-1]

        if isinstance(row, dict):
            return as_float(
                first_value(
                    row,
                    [
                        "stickiness_score",
                        "stickiness",
                    ],
                )
            )

    return np.nan


def get_stickiness_level(
    data,
    score,
):
    value = first_value(
        data,
        [
            "stickiness_level",
            "stickiness_label",
            "level",
        ],
    )

    if value is not None:
        return str(value).upper()

    if np.isfinite(score):
        if score >= 75:
            return "HIGH"
        if score >= 50:
            return "MODERATE"
        return "LOW"

    return "N/A"


def get_probabilities(data):
    value = first_value(
        data,
        [
            "probabilities",
            "next_day_probabilities",
            "next_probabilities",
        ],
        {},
    )

    if not isinstance(value, dict):
        return {}

    result = {}

    for key, value in value.items():
        x = as_float(value)

        if np.isfinite(x):
            result[str(key).upper()] = max(0.0, x)

    total = sum(result.values())

    if total > 0:
        result = {
            k: v / total
            for k, v in result.items()
        }

    return result


def get_transition_matrix(data):
    """
    Handles both:
    1. final runner's wide split-format matrix
    2. older long-format matrix_output
    """

    value = data.get("transition_matrix")

    # Wide split format.
    if isinstance(value, dict):
        if {
            "index",
            "columns",
            "data",
        }.issubset(value.keys()):
            try:
                matrix = pd.DataFrame(
                    value["data"],
                    index=value["index"],
                    columns=value["columns"],
                )

                matrix.index = [
                    str(x).upper()
                    for x in matrix.index
                ]

                matrix.columns = [
                    str(x).upper()
                    for x in matrix.columns
                ]

                matrix = matrix.apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                states = MARKET_STATES

                if set(states).issubset(matrix.index) and set(states).issubset(matrix.columns):
                    return matrix.loc[
                        states,
                        states,
                    ]

                return matrix
            except Exception:
                pass

    # Long format fallback.
    records = data.get("matrix_output")

    if isinstance(records, list) and records:
        try:
            df = pd.DataFrame(records)

            current_col = next(
                (
                    c for c in df.columns
                    if str(c).lower()
                    in [
                        "current_regime",
                        "current_state",
                    ]
                ),
                None,
            )

            next_col = next(
                (
                    c for c in df.columns
                    if str(c).lower()
                    in [
                        "next_regime",
                        "next_state",
                    ]
                ),
                None,
            )

            probability_col = next(
                (
                    c for c in df.columns
                    if str(c).lower()
                    in [
                        "probability",
                        "transition_probability",
                    ]
                ),
                None,
            )

            if (
                current_col is None
                or next_col is None
                or probability_col is None
            ):
                return None

            matrix = pd.DataFrame(
                0.0,
                index=MARKET_STATES,
                columns=MARKET_STATES,
            )

            for _, row in df.iterrows():
                current = str(
                    row[current_col]
                ).upper()

                nxt = str(
                    row[next_col]
                ).upper()

                probability = as_float(
                    row[probability_col]
                )

                if (
                    current in MARKET_STATES
                    and nxt in MARKET_STATES
                    and np.isfinite(probability)
                ):
                    matrix.loc[
                        current,
                        nxt,
                    ] = probability

            return matrix

        except Exception:
            pass

    return None


def get_stationary(data):
    value = first_value(
        data,
        [
            "stationary_distribution",
            "long_run_distribution",
            "longrun_distribution",
            "stationary",
        ],
        {},
    )

    if isinstance(value, dict):
        result = {}

        for key, value in value.items():
            x = as_float(value)

            if np.isfinite(x):
                result[str(key).upper()] = max(
                    0.0,
                    x,
                )

        total = sum(result.values())

        if total > 0:
            return {
                k: v / total
                for k, v in result.items()
            }

    records = data.get(
        "stationary_output"
    )

    if isinstance(records, list) and records:
        try:
            df = pd.DataFrame(records)

            regime_col = next(
                (
                    c for c in df.columns
                    if str(c).lower()
                    in [
                        "regime",
                        "state",
                        "label",
                    ]
                ),
                None,
            )

            probability_col = next(
                (
                    c for c in df.columns
                    if str(c).lower()
                    in [
                        "long_run_probability",
                        "stationary_probability",
                        "probability",
                        "share",
                    ]
                ),
                None,
            )

            if (
                regime_col is None
                or probability_col is None
            ):
                return {}

            result = {}

            for _, row in df.iterrows():
                regime = str(
                    row[regime_col]
                ).upper()

                x = as_float(
                    row[probability_col]
                )

                if np.isfinite(x):
                    result[regime] = max(
                        0.0,
                        x,
                    )

            total = sum(result.values())

            if total > 0:
                return {
                    k: v / total
                    for k, v in result.items()
                }

        except Exception:
            pass

    return {}


def remap_vix_probs(probs):
    return {
        "LOW": probs.get("BEAR", 0.0),
        "MID": probs.get("SIDE", 0.0),
        "HIGH": probs.get("BULL", 0.0),
    }


def remap_vix_matrix(matrix):
    if matrix is None:
        return None

    if not set(MARKET_STATES).issubset(
        matrix.index
    ):
        return matrix

    if not set(MARKET_STATES).issubset(
        matrix.columns
    ):
        return matrix

    # Underlying:
    # BEAR -> LOW
    # SIDE -> MID
    # BULL -> HIGH
    order = [
        "BEAR",
        "SIDE",
        "BULL",
    ]

    result = matrix.loc[
        order,
        order,
    ].copy()

    result.index = VIX_STATES
    result.columns = VIX_STATES

    return result


def build_cross_index(data):
    rows = []

    for symbol in SYMBOLS:
        payload = data.get(symbol)

        if not isinstance(payload, dict):
            continue

        current = str(
            first_value(
                payload,
                [
                    "current_regime",
                    "current_state",
                    "regime",
                ],
                "N/A",
            )
        ).upper()

        nxt = str(
            first_value(
                payload,
                [
                    "next_regime",
                    "next_state",
                    "forecast_regime",
                ],
                "N/A",
            )
        ).upper()

        probabilities = get_probabilities(
            payload
        )

        next_probability = as_float(
            first_value(
                payload,
                [
                    "next_day_probability",
                    "next_probability",
                ],
            )
        )

        if (
            not np.isfinite(next_probability)
            and probabilities
        ):
            next_probability = max(
                probabilities.values()
            )

        stickiness = get_stickiness(
            payload
        )

        streak = get_streak(
            payload
        )

        rows.append(
            {
                "Instrument": DISPLAY_NAMES[
                    symbol
                ],
                "Current": display_regime(
                    symbol,
                    current,
                ),
                "Next Day": display_regime(
                    symbol,
                    nxt,
                ),
                "Next Probability": next_probability,
                "Stickiness": stickiness,
                "Streak": streak,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STOCK INTELLIGENCE HELPERS
# ============================================================

def _normalise_stock_frame(df):
    """Return a clean timestamp/close dataframe from common parquet layouts."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "close"])

    df = df.copy()

    date_col = next(
        (c for c in ["timestamp", "date", "Date", "Datetime"] if c in df.columns),
        None,
    )
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        date_col = df.columns[0]

    close_col = next(
        (c for c in ["close", "Close", "adj_close", "Adj Close"] if c in df.columns),
        None,
    )
    if date_col is None or close_col is None:
        return pd.DataFrame(columns=["timestamp", "close"])

    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
    try:
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    except Exception:
        pass

    df["timestamp"] = df["timestamp"].dt.normalize()
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    return (
        df[["timestamp", "close"]]
        .dropna()
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_stock_prices(symbol):
    """Load an individual stock series without touching the HMM/index layer."""
    roots = [
        PROJECT_ROOT / "data" / "portfolio",
        PROJECT_ROOT / "data" / "stocks",
        PROJECT_ROOT / "data" / "regime" / "daily",
        PROJECT_ROOT / "data",
    ]

    # Prefer an individual symbol file so native stock prices are preserved.
    preferred_names = [
        symbol,
        symbol.lower(),
        symbol.upper(),
    ]

    for root in roots:
        if not root.exists():
            continue
        for name in preferred_names:
            for ext in [".parquet", ".pq"]:
                path = root / f"{name}{ext}"
                if path.exists():
                    try:
                        out = _normalise_stock_frame(pd.read_parquet(path))
                        if not out.empty:
                            return out
                    except Exception:
                        pass

    # Then inspect parquet files whose filename contains the ticker.
    for root in roots:
        if not root.exists():
            continue
        try:
            paths = root.rglob("*.parquet")
        except Exception:
            continue
        for path in paths:
            if symbol.upper() not in path.stem.upper():
                continue
            try:
                out = _normalise_stock_frame(pd.read_parquet(path))
                if not out.empty:
                    return out
            except Exception:
                continue

    # Finally support an existing unified price matrix.
    matrix_names = [
        "unified_price_matrix.parquet",
        "price_matrix.parquet",
        "unified_inr_price_matrix.parquet",
        "inr_price_matrix.parquet",
    ]
    for root in roots[:2]:
        for name in matrix_names:
            path = root / name
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                if symbol in df.columns:
                    out = pd.DataFrame({
                        "timestamp": df.index,
                        "close": df[symbol].values,
                    })
                    out = _normalise_stock_frame(out)
                    if not out.empty:
                        return out
            except Exception:
                continue

    return pd.DataFrame(columns=["timestamp", "close"])


def stock_snapshot(symbol):
    """Small first-stage stock view. No stock prediction is implied."""
    df = load_stock_prices(symbol)
    if df.empty:
        return None

    prices = df.set_index("timestamp")["close"].astype(float).sort_index()
    returns = prices.pct_change().dropna()

    latest = float(prices.iloc[-1])

    def trailing_return(days):
        cutoff = prices.index[-1] - pd.Timedelta(days=days)
        prior = prices[prices.index <= cutoff]
        if prior.empty:
            return np.nan
        return latest / float(prior.iloc[-1]) - 1.0

    vol20 = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else np.nan

    return {
        "prices": prices,
        "latest": latest,
        "latest_date": prices.index[-1],
        "return_1d": float(returns.iloc[-1]) if len(returns) else np.nan,
        "return_1m": trailing_return(30),
        "return_1y": trailing_return(365),
        "vol20": vol20,
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# ◈ QuantOS Market Regime Intelligence"
)

st.caption(
    "Live V10 HMM · causal regime inference · "
    "next-trading-day forecast · persistence intelligence"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Controls")

    selected_symbol = st.selectbox(
        "Index / Market Instrument",
        SYMBOLS,
        format_func=lambda x: DISPLAY_NAMES[x],
        key="market_instrument",
    )

    selected_stock = st.selectbox(
        "Stock",
        STOCK_SYMBOLS,
        format_func=lambda x: f"{x} · {STOCK_DISPLAY_NAMES[x]}",
        key="stock_instrument",
    )

    if st.button(
        "Refresh prediction artifact",
        use_container_width=True,
    ):
        load_predictions.clear()
        load_hmm_prediction_history.clear()
        st.rerun()

    st.caption(
        "Reads the latest artifact produced by "
        "`src/run_live_hmm.py`. It does not retrain the model."
    )


# ============================================================
# LOAD SELECTED
# ============================================================

try:
    all_data = load_predictions()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if selected_symbol not in all_data:
    st.error(
        f"{selected_symbol} is missing from "
        "latest_predictions.json. Run the live HMM runner first."
    )
    st.stop()

selected = all_data[selected_symbol]


# ============================================================
# SELECTED VALUES
# ============================================================

current_raw = str(
    first_value(
        selected,
        [
            "current_regime",
            "current_state",
            "regime",
        ],
        "N/A",
    )
).upper()

next_raw = str(
    first_value(
        selected,
        [
            "next_regime",
            "next_state",
            "forecast_regime",
        ],
        "N/A",
    )
).upper()

current = display_regime(
    selected_symbol,
    current_raw,
)

next_regime = display_regime(
    selected_symbol,
    next_raw,
)

probabilities = get_probabilities(
    selected
)

if selected_symbol == "INDIA_VIX":
    probabilities = remap_vix_probs(
        probabilities
    )

states = display_states(
    selected_symbol
)

next_probability = as_float(
    first_value(
        selected,
        [
            "next_day_probability",
            "next_probability",
        ],
    )
)

if (
    not np.isfinite(next_probability)
    and probabilities
):
    next_probability = max(
        probabilities.values()
    )

stickiness = get_stickiness(
    selected
)

stickiness_level = get_stickiness_level(
    selected,
    stickiness,
)

streak = get_streak(
    selected
)

latest_date = str(
    first_value(
        selected,
        [
            "latest_date",
            "date",
        ],
        "N/A",
    )
)

latest_close = as_float(
    first_value(
        selected,
        [
            "latest_close",
            "close",
        ],
    )
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Current Regime",
        current,
    )

with c2:
    st.metric(
        "Next Trading Day",
        next_regime,
    )

with c3:
    st.metric(
        "Next-State Probability",
        (
            f"{next_probability:.1%}"
            if np.isfinite(next_probability)
            else "N/A"
        ),
    )

with c4:
    st.metric(
        "Stickiness",
        (
            f"{stickiness:.1f}/100"
            if np.isfinite(stickiness)
            else "N/A"
        ),
        delta=stickiness_level,
    )

close_text = (
    f"{latest_close:,.2f}"
    if np.isfinite(latest_close)
    else "N/A"
)

st.caption(
    f"{DISPLAY_NAMES[selected_symbol]} · "
    f"Latest observation: {latest_date[:10]} · "
    f"Latest close: {close_text} · "
    f"Current streak: {streak} trading days"
)

if selected_symbol == "INDIA_VIX":
    st.info(
        "India VIX is displayed as LOW / MID / HIGH. "
        "Underlying V10 labels are mapped BEAR → LOW, "
        "SIDE → MID and BULL → HIGH."
    )


# ============================================================
# PROBABILITIES + STICKINESS
# ============================================================

left, right = st.columns(
    [1.45, 1]
)

with left:
    st.subheader(
        "Next-Trading-Day Regime Probabilities"
    )

    prob_df = pd.DataFrame(
        [
            {
                "Regime": regime,
                "Probability": probabilities.get(
                    regime,
                    0.0,
                ),
            }
            for regime in states
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=prob_df["Regime"],
            y=prob_df["Probability"] * 100,
            text=[
                f"{x:.1f}%"
                for x in (
                    prob_df["Probability"] * 100
                )
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        height=350,
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=25,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:
    st.subheader(
        "Persistence / Stickiness"
    )

    gauge_value = (
        0
        if not np.isfinite(stickiness)
        else max(
            0,
            min(
                100,
                stickiness,
            ),
        )
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "/100"},
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "threshold": {
                    "line": {
                        "width": 4
                    },
                    "thickness": 0.75,
                    "value": gauge_value,
                },
            },
            title={
                "text": stickiness_level
            },
        )
    )

    gauge.update_layout(
        height=260,
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=15,
        ),
    )

    st.plotly_chart(
        gauge,
        use_container_width=True,
    )

    s1, s2 = st.columns(2)

    with s1:
        st.metric(
            "Current Streak",
            f"{streak} days",
        )

    with s2:
        st.metric(
            "Forecast",
            next_regime,
        )

    st.caption(
        "Stickiness is a persistence score, not a "
        "probability of a positive return."
    )


# ============================================================
# MATRIX + LONG RUN
# ============================================================

matrix = get_transition_matrix(
    selected
)

stationary = get_stationary(
    selected
)

if selected_symbol == "INDIA_VIX":
    matrix = remap_vix_matrix(
        matrix
    )

    stationary = remap_vix_probs(
        stationary
    )

left, right = st.columns(
    [1.35, 1]
)

with left:
    st.subheader(
        "3 × 3 Transition Matrix"
    )

    st.caption(
        "Rows = current regime. "
        "Columns = next trading-day regime."
    )

    if matrix is not None:
        matrix = matrix.reindex(
            index=states,
            columns=states,
        )

        fig_matrix = px.imshow(
            matrix.values,
            x=matrix.columns,
            y=matrix.index,
            text_auto=".1%",
            aspect="auto",
        )

        fig_matrix.update_layout(
            height=420,
            xaxis_title="Next Regime",
            yaxis_title="Current Regime",
            margin=dict(
                l=20,
                r=20,
                t=25,
                b=20,
            ),
        )

        st.plotly_chart(
            fig_matrix,
            use_container_width=True,
        )

        display_matrix = matrix.map(
            lambda x: (
                f"{x:.1%}"
                if pd.notna(x)
                else "—"
            )
        )

        st.dataframe(
            display_matrix,
            use_container_width=True,
            height=170,
        )

    else:
        st.error(
            "Transition matrix is missing from the artifact. "
            "Run the final live HMM runner once."
        )

with right:
    st.subheader(
        "Long-Run Regime Mix"
    )

    st.caption(
        "Stationary distribution implied by the "
        "reference transition process."
    )

    if stationary:
        stationary_df = pd.DataFrame(
            [
                {
                    "Regime": regime,
                    "Share": stationary.get(
                        regime,
                        0.0,
                    ),
                }
                for regime in states
            ]
        )

        fig_stationary = go.Figure(
            go.Bar(
                x=stationary_df["Regime"],
                y=stationary_df["Share"] * 100,
                text=[
                    f"{x:.1f}%"
                    for x in (
                        stationary_df["Share"]
                        * 100
                    )
                ],
                textposition="auto",
            )
        )

        fig_stationary.update_layout(
            height=300,
            yaxis_title="Long-Run Share (%)",
            yaxis_range=[0, 100],
            margin=dict(
                l=20,
                r=20,
                t=25,
                b=20,
            ),
        )

        st.plotly_chart(
            fig_stationary,
            use_container_width=True,
        )

    else:
        st.error(
            "Long-run distribution is missing from the artifact."
        )


# ============================================================
# CURRENT VS NEXT
# ============================================================

st.subheader(
    "Current State vs Next-Day Distribution"
)

current_probabilities = first_value(
    selected,
    [
        "current_probabilities",
        "current_probs",
        "filtered_probabilities",
    ],
    {},
)

if not isinstance(
    current_probabilities,
    dict,
):
    current_probabilities = {}

current_probabilities = {
    str(k).upper(): as_float(v, 0.0)
    for k, v in current_probabilities.items()
}

if selected_symbol == "INDIA_VIX":
    current_probabilities = remap_vix_probs(
        current_probabilities
    )

comparison = pd.DataFrame(
    [
        {
            "Regime": regime,
            "Current": current_probabilities.get(
                regime,
                np.nan,
            ),
            "Next Day": probabilities.get(
                regime,
                np.nan,
            ),
        }
        for regime in states
    ]
)

fig_compare = go.Figure()

if comparison["Current"].notna().any():
    fig_compare.add_trace(
        go.Bar(
            x=comparison["Regime"],
            y=comparison["Current"] * 100,
            name="Current filtered probability",
        )
    )

fig_compare.add_trace(
    go.Bar(
        x=comparison["Regime"],
        y=comparison["Next Day"] * 100,
        name="Next-day forecast",
    )
)

fig_compare.update_layout(
    barmode="group",
    height=330,
    yaxis_title="Probability (%)",
    yaxis_range=[0, 100],
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20,
    ),
)

st.plotly_chart(
    fig_compare,
    use_container_width=True,
)


# ============================================================
# CROSS INDEX
# ============================================================

st.subheader(
    "Cross-Index Regime Matrix"
)

cross_df = build_cross_index(
    all_data
)

if not cross_df.empty:
    st.dataframe(
        cross_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Next Probability": st.column_config.ProgressColumn(
                "Next Probability",
                format="%.1%",
                min_value=0,
                max_value=1,
            ),
            "Stickiness": st.column_config.ProgressColumn(
                "Stickiness",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
        },
    )


# ============================================================
# STOCK INTELLIGENCE
# ============================================================

st.divider()
st.subheader(
    f"Stock Intelligence · {STOCK_DISPLAY_NAMES[selected_stock]} ({selected_stock})"
)
st.caption(
    "Stock selector is independent of the market-index HMM. "
    "This first stock layer shows the selected stock's own price history "
    "and basic quantitative state; it does not label the stock with the NIFTY 50 regime."
)

stock = stock_snapshot(selected_stock)

if stock is None:
    st.warning(
        f"No historical price file was found for {selected_stock}. "
        "The market-index dashboard remains fully available."
    )
else:
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric(
            "Latest Price",
            f"{stock['latest']:,.2f}",
        )

    with k2:
        value = stock["return_1d"]
        st.metric(
            "1D Return",
            f"{value:.2%}" if pd.notna(value) else "N/A",
        )

    with k3:
        value = stock["return_1m"]
        st.metric(
            "1M Return",
            f"{value:.2%}" if pd.notna(value) else "N/A",
        )

    with k4:
        value = stock["return_1y"]
        st.metric(
            "1Y Return",
            f"{value:.2%}" if pd.notna(value) else "N/A",
        )

    with k5:
        value = stock["vol20"]
        st.metric(
            "20D Annualized Vol",
            f"{value:.2%}" if pd.notna(value) else "N/A",
        )

    st.caption(
        f"Latest stock observation: {stock['latest_date'].strftime('%Y-%m-%d')}"
    )

    fig_stock = go.Figure()
    fig_stock.add_trace(
        go.Scatter(
            x=stock["prices"].index,
            y=stock["prices"].values,
            mode="lines",
            name=selected_stock,
        )
    )
    fig_stock.update_layout(
        height=380,
        title=f"{STOCK_DISPLAY_NAMES[selected_stock]} Price History",
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(
        fig_stock,
        use_container_width=True,
    )

# ============================================================
# HISTORICAL PREDICTION LOG
# ============================================================

prediction_history = load_hmm_prediction_history(
    selected_symbol
)

history_file = history_file_for_symbol(
    selected_symbol
)

if history_file is not None and not prediction_history.empty:

    history = prediction_history.copy()

    probability_columns = [
        "bull_probability",
        "side_probability",
        "bear_probability",
    ]

    for col in probability_columns:
        if col in history.columns:
            history[col] = pd.to_numeric(
                history[col],
                errors="coerce",
            )

    if "agreement" in history.columns:
        history["agreement"] = history[
            "agreement"
        ].astype("boolean")

    st.divider()

    st.subheader(
        f"Historical Prediction Log · "
        f"{DISPLAY_NAMES[selected_symbol]}"
    )

    st.caption(
        f"Instrument: {DISPLAY_NAMES[selected_symbol]} · "
        f"Source: {history_file.name} · "
        "Historical next-trading-day regime distributions "
        "produced by the causal HMM."
    )

    detailed = history[
        [
            "prediction_date",
            "target_trading_date",
            "current_regime",
            "next_regime",
            "bull_probability",
            "side_probability",
            "bear_probability",
            "actual_next_regime",
            "agreement",
        ]
    ].copy()

    detailed = detailed.sort_values(
        "prediction_date",
        ascending=False,
    )

    detailed[
        "prediction_date"
    ] = detailed[
        "prediction_date"
    ].dt.strftime("%Y-%m-%d")

    detailed[
        "target_trading_date"
    ] = detailed[
        "target_trading_date"
    ].dt.strftime("%Y-%m-%d")

    for col in [
        "bull_probability",
        "side_probability",
        "bear_probability",
    ]:
        detailed[col] = detailed[col].map(
            lambda x: (
                f"{x:.2%}"
                if pd.notna(x)
                else "N/A"
            )
        )

    detailed.columns = [
        "Prediction Date",
        "Target Date",
        "Current Regime",
        "Predicted Regime",
        "P(BULL)",
        "P(SIDE)",
        "P(BEAR)",
        "Actual Regime",
        "Agreement",
    ]

    st.dataframe(
        detailed,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

else:

    st.divider()

    st.subheader(
        f"Historical Prediction Log · "
        f"{DISPLAY_NAMES[selected_symbol]}"
    )

    st.info(
        f"No instrument-specific historical prediction file "
        f"is available for {DISPLAY_NAMES[selected_symbol]}. "
        "The dashboard will not substitute NIFTY 50 history here, "
        "because that would make the displayed history incorrect."
    )

    st.caption(
        "NIFTY 50 history is available in the current research "
        "artifact if you need a verified historical example."
    )


# METHODOLOGY
# ============================================================

with st.expander(
    "What QuantOS is calculating"
):
    st.markdown(
        """
        **HMM regime layer**

        Each instrument is modelled independently with three latent states:
        BULL, SIDE and BEAR.

        **Current regime**

        The current state is inferred causally using observations available
        through the latest trading session.

        **Next trading day**

        The displayed probability is the calibrated one-step-ahead probability
        of the latent regime. It is NOT a probability that the asset's return
        will be positive or negative.

        **Transition matrix**

        Rows represent today's regime and columns represent the next
        trading-day regime. Each row is a conditional probability distribution.

        **Stickiness**

        Stickiness combines persistence evidence, recent consistency and the
        current regime streak. It is an explanatory persistence score.

        **India VIX**

        India VIX is presented as LOW / MID / HIGH rather than BULL / SIDE /
        BEAR because it is a volatility index.
        """
    )

st.caption(
    "QuantOS · V10 regime layer · live artifact view"
)

