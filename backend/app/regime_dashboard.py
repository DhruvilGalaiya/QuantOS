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
    """Calculate historical price statistics for a stock with local price data."""
    df = load_stock_prices(symbol)
    if df.empty:
        return None

    prices = (
        df.set_index("timestamp")["close"]
        .astype(float)
        .sort_index()
    )
    prices = prices[~prices.index.duplicated(keep="last")].dropna()

    if prices.empty:
        return None

    returns = prices.pct_change().dropna()
    latest = float(prices.iloc[-1])
    latest_date = prices.index[-1]

    def trailing_return(days):
        cutoff = latest_date - pd.Timedelta(days=days)
        prior = prices[prices.index <= cutoff]
        if prior.empty or float(prior.iloc[-1]) == 0:
            return np.nan
        return latest / float(prior.iloc[-1]) - 1.0

    vol20 = (
        float(returns.tail(20).std() * np.sqrt(252))
        if len(returns) >= 20
        else np.nan
    )

    trailing_year = prices[
        prices.index >= latest_date - pd.Timedelta(days=365)
    ]
    high_52w = float(trailing_year.max()) if not trailing_year.empty else np.nan
    low_52w = float(trailing_year.min()) if not trailing_year.empty else np.nan

    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1.0
    max_drawdown = (
        float(drawdown.tail(252).min())
        if not drawdown.empty
        else np.nan
    )

    return {
        "prices": prices,
        "latest": latest,
        "latest_date": latest_date,
        "return_1d": float(returns.iloc[-1]) if len(returns) else np.nan,
        "return_1m": trailing_return(30),
        "return_3m": trailing_return(90),
        "return_6m": trailing_return(180),
        "return_1y": trailing_return(365),
        "vol20": vol20,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# QUANTOS DASHBOARD UI
# ============================================================

st.set_page_config(
    page_title="QuantOS | Regime Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] {
        background: #101c32;
        border-right: 1px solid #263651;
    }
    [data-testid="stSidebar"] {
        color: #f1f5fb;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #e6edf7 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] input {
        color: #14233b !important;
        -webkit-text-fill-color: #14233b !important;
    }
    [data-testid="stSidebar"] input {
        background: #ffffff !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: #1769e0 !important;
        border: 1px solid #1769e0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .q-stock-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 14px;
    }
    .q-stock-symbol {
        color: #14233b;
        font-size: 1.9rem;
        font-weight: 750;
        margin-bottom: 0;
    }
    .q-stock-company {
        color: #64748b;
        font-size: 1rem;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .q-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.7px;
        color: #14233b;
        margin-bottom: 0.1rem;
    }
    .q-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    div[data-testid="stMetricValue"] { color: #17243b; }
    div[data-testid="stPlotlyChart"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 8px;
    }
    h2, h3 { color: #17243b; }
    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

PAGES = [
    "Overview",
    "Market Regime",
    "Cross-Market Analysis",
    "Stock Intelligence",
    "Prediction History",
    "Methodology",
]

with st.sidebar:
    st.markdown("## ◈ QuantOS")
    st.caption("MARKET INTELLIGENCE")
    page = st.radio("NAVIGATION", PAGES, label_visibility="visible")
    st.divider()

    selected_symbol = st.selectbox(
        "Market instrument",
        SYMBOLS,
        format_func=lambda x: DISPLAY_NAMES[x],
        key="market_instrument",
    )
    selected_stock_option = st.selectbox(
        "Stock universe",
        STOCK_SYMBOLS,
        format_func=lambda x: f"{x} · {STOCK_DISPLAY_NAMES[x]}",
        key="stock_instrument",
    )
    custom_stock = st.text_input(
        "Or enter another ticker",
        placeholder="e.g. JPM, ORCL, INTC",
        help="Uses local historical data if a matching stock dataset exists.",
    ).strip().upper()
    selected_stock = custom_stock if custom_stock else selected_stock_option

    if st.button("Refresh prediction data", use_container_width=True):
        load_predictions.clear()
        load_hmm_prediction_history.clear()
        st.rerun()

    st.caption("Uses existing prediction artifacts. This dashboard does not retrain the HMM.")

try:
    all_data = load_predictions()
except Exception as exc:
    st.error(f"Unable to load prediction data: {exc}")
    st.stop()

if selected_symbol not in all_data:
    st.error(
        f"{selected_symbol} is missing from latest_predictions.json. "
        "Run the live HMM pipeline to generate the artifact."
    )
    st.stop()

selected = all_data[selected_symbol]

current_raw = str(first_value(
    selected, ["current_regime", "current_state", "regime"], "N/A"
)).upper()
next_raw = str(first_value(
    selected, ["next_regime", "next_state", "forecast_regime"], "N/A"
)).upper()

current = display_regime(selected_symbol, current_raw)
next_regime = display_regime(selected_symbol, next_raw)
probabilities = get_probabilities(selected)

if selected_symbol == "INDIA_VIX":
    probabilities = remap_vix_probs(probabilities)

states = display_states(selected_symbol)
matrix = get_transition_matrix(selected)
stationary = get_stationary(selected)
stickiness = get_stickiness(selected)
streak = get_streak(selected)
latest_date = first_value(selected, ["latest_date", "date"], "Unavailable")
latest_close = as_float(first_value(selected, ["latest_close", "close"]))

def heading(title, subtitle):
    st.markdown(f'<div class="q-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="q-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def probability_frame(values):
    if not values:
        return pd.DataFrame(columns=["Regime", "Probability"])
    return pd.DataFrame([
        {"Regime": str(k), "Probability": float(v) * 100}
        for k, v in values.items()
    ]).sort_values("Probability", ascending=False)

def probability_chart(values, title):
    frame = probability_frame(values)
    if frame.empty:
        st.info("Probability data is not available in the current artifact.")
        return
    fig = px.bar(
        frame, x="Regime", y="Probability", text_auto=".1f",
        title=title, color="Regime",
        color_discrete_sequence=["#233f68", "#6286b5", "#a7b9d1", "#d1dbea"],
    )
    fig.update_layout(
        template="plotly_white", showlegend=False,
        yaxis_title="Probability (%)", xaxis_title="",
        margin=dict(l=20, r=20, t=65, b=20),
        height=360,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

def show_transition_matrix():
    if isinstance(matrix, pd.DataFrame) and not matrix.empty:
        display_matrix = matrix.copy()
        numeric = display_matrix.apply(pd.to_numeric, errors="coerce")
        if numeric.notna().any().any() and numeric.max().max() <= 1.0:
            display_matrix = numeric * 100
            st.dataframe(
                display_matrix.style.format("{:.1f}%"),
                use_container_width=True,
            )
        else:
            st.dataframe(display_matrix, use_container_width=True)
    else:
        st.info("Transition matrix is not available in this prediction artifact.")

def show_market_metrics():
    a, b, c, d = st.columns(4)
    a.metric("Current regime", str(current))
    b.metric("Next forecast", str(next_regime))
    c.metric("Regime duration", f"{streak} days")
    d.metric("Stickiness score", f"{stickiness:.2f}" if np.isfinite(stickiness) else "N/A")

if page == "Overview":
    heading("Market Regime Intelligence", "A compact view of the current HMM state and its next-session forecast.")
    st.caption(f"Instrument: {DISPLAY_NAMES[selected_symbol]}  ·  Artifact date: {latest_date}")

    show_market_metrics()
    st.divider()

    left, right = st.columns([1.35, 1])
    with left:
        probability_chart(probabilities, "Next-session regime probabilities")
    with right:
        st.subheader("Market snapshot")
        st.metric("Latest recorded close", f"{latest_close:,.2f}" if np.isfinite(latest_close) else "N/A")
        st.metric("Current regime", str(current))
        st.metric("Forecast regime", str(next_regime))
        st.caption("Values reflect the loaded artifact, not the computer's current date.")

    st.subheader("Cross-market snapshot")
    cross = build_cross_index(all_data)
    if isinstance(cross, pd.DataFrame) and not cross.empty:
        st.dataframe(cross, use_container_width=True, hide_index=True)
    else:
        st.info("Cross-market comparison is unavailable.")

elif page == "Market Regime":
    heading("Market Regime", "Current state, forecast probabilities, and regime persistence.")
    st.caption(f"{DISPLAY_NAMES[selected_symbol]}  ·  Data date: {latest_date}")
    show_market_metrics()

    left, right = st.columns(2)
    with left:
        probability_chart(probabilities, "Forecast probability by regime")
    with right:
        st.subheader("Regime persistence")
        if np.isfinite(stickiness):
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(stickiness),
                title={"text": "Stickiness score"},
                gauge={"axis": {"range": [0, 1]},
                       "bar": {"color": "#233f68"},
                       "steps": [
                           {"range": [0, 0.33], "color": "#edf2f7"},
                           {"range": [0.33, 0.66], "color": "#dbe5f1"},
                           {"range": [0.66, 1], "color": "#b8cbe2"},
                       ]},
            ))
            gauge.update_layout(template="plotly_white", height=330, margin=dict(l=25, r=25, t=55, b=20))
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("Stickiness score is unavailable.")

    st.subheader("Transition matrix")
    show_transition_matrix()

    if isinstance(stationary, dict) and stationary:
        st.subheader("Stationary distribution")
        probability_chart(stationary, "Long-run state distribution")

elif page == "Cross-Market Analysis":
    heading("Cross-Market Analysis", "Compare the regime outputs available in the current prediction artifact.")
    cross = build_cross_index(all_data)
    if isinstance(cross, pd.DataFrame) and not cross.empty:
        st.dataframe(cross, use_container_width=True, hide_index=True)
        st.caption("This view reports model outputs. It does not imply that every instrument has the same model calibration.")
    else:
        st.info("Cross-market comparison is unavailable in the current artifact.")

elif page == "Stock Intelligence":
    heading(
        "Stock Intelligence",
        "Historical price analysis and key statistics for the selected stock."
    )

    snapshot = stock_snapshot(selected_stock)

    if snapshot is None:
        st.warning(f"No local historical price data was found for {selected_stock}.")
        st.caption(
            "Select a ticker with available local price history. Custom ticker entry "
            "does not automatically download market data."
        )
    else:
        prices = snapshot["prices"].sort_index()
        last_date = pd.Timestamp(snapshot["latest_date"]).strftime("%d %b %Y")

        company_names = {
            "NVDA": "NVIDIA", "AAPL": "Apple Inc.", "MSFT": "Microsoft",
            "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms",
            "AVGO": "Broadcom", "TSLA": "Tesla", "NFLX": "Netflix",
        }
        company = company_names.get(selected_stock, selected_stock)

        st.markdown(
            f"""
            <div class="q-stock-panel">
                <div class="q-stock-symbol">{selected_stock}</div>
                <div class="q-stock-company">{company} · Historical price analysis</div>
                <div style="margin-top:10px;color:#64748b">
                    Last price date: <strong style="color:#14233b">{last_date}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def fmt_pct(value):
            return f"{value:+.2%}" if np.isfinite(value) else "N/A"

        # Only three headline metrics
        a, b, c = st.columns(3)
        a.metric("Latest close", f'{snapshot["latest"]:,.2f}')
        b.metric(
            "1-day return",
            fmt_pct(snapshot["return_1d"]),
            delta=fmt_pct(snapshot["return_1d"])
            if np.isfinite(snapshot["return_1d"]) else None,
        )
        c.metric(
            "20-day volatility (ann.)",
            f'{snapshot["vol20"]:.2%}'
            if np.isfinite(snapshot["vol20"]) else "N/A",
        )

        st.divider()

        # Full-width historical chart
        st.subheader("Historical Price Chart")
        chart_range = st.radio(
            "Chart range",
            ["1Y", "2Y", "3Y", "5Y", "ALL"],
            index=4,
            horizontal=True,
            key=f"stock_range_{selected_stock}",
        )

        if chart_range == "ALL":
            chart_prices = prices
        else:
            years = int(chart_range[:-1])
            cutoff = prices.index[-1] - pd.Timedelta(days=365 * years)
            chart_prices = prices[prices.index >= cutoff]

        price_frame = chart_prices.rename("Close").reset_index()
        price_frame.columns = ["Date", "Close"]

        fig = px.line(
            price_frame,
            x="Date",
            y="Close",
            title=f"{selected_stock} historical close",
        )
        fig.update_traces(line=dict(color="#1769e0", width=2))
        fig.update_layout(
            template="plotly_white",
            height=470,
            margin=dict(l=15, r=15, t=55, b=20),
            xaxis_title="Date",
            yaxis_title="Close price",
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed statistics live below the full-width chart
        st.subheader("Key Statistics")

        stat_rows = [
            ("Latest close", f'{snapshot["latest"]:,.2f}', "neutral"),
            ("1-day return", fmt_pct(snapshot["return_1d"]), "return_1d"),
            ("1-month return", fmt_pct(snapshot["return_1m"]), "return_1m"),
            ("3-month return", fmt_pct(snapshot["return_3m"]), "return_3m"),
            ("6-month return", fmt_pct(snapshot["return_6m"]), "return_6m"),
            ("1-year return", fmt_pct(snapshot["return_1y"]), "return_1y"),
            (
                "20-day volatility (annualized)",
                f'{snapshot["vol20"]:.2%}' if np.isfinite(snapshot["vol20"]) else "N/A",
                "neutral",
            ),
            (
                "52-week high",
                f'{snapshot["high_52w"]:,.2f}' if np.isfinite(snapshot["high_52w"]) else "N/A",
                "neutral",
            ),
            (
                "52-week low",
                f'{snapshot["low_52w"]:,.2f}' if np.isfinite(snapshot["low_52w"]) else "N/A",
                "neutral",
            ),
            ("252-day maximum drawdown", fmt_pct(snapshot["max_drawdown"]), "max_drawdown"),
            ("Last price date", last_date, "neutral"),
        ]

        stats_df = pd.DataFrame(
            [{"Metric": label, "Value": value, "_kind": kind}
             for label, value, kind in stat_rows]
        )

        def colour_stat_row(row):
            kind = row["_kind"]
            if kind == "neutral":
                colour = "#14233b"
            else:
                raw = snapshot.get(kind, np.nan)
                colour = (
                    "#07804b" if np.isfinite(raw) and raw > 0
                    else "#dc2626" if np.isfinite(raw) and raw < 0
                    else "#14233b"
                )
            return [
                "color: #14233b; font-weight: 500",
                f"color: {colour}; font-weight: 600",
                "display: none",
            ]

        styled_stats = (
            stats_df.style
            .apply(colour_stat_row, axis=1)
            .hide(axis="columns", subset=["_kind"])
        )
        st.dataframe(
            styled_stats,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Green indicates positive returns; red indicates negative returns or drawdowns. "
            "Statistics are calculated from available local price history and are not forecasts."
        )

elif page == "Prediction History":
    heading("Prediction History", "Review saved HMM prediction history for the selected instrument.")
    try:
        history = load_hmm_prediction_history(selected_symbol)
        if isinstance(history, pd.DataFrame) and not history.empty:
            st.dataframe(history, use_container_width=True, hide_index=True)
        elif isinstance(history, list) and history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("No prediction history was returned for this instrument.")
    except Exception as exc:
        st.warning(f"Prediction history could not be loaded: {exc}")

elif page == "Methodology":
    heading("Methodology & Data Notes", "What this dashboard displays and what it does not claim.")
    st.subheader("Regime inference")
    st.write(
        "The dashboard reads the saved live HMM prediction artifact. "
        "It does not train the model or independently generate forecasts."
    )
    st.subheader("Probability interpretation")
    st.write(
        "Forecast probabilities describe the model's next-session state distribution. "
        "They are estimates, not guarantees of market direction or investment returns."
    )
    st.subheader("Data freshness")
    st.write(
        f"The selected instrument's artifact reports the date: {latest_date}. "
        "This is the artifact date, not today's system date. A newer date requires "
        "successful ingestion and execution of the upstream pipeline."
    )
    st.subheader("Scope")
    st.write(
        "This dashboard covers HMM regime intelligence, cross-market comparisons, "
        "and historical stock statistics where local price data exists. "
        "It does not fabricate options surfaces, portfolio backtests, or Monte Carlo outputs."
    )

st.divider()
st.caption("QuantOS · Research dashboard · Model outputs are informational, not investment advice.")
