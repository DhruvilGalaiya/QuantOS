"""
QuantOS | Market Regimes page

Indian index regimes (V10 trend HMM), the V11 next-session forecast for
NIFTY 50 / BANK NIFTY, persistence, regime dynamics and the prediction log.
Launched through backend/app/regime_dashboard_v10.py (page router).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent          # backend/app/views
PROJECT_ROOT = APP_DIR.parents[2]                   # project root

import ui  # noqa: E402  (backend/app/ui.py)
from quantos_platform.data import daily as live_data  # noqa: E402

# How long a launch/rerun reuses the last refresh before checking Upstox again.
AUTO_REFRESH_TTL_SECONDS = 15 * 60

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


NEXT_DAY_CLOSE_SYMBOLS = {
    "NIFTY_50": "NIFTY 50",
    "NIFTY_BANK": "BANK NIFTY",
}


@st.cache_data(ttl=300, show_spinner=False)
def load_daily_close_changes(symbol: str) -> pd.Series:
    """
    Close-to-close % change per session, indexed by IST calendar date.

    Source: the Upstox daily candles in data/regime/daily, which the
    scheduled `india` job (quantos_platform.jobs) keeps current.
    """
    path = (
        PROJECT_ROOT / "data" / "regime" / "daily"
        / f"{symbol.lower()}.parquet"
    )
    if not path.exists():
        return pd.Series(dtype=float)

    df = pd.read_parquet(path, columns=["timestamp", "close"])
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    dates = ts.dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)

    closes = (
        pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=dates)
        .dropna()
        .sort_index()
    )
    closes = closes[~closes.index.duplicated(keep="last")]
    return closes.pct_change() * 100


V11_DIR = PROJECT_ROOT / "data" / "regime" / "live_predictions" / "v11"
V11_SYMBOLS = ("NIFTY_50", "NIFTY_BANK")


@st.cache_data(ttl=300, show_spinner=False)
def load_v11_history(symbol: str) -> pd.DataFrame:
    """V11 walk-forward prediction log (same columns as the V10 log)."""
    path = V11_DIR / f"prediction_history_v11_{symbol.lower()}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["prediction_date", "target_trading_date"])
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=2)
    return df[df["prediction_date"] >= cutoff].sort_values("prediction_date").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_v11_metrics(symbol: str) -> dict:
    path = V11_DIR / f"metrics_v11_{symbol.lower()}.json"
    return json.loads(path.read_text()) if path.exists() else {}


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


def build_market_overview(data):
    """One row per instrument: regime, persistence, latest move, next session."""
    rows = []
    for symbol in SYMBOLS:
        payload = data.get(symbol)
        if not isinstance(payload, dict):
            continue
        current_regime = str(first_value(payload, ["current_regime", "current_state", "regime"], "N/A")).upper()
        moves = load_daily_close_changes(symbol)
        if symbol in V11_SYMBOLS and load_v11_metrics(symbol).get("latest"):
            nxt = load_v11_metrics(symbol)["latest"]["next_regime"]
            model = "V11"
        else:
            nxt = display_regime(symbol, str(first_value(payload, ["next_regime", "next_state"], "N/A")).upper())
            model = "V10"
        rows.append({
            "Instrument": DISPLAY_NAMES[symbol],
            "Last close": as_float(first_value(payload, ["latest_close", "close"])),
            "1D change": float(moves.iloc[-1]) if len(moves) else np.nan,
            "Regime": display_regime(symbol, current_regime),
            "Streak": get_streak(payload),
            "Persistence": get_stickiness(payload),
            "Next session": f"{nxt} ({model})",
        })
    return pd.DataFrame(rows)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# ◈ QuantOS Market Regime Intelligence"
)

st.caption(
    "Indian index regimes · V11 next-session forecast · "
    "persistence · walk-forward prediction log"
)


# ============================================================
# DATA REFRESH (scheduled jobs; local mode also refreshes stale data on open)
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def load_live_quotes(symbols: tuple[str, ...]):
    if ui.PUBLIC:   # public mode: no live API calls from page views
        return {}
    try:
        return live_data.fetch_live_quotes(list(symbols))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _clear_artifact_caches():
    load_predictions.clear()
    load_hmm_prediction_history.clear()
    load_daily_close_changes.clear()
    load_v11_history.clear()
    load_v11_metrics.clear()


with st.spinner("Checking for new market data…"):
    refresh = ui.refresh_on_launch("india")
if refresh and refresh.get("updated") or (refresh and refresh.get("v11")):
    _clear_artifact_caches()

ui.job_health("india", "india_daily")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Market")

    selected_symbol = st.selectbox(
        "Index / Market Instrument",
        SYMBOLS,
        format_func=lambda x: DISPLAY_NAMES[x],
        key="market_instrument",
    )

    if ui.refresh_button("india", "Pull latest from Upstox"):
        load_live_quotes.clear()
        _clear_artifact_caches()
        st.rerun()

    st.caption(ui.freshness_caption("india_daily", "Daily data"))
    st.caption(ui.freshness_caption("v11", "V11 forecast for"))
    st.caption(
        "Data is updated by scheduled jobs (`python -m quantos_platform.jobs india`)"
        + ("." if ui.PUBLIC else "; in local mode stale data is also refreshed when this page opens.")
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
# HEADLINE: price · regime · next-session forecast
# ============================================================
# Intraday quote is display only; models use completed daily sessions.

quotes = load_live_quotes((selected_symbol,))
q = quotes.get(selected_symbol) if "_error" not in quotes else None
v11_latest = (
    load_v11_metrics(selected_symbol).get("latest")
    if selected_symbol in V11_SYMBOLS
    else None
)

close_text = f"{latest_close:,.2f}" if np.isfinite(latest_close) else "N/A"

st.markdown(f"### {DISPLAY_NAMES[selected_symbol]}")

h1, h2, h3, h4 = st.columns(4)

with h1:
    if q is not None:
        st.metric(
            "Live price",
            f"{q['ltp']:,.2f}",
            delta=(
                f"{q['ltp'] / q['prev_close'] - 1:+.2%}"
                if q.get("prev_close")
                else None
            ),
            delta_color="inverse" if selected_symbol == "INDIA_VIX" else "normal",
        )
    else:
        st.metric("Last close", close_text)

with h2:
    st.metric(
        "Current regime",
        current,
        delta=f"{streak} session{"s" if streak != 1 else ""}",
        delta_color="off",
        help="V10 trend regime (Gaussian HMM, causal filtering). "
             "The delta shows how many sessions it has lasted.",
    )

with h3:
    if v11_latest:
        st.metric(
            f"Next session · {v11_latest['target_trading_date']}",
            v11_latest["next_regime"],
            delta=v11_latest["status"],
            delta_color="off",
            help="V11 pre-open forecast of the next close-to-close move: "
                 "BULL > +0.25%, BEAR < −0.25%, otherwise SIDE. FINAL once the "
                 "US session has closed (~02:30 IST).",
        )
    else:
        st.metric(
            "Next session",
            next_regime,
            help="V10 trend model's one-step-ahead regime.",
        )

with h4:
    confidence = v11_latest["next_day_probability"] if v11_latest else next_probability
    st.metric(
        "Forecast confidence",
        f"{confidence:.1%}" if np.isfinite(confidence) else "N/A",
        help="Probability the model assigns to the forecast regime.",
    )

if v11_latest:
    st.caption(
        f"Last close {close_text} on {latest_date[:10]} · "
        f"P(BULL) {v11_latest['bull_probability']:.0%} · "
        f"P(SIDE) {v11_latest['side_probability']:.0%} · "
        f"P(BEAR) {v11_latest['bear_probability']:.0%} · "
        "V11 model (US overnight session + India end-of-day + HMM regime)"
    )
else:
    st.caption(
        f"Last close {close_text} on {latest_date[:10]} · V10 trend model"
    )

if selected_symbol == "INDIA_VIX":
    st.info(
        "India VIX regimes are shown as LOW / MID / HIGH volatility "
        "(underlying V10 labels BEAR → LOW, SIDE → MID, BULL → HIGH)."
    )


# ============================================================
# PERSISTENCE + REGIME DYNAMICS
# ============================================================

matrix = get_transition_matrix(selected)
stationary = get_stationary(selected)

if selected_symbol == "INDIA_VIX":
    matrix = remap_vix_matrix(matrix)
    stationary = remap_vix_probs(stationary)

st.divider()

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Persistence")

    gauge_value = 0 if not np.isfinite(stickiness) else max(0, min(100, stickiness))
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 50], "color": "rgba(148,163,184,0.15)"},
                    {"range": [50, 75], "color": "rgba(148,163,184,0.28)"},
                    {"range": [75, 100], "color": "rgba(148,163,184,0.42)"},
                ],
            },
            title={"text": stickiness_level},
        )
    )
    gauge.update_layout(height=230, margin=dict(l=25, r=25, t=50, b=5))
    st.plotly_chart(gauge, width="stretch")

    stay = as_float(first_value(selected, ["stay_probability"]))
    consistency = as_float(first_value(selected, ["recent_consistency"]))
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Stay probability", f"{stay:.0%}" if np.isfinite(stay) else "N/A")
    with m2:
        st.metric("10-day consistency", f"{consistency:.0%}" if np.isfinite(consistency) else "N/A")
    with m3:
        st.metric("Streak", f"{streak} d")
    st.caption(
        "Score = 60% stay probability + 25% 10-day consistency + "
        "15% streak (capped at 10 sessions). Measures how entrenched the "
        "current regime is, not where price goes next."
    )

with right:
    st.subheader("Regime dynamics")
    st.caption(
        "How each regime behaves, from the trailing ~3-year transition "
        "process. The current regime is marked ●."
    )

    if matrix is not None:
        matrix = matrix.reindex(index=states, columns=states)
        rows = []
        for regime in states:
            stay_p = float(matrix.loc[regime, regime])
            exits = matrix.loc[regime].drop(regime)
            exit_to = exits.idxmax() if exits.sum() > 0 else "—"
            rows.append({
                "Regime": f"● {regime}" if regime == current else regime,
                "Continues next session": stay_p,
                "Typical duration": 1 / max(1 - stay_p, 1e-6),
                "When it ends, usually →": (
                    f"{exit_to} ({exits.max() / exits.sum():.0%})"
                    if exits.sum() > 0 else "—"
                ),
                "Long-run share": stationary.get(regime, np.nan) if stationary else np.nan,
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Continues next session": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1),
                "Typical duration": st.column_config.NumberColumn(format="%.1f sessions"),
                "Long-run share": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1),
            },
        )
        st.caption(
            "Typical duration = 1 / (1 − continuation probability). "
            "Long-run share = stationary distribution of the full-history process."
        )
    else:
        st.info("Transition data is not available for this instrument yet.")


# ============================================================
# MARKET OVERVIEW (all instruments)
# ============================================================

st.divider()
st.subheader("Market overview")

overview = build_market_overview(all_data)
if not overview.empty:
    st.dataframe(
        overview,
        hide_index=True,
        width="stretch",
        column_config={
            "Last close": st.column_config.NumberColumn(format="%.2f"),
            "1D change": st.column_config.NumberColumn(format="%+.2f%%"),
            "Streak": st.column_config.NumberColumn(format="%d sessions"),
            "Persistence": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100),
        },
    )
    st.caption(
        "Regime = V10 trend regime. Next session = V11 forecast for NIFTY 50 "
        "and NIFTY BANK; other instruments show the V10 trend model."
    )


# ============================================================
# HISTORICAL PREDICTION LOG
# ============================================================

log_model = "V10"
if selected_symbol in V11_SYMBOLS and not load_v11_history(selected_symbol).empty:
    st.divider()
    log_model = st.radio(
        "Prediction log model",
        ["V11", "V10"],
        horizontal=True,
        key="log_model",
        help="V11 = new pre-open next-session forecaster. V10 = original trend-regime HMM.",
    )

if log_model == "V11":
    prediction_history = load_v11_history(selected_symbol)
    history_file = V11_DIR / f"prediction_history_v11_{selected_symbol.lower()}.csv"
else:
    prediction_history = load_hmm_prediction_history(selected_symbol)
    history_file = history_file_for_symbol(selected_symbol)

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

    if log_model == "V10":
        st.divider()

    st.subheader(
        f"Historical Prediction Log · "
        f"{DISPLAY_NAMES[selected_symbol]} · {log_model}"
    )

    # Scorecard: predicted regime vs the actual next-session close.
    own_moves = load_daily_close_changes(selected_symbol)
    scored = prediction_history.assign(
        move=prediction_history["target_trading_date"].dt.normalize().map(own_moves)
    ).dropna(subset=["move"])
    if not scored.empty and selected_symbol != "INDIA_VIX":
        calls = scored[scored["next_regime"].isin(["BULL", "BEAR"])]
        hits = ((calls["next_regime"] == "BULL") & (calls["move"] > 0)) | (
            (calls["next_regime"] == "BEAR") & (calls["move"] < 0)
        )
        actual3 = np.where(
            scored["move"] > 0.25, "BULL",
            np.where(scored["move"] < -0.25, "BEAR", "SIDE"),
        )
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric(
                "Direction accuracy",
                f"{hits.mean():.1%}" if len(calls) else "N/A",
                help="BULL call → next close up; BEAR call → next close down. "
                     "SIDE calls are excluded.",
            )
        with a2:
            st.metric(
                "3-class accuracy",
                f"{(scored['next_regime'].values == actual3).mean():.1%}",
                help="Actual: BULL > +0.25%, BEAR < −0.25%, otherwise SIDE.",
            )
        with a3:
            st.metric("Always-UP baseline", f"{(scored['move'] > 0).mean():.1%}")
        with a4:
            st.metric("Scored sessions", f"{len(scored):,}", delta=f"{len(calls):,} BULL/BEAR calls",
                      delta_color="off")
        st.caption(
            f"Scored against {DISPLAY_NAMES[selected_symbol]}'s own next-session "
            "close for the rows shown below. V11 rows are walk-forward: each was "
            "predicted by a model trained only on earlier data."
        )

    st.caption(
        f"Instrument: {DISPLAY_NAMES[selected_symbol]} · "
        f"Source: {history_file.name} · "
        "Historical next-trading-day regime distributions. "
        "Next Day Closing = close-to-close % move of NIFTY 50 and "
        "BANK NIFTY on the Target Date (Upstox daily closes, refreshed "
        "on every launch). Pending = target session not closed yet."
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
        ]
    ].copy()

    target_dates = detailed["target_trading_date"].dt.normalize()
    close_columns = []
    for sym, label in NEXT_DAY_CLOSE_SYMBOLS.items():
        column = f"Next Day Closing · {label}"
        detailed[column] = target_dates.map(load_daily_close_changes(sym))
        close_columns.append(column)

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

    detailed = detailed.rename(
        columns={
            "prediction_date": "Prediction Date",
            "target_trading_date": "Target Date",
            "current_regime": "Current Regime",
            "next_regime": "Predicted Regime",
            "bull_probability": "P(BULL)",
            "side_probability": "P(SIDE)",
            "bear_probability": "P(BEAR)",
        }
    )

    def _move_colour(value):
        if pd.isna(value) or value == 0:
            return ""
        return "color: #16a34a" if value > 0 else "color: #dc2626"

    styled = (
        detailed.style
        .format("{:+.2f}%", subset=close_columns, na_rep="Pending")
        .map(_move_colour, subset=close_columns)
    )

    st.dataframe(
        styled,
        width="stretch",
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

with st.expander("Methodology"):
    st.markdown(
        """
        **Trend regime (V10).** Each index is modelled with a three-state
        Gaussian HMM (BULL / SIDE / BEAR) using causal filtering, so today's
        regime only uses information available through today's close.

        **Next-session forecast (V11, NIFTY 50 and NIFTY BANK).** Forecasts the
        next close-to-close move (BULL > +0.25%, BEAR < −0.25%, otherwise SIDE)
        from the US session that trades after India's close, India's
        end-of-day state and a causal HMM regime. Trained walk-forward; every
        logged prediction was made by a model fit only on earlier data.

        **Persistence.** 60% stay probability + 25% ten-session consistency +
        15% streak. It describes how entrenched a regime is, not the direction
        of the next move.

        **Regime dynamics.** Continuation probabilities come from the trailing
        transition process; typical duration is 1 / (1 − continuation).

        **India VIX** regimes are shown as LOW / MID / HIGH volatility.
        """
    )

st.caption("QuantOS · Market Regimes")
