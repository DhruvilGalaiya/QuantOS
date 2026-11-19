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

import theme  # noqa: E402  (backend/app/theme.py)
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
def load_daily_closes(symbol: str) -> pd.Series:
    """
    Daily closes indexed by IST calendar date.

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
    return closes[~closes.index.duplicated(keep="last")]


@st.cache_data(ttl=300, show_spinner=False)
def load_daily_close_changes(symbol: str) -> pd.Series:
    """Close-to-close % change per session, indexed by IST calendar date."""
    return load_daily_closes(symbol).pct_change() * 100


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
            "Symbol": symbol,
            "Instrument": DISPLAY_NAMES[symbol],
            "Last close": as_float(first_value(payload, ["latest_close", "close"])),
            "1D change": float(moves.iloc[-1]) if len(moves) else np.nan,
            "Regime": display_regime(symbol, current_regime),
            "Streak": get_streak(payload),
            "Persistence": get_stickiness(payload),
            "Next session": nxt,
            "Model": model,
        })
    return pd.DataFrame(rows)


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
    load_daily_closes.clear()
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
    st.subheader("Market")

    selected_symbol = st.selectbox(
        "Instrument",
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
        "Scheduled jobs keep this data current (`python -m quantos_platform.jobs india`)"
        + ("." if ui.PUBLIC else ". In local mode, stale data is also refreshed when the page opens.")
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
        f"{selected_symbol} is missing from latest_predictions.json. "
        "Run the india job to rebuild the predictions."
    )
    st.stop()

selected = all_data[selected_symbol]
IS_VIX = selected_symbol == "INDIA_VIX"
NAME = DISPLAY_NAMES[selected_symbol]

current = display_regime(
    selected_symbol,
    str(first_value(selected, ["current_regime", "current_state", "regime"], "N/A")).upper(),
)
next_regime = display_regime(
    selected_symbol,
    str(first_value(selected, ["next_regime", "next_state", "forecast_regime"], "N/A")).upper(),
)

probabilities = get_probabilities(selected)
if IS_VIX:
    probabilities = remap_vix_probs(probabilities)

states = display_states(selected_symbol)

next_probability = as_float(first_value(selected, ["next_day_probability", "next_probability"]))
if not np.isfinite(next_probability) and probabilities:
    next_probability = max(probabilities.values())

stickiness = get_stickiness(selected)
stickiness_level = get_stickiness_level(selected, stickiness)
streak = get_streak(selected)
latest_date = str(first_value(selected, ["latest_date", "date"], "N/A"))
latest_close = as_float(first_value(selected, ["latest_close", "close"]))
close_text = f"{latest_close:,.2f}" if np.isfinite(latest_close) else "N/A"


def nice_day(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%a %d %b")
    except (TypeError, ValueError):
        return str(value)


# ============================================================
# HERO: regime, price, next-session forecast
# ============================================================
# Intraday quote is display only; models use completed daily sessions.

quotes = load_live_quotes((selected_symbol,))
q = quotes.get(selected_symbol) if "_error" not in quotes else None
v11_latest = load_v11_metrics(selected_symbol).get("latest") if selected_symbol in V11_SYMBOLS else None
moves = load_daily_close_changes(selected_symbol)

if q is not None:
    price_label, price_text, live = "Live", f"{q['ltp']:,.2f}", True
    day_move = (q["ltp"] / q["prev_close"] - 1) * 100 if q.get("prev_close") else np.nan
else:
    price_label = f"Close, {nice_day(latest_date[:10])}" if latest_date != "N/A" else "Last close"
    price_text, live = close_text, False
    day_move = float(moves.iloc[-1]) if len(moves) else np.nan
change_html = (
    theme.signed_html(day_move, invert=IS_VIX) + ' <span class="t-muted">on the day</span>'
    if np.isfinite(day_move) else ""
)

meta_html = f"Trend regime for <b>{streak} session{'s' if streak != 1 else ''}</b>"
if np.isfinite(stickiness):
    meta_html += f", persistence <b>{stickiness:.0f}/100</b>"

if v11_latest:
    fc_probs = {
        "BULL": v11_latest["bull_probability"],
        "SIDE": v11_latest["side_probability"],
        "BEAR": v11_latest["bear_probability"],
    }
    fc_pick, fc_conf = v11_latest["next_regime"], v11_latest["next_day_probability"]
    fc_when = f"Next session, {nice_day(v11_latest['target_trading_date'])}"
    fc_model = f"V11 forecast, {str(v11_latest['status']).lower()}"
else:
    fc_probs = {s: probabilities.get(s, 0.0) for s in states} if probabilities else None
    fc_pick, fc_conf = next_regime, next_probability
    fc_when, fc_model = "Next session", "V10 trend model"

ribbon_head = (
    f'<span>{theme.esc(fc_when)}: <b class="t-{theme.tone_of(fc_pick)}">{theme.esc(theme.word_of(fc_pick))}</b></span>'
    + (f"<span><b>{fc_conf:.0%}</b> confidence, {theme.esc(fc_model)}</span>" if np.isfinite(fc_conf) else "")
)

theme.regime_hero(
    instrument=NAME,
    regime=current,
    meta_html=meta_html,
    price_label=price_label,
    price=price_text,
    change_html=change_html,
    live=live,
    ribbon_head=ribbon_head,
    probs=fc_probs,
    pick=fc_pick,
)

if v11_latest:
    theme.note(
        "V11 forecasts the next close-to-close move from the overnight US session, India's end-of-day "
        "state and the HMM regime: Bull above +0.25%, Bear below −0.25%, otherwise Sideways. "
        "It becomes final once the US session has closed, around 02:30 IST."
    )
if IS_VIX:
    theme.note(
        "India VIX regimes read as volatility levels: the model's Bear, Sideways and Bull states "
        "are shown as Low, Mid and High vol."
    )


# ============================================================
# REGIME TIMELINE
# ============================================================

hist10 = load_hmm_prediction_history(selected_symbol)
closes = load_daily_closes(selected_symbol)
if not hist10.empty and len(closes) and "current_regime" in hist10:
    theme.section(
        "Regime timeline",
        "Daily close over the last two years. The strip along the bottom shows the trend regime the model "
        "assigned each day.",
    )
    reg = (
        hist10.assign(day=hist10["prediction_date"].dt.normalize())
        .drop_duplicates("day", keep="last")
        .set_index("day")["current_regime"]
        .map(lambda r: display_regime(selected_symbol, str(r).upper()))
    )
    px_ = closes[closes.index >= reg.index.min()]
    reg_px = reg.reindex(px_.index, method="ffill")

    timeline = go.Figure()
    runs = (reg_px != reg_px.shift()).cumsum()
    dates = list(px_.index)
    for _, seg in reg_px.groupby(runs):
        r = seg.iloc[0]
        if pd.isna(r):
            continue
        end_pos = dates.index(seg.index[-1])
        x1 = dates[end_pos + 1] if end_pos + 1 < len(dates) else seg.index[-1]
        timeline.add_vrect(
            x0=seg.index[0], x1=x1, y0=-0.07, y1=-0.025, line_width=0, layer="above",
            fillcolor=theme.TONES[theme.tone_of(r)], opacity=0.9,
        )
    timeline.add_scatter(
        x=px_.index, y=px_.values, mode="lines", name="Close", showlegend=False,
        line=dict(color=theme.ACCENT, width=1.6),
        customdata=reg_px.map(lambda r: theme.word_of(r) if isinstance(r, str) else "—").to_numpy(),
        hovertemplate="%{x|%a %d %b %Y}<br><b>%{y:,.2f}</b><br>%{customdata} regime<extra></extra>",
    )
    for s in states:
        timeline.add_scatter(
            x=[None], y=[None], mode="markers", name=theme.word_of(s), hoverinfo="skip",
            marker=dict(size=11, symbol="square", color=theme.TONES[theme.tone_of(s)]),
        )
    timeline.update_layout(
        height=380, hovermode="closest",
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", itemclick=False, itemdoubleclick=False),
        margin=dict(l=8, r=64, t=34, b=36),
        xaxis=dict(showgrid=False, automargin=True, ticklabelstandoff=22),
        yaxis=dict(side="right", tickformat=",.0f", automargin=True),
    )
    with theme.card("timeline"):
        theme.chart(timeline)


# ============================================================
# PERSISTENCE + REGIME DYNAMICS
# ============================================================

matrix = get_transition_matrix(selected)
stationary = get_stationary(selected)
if IS_VIX:
    matrix = remap_vix_matrix(matrix)
    stationary = remap_vix_probs(stationary)

left, right = st.columns([1, 1.75], gap="large")

with left:
    theme.section("Persistence", "How entrenched the current regime is. It says nothing about direction.")
    theme.meter(stickiness, stickiness_level)
    stay = as_float(first_value(selected, ["stay_probability"]))
    consistency = as_float(first_value(selected, ["recent_consistency"]))
    theme.stats([
        theme.stat("Stay probability", f"{stay:.0%}" if np.isfinite(stay) else "N/A"),
        theme.stat("10-day consistency", f"{consistency:.0%}" if np.isfinite(consistency) else "N/A"),
        theme.stat("Streak", f"{streak} d"),
    ], enter=False, cols=3)
    theme.note(
        "Score = 60% stay probability + 25% ten-session consistency + 15% streak (capped at 10 sessions)."
    )

with right:
    theme.section("Regime dynamics", "How each regime behaves, from the trailing three-year transition process.")
    if matrix is not None:
        matrix = matrix.reindex(index=states, columns=states)
        rows, current_row = [], None
        for i, regime in enumerate(states):
            tone = theme.tone_of(regime)
            stay_p = float(matrix.loc[regime, regime])
            exits = matrix.loc[regime].drop(regime)
            if regime == current:
                current_row = i
            share = stationary.get(regime, np.nan) if stationary else np.nan
            rows.append([
                theme.chip(regime) + (' <span class="dim">now</span>' if regime == current else ""),
                theme.mini_bar(stay_p, f"{stay_p:.0%}", tone),
                f"{1 / max(1 - stay_p, 1e-6):.1f} d",
                (theme.chip(exits.idxmax()) + f' <span class="dim">{exits.max() / exits.sum():.0%}</span>')
                if exits.sum() > 0 else '<span class="dim">—</span>',
                f"{share:.0%}" if np.isfinite(share) else "—",
            ])
        theme.table(
            [("Regime", ""), ("Continues", ""), ("Lasts", "num"), ("Usually ends in", ""), ("Long-run", "num")],
            rows,
            current=current_row,
        )
        theme.note(
            "Continues = chance the regime holds into the next session. Lasts = typical duration in sessions, "
            "1 / (1 − continues). Long-run = share of time spent in the regime over the full history."
        )
    else:
        theme.note("Transition data is not available for this instrument yet.")


# ============================================================
# MARKET OVERVIEW (all instruments)
# ============================================================

overview = build_market_overview(all_data)
if not overview.empty:
    theme.section("Market overview", "Every tracked index at a glance. The highlighted row is the one selected.")
    rows, current_row = [], None
    for i, r in enumerate(overview.to_dict("records")):
        if r["Symbol"] == selected_symbol:
            current_row = i
        persistence = r["Persistence"] if np.isfinite(r["Persistence"]) else np.nan
        rows.append([
            f'<span class="strong">{theme.esc(r["Instrument"])}</span>',
            f"{r['Last close']:,.2f}" if np.isfinite(r["Last close"]) else "—",
            theme.signed_html(r["1D change"], invert=r["Symbol"] == "INDIA_VIX"),
            theme.chip(r["Regime"]),
            f"{r['Streak']} sessions",
            theme.mini_bar(persistence / 100, f"{persistence:.0f}" if np.isfinite(persistence) else "—"),
            theme.chip(r["Next session"]) + f' <span class="dim">{r["Model"]}</span>',
        ])
    theme.table(
        [("Instrument", ""), ("Last close", "num"), ("1D change", "num"), ("Regime", ""),
         ("Streak", "num"), ("Persistence", ""), ("Next session", "")],
        rows,
        current=current_row,
    )
    theme.note(
        "Regime is the V10 trend regime. Next session uses the V11 forecast for NIFTY 50 and NIFTY BANK "
        "and the V10 trend model for the others."
    )


# ============================================================
# PREDICTION LOG
# ============================================================

theme.section(
    "Prediction log",
    f"Every logged next-session forecast for {NAME} over the last two years, scored against the close that followed.",
)

log_model = "V10"
if selected_symbol in V11_SYMBOLS and not load_v11_history(selected_symbol).empty:
    log_model = st.segmented_control(
        "Model",
        ["V11", "V10"],
        default="V11",
        key="log_model",
        label_visibility="collapsed",
        help="V11 is the pre-open next-session forecaster. V10 is the original trend-regime HMM.",
    ) or "V11"

if log_model == "V11":
    prediction_history = load_v11_history(selected_symbol)
    history_file = V11_DIR / f"prediction_history_v11_{selected_symbol.lower()}.csv"
else:
    prediction_history = load_hmm_prediction_history(selected_symbol)
    history_file = history_file_for_symbol(selected_symbol)

if history_file is not None and not prediction_history.empty:
    history = prediction_history.copy()
    for col in ["bull_probability", "side_probability", "bear_probability"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    # Scorecard: predicted regime vs the actual next-session close.
    scored = prediction_history.assign(
        move=prediction_history["target_trading_date"].dt.normalize().map(moves)
    ).dropna(subset=["move"])
    if not scored.empty and not IS_VIX:
        calls = scored[scored["next_regime"].isin(["BULL", "BEAR"])]
        hits = ((calls["next_regime"] == "BULL") & (calls["move"] > 0)) | (
            (calls["next_regime"] == "BEAR") & (calls["move"] < 0)
        )
        actual3 = np.where(scored["move"] > 0.25, "BULL", np.where(scored["move"] < -0.25, "BEAR", "SIDE"))
        base_up = (scored["move"] > 0).mean()
        direction = hits.mean() if len(calls) else np.nan
        theme.stats([
            theme.stat(
                "Direction accuracy",
                f"{direction:.1%}" if np.isfinite(direction) else "N/A",
                sub=(f"{(direction - base_up) * 100:+.1f} pts vs always-up" if np.isfinite(direction) else None),
                tone=("bull" if np.isfinite(direction) and direction > base_up else None),
                help="Bull call and the next close is up, or Bear call and it is down. Sideways calls are excluded.",
            ),
            theme.stat(
                "3-class accuracy",
                f"{(scored['next_regime'].values == actual3).mean():.1%}",
                help="Actual: Bull above +0.25%, Bear below −0.25%, otherwise Sideways.",
            ),
            theme.stat("Always-up baseline", f"{base_up:.1%}", help="Share of sessions that closed higher."),
            theme.stat("Scored sessions", f"{len(scored):,}", sub=f"{len(calls):,} Bull or Bear calls"),
        ])

    detailed = history[
        ["prediction_date", "target_trading_date", "current_regime", "next_regime",
         "bull_probability", "side_probability", "bear_probability"]
    ].copy()

    target_dates = detailed["target_trading_date"].dt.normalize()
    close_columns = []
    for sym, label in NEXT_DAY_CLOSE_SYMBOLS.items():
        column = f"Next close · {label}"
        detailed[column] = target_dates.map(load_daily_close_changes(sym))
        close_columns.append(column)

    detailed = detailed.sort_values("prediction_date", ascending=False)
    detailed["prediction_date"] = detailed["prediction_date"].dt.strftime("%Y-%m-%d")
    detailed["target_trading_date"] = detailed["target_trading_date"].dt.strftime("%Y-%m-%d")
    for col in ["current_regime", "next_regime"]:
        detailed[col] = detailed[col].map(lambda r: display_regime(selected_symbol, str(r).upper()))
    detailed = detailed.rename(columns={
        "prediction_date": "Predicted on",
        "target_trading_date": "For session",
        "current_regime": "Regime then",
        "next_regime": "Forecast",
        "bull_probability": "P(Bull)",
        "side_probability": "P(Side)",
        "bear_probability": "P(Bear)",
    })

    tone_css = {k: f"color: {theme.TONES[v]}" for k, v in theme.REGIME_TONE.items()}

    for column in close_columns:   # text, so pending sessions read "Pending" rather than an empty cell
        detailed[column] = detailed[column].map(lambda v: "Pending" if pd.isna(v) else f"{v:+.2f}%")

    def _move_colour(value):
        if value.startswith("+") and value != "+0.00%":
            return f"color: {theme.BULL}"
        if value.startswith("-"):
            return f"color: {theme.BEAR}"
        return f"color: {theme.FAINT}"

    styled = (
        detailed.style
        .format("{:.1%}", subset=["P(Bull)", "P(Side)", "P(Bear)"], na_rep="—")
        .map(_move_colour, subset=close_columns)
        .map(lambda r: tone_css.get(r, ""), subset=["Regime then", "Forecast"])
    )
    st.dataframe(styled, width="stretch", hide_index=True, height=460)
    theme.note(
        f"Source: {history_file.name}. Next close is the close-to-close move of NIFTY 50 and BANK NIFTY on "
        "the forecast session (Upstox daily closes). Pending means that session has not closed yet. "
        "V11 rows are walk-forward: each was predicted by a model trained only on earlier data."
    )
else:
    theme.note(
        f"There is no prediction log for {NAME} yet. NIFTY 50 history is not substituted here, "
        "because that would show the wrong instrument's record."
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.space("medium")
with st.expander("Methodology"):
    st.markdown(
        """
        **Trend regime (V10).** Each index is modelled with a three-state
        Gaussian HMM (Bull / Sideways / Bear) using causal filtering, so today's
        regime only uses information available through today's close.

        **Next-session forecast (V11, NIFTY 50 and NIFTY BANK).** Forecasts the
        next close-to-close move (Bull > +0.25%, Bear < −0.25%, otherwise Sideways)
        from the US session that trades after India's close, India's
        end-of-day state and a causal HMM regime. Trained walk-forward; every
        logged prediction was made by a model fit only on earlier data.

        **Persistence.** 60% stay probability + 25% ten-session consistency +
        15% streak. It describes how entrenched a regime is, not the direction
        of the next move.

        **Regime dynamics.** Continuation probabilities come from the trailing
        transition process; typical duration is 1 / (1 − continuation).

        **India VIX** regimes are shown as Low / Mid / High volatility.
        """
    )
