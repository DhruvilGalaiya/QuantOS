from __future__ import annotations

import sys
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from quantos_volatility.option_analytics import (
    load_latest_snapshot,
    build_option_chain,
    get_available_expiries,
    get_option_summary,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="QuantOS Options Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MINIMAL CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.live-dot {
    color: #18a957;
    font-weight: 700;
}

.snapshot-info {
    color: #7a828c;
    font-size: 0.78rem;
    margin-top: 0.3rem;
    margin-bottom: 1rem;
}

.chain-wrapper {
    width: 100%;
    overflow-x: auto;
    overflow-y: auto;
    max-height: 700px;
    border: 1px solid #d9d9d9;
    border-radius: 8px;
    background: white;
}

.option-chain {
    width: 100%;
    min-width: 1200px;
    border-collapse: collapse;
    font-size: 0.78rem;
    color: #202124;
}

.option-chain .group-header {
    background: #f5f5f5;
    color: #303030;
    font-size: 0.9rem;
    font-weight: 700;
    text-align: center;
    padding: 10px 6px;
    border-bottom: 1px solid #d6d6d6;
}

.option-chain .column-header {
    background: #fafafa;
    color: #6b6b6b;
    font-weight: 600;
    text-align: center;
    padding: 8px 6px;
    border-bottom: 1px solid #dedede;
    white-space: nowrap;
}

.option-chain td {
    padding: 8px 6px;
    text-align: right;
    border-bottom: 1px solid #eeeeee;
    white-space: nowrap;
}

.option-chain td.strike-cell {
    text-align: center;
    font-weight: 700;
    background: #f7f7f7;
    border-left: 1px solid #d8d8d8;
    border-right: 1px solid #d8d8d8;
}

.option-chain tr.atm-row td {
    background: #fff2b3 !important;
    border-top: 1px solid #e2c84d;
    border-bottom: 1px solid #e2c84d;
}

.option-chain tr.atm-row td.strike-cell {
    background: #ffe477 !important;
    color: #202020;
    font-weight: 800;
}

.atm-label {
    margin-left: 5px;
    padding: 2px 5px;
    border-radius: 3px;
    background: #e5c84b;
    color: #202020;
    font-size: 0.62rem;
    font-weight: 800;
}

.na-value {
    color: #b0b0b0;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_latest_snapshot()

if df.empty:
    st.error("No live option snapshots are currently available.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")

symbols = sorted(
    df["symbol"].dropna().unique().tolist()
)

selected_symbol = st.sidebar.selectbox(
    "Underlying",
    symbols,
)

symbol_df = df[
    df["symbol"] == selected_symbol
].copy()

expiries = get_available_expiries(symbol_df)

if not expiries:
    st.error(f"No expiries available for {selected_symbol}.")
    st.stop()

selected_expiry = st.sidebar.selectbox(
    "Expiry",
    expiries,
)

strike_range = st.sidebar.slider(
    "Strikes around ATM",
    min_value=5,
    max_value=15,
    value=15,
    step=1,
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "Refresh Data",
    use_container_width=True,
):
    st.rerun()


# ============================================================
# BUILD OPTION CHAIN
# ============================================================

chain = build_option_chain(
    symbol_df,
    expiry=selected_expiry,
    strike_range=strike_range,
)

if chain.empty:
    st.warning("No option data available for this selection.")
    st.stop()


# ============================================================
# SUMMARY
# ============================================================

summary = get_option_summary(symbol_df)

spot = summary.get("spot")
oi_pcr = summary.get("oi_pcr")
atm_iv = summary.get("atm_iv")
snapshot_ts = summary.get("snapshot_ts")
iv_completeness = summary.get("iv_completeness")


# ============================================================
# ATM STRIKE
# ============================================================

valid_strikes = (
    chain["strike"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .to_numpy()
)

if len(valid_strikes) == 0:
    st.error("No valid strikes available.")
    st.stop()

if spot is not None and np.isfinite(spot):

    atm_strike = float(
        valid_strikes[
            np.argmin(
                np.abs(
                    valid_strikes - float(spot)
                )
            )
        ]
    )

else:

    atm_strike = float(
        valid_strikes[
            len(valid_strikes) // 2
        ]
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "QuantOS Options & Volatility Intelligence"
)

st.caption(
    "Live options market analytics powered by Upstox "
    " · "
    "🟢 LIVE"
)


# ============================================================
# TOP ROW
# ============================================================

left_top, right_top = st.columns(2)

with left_top:

    st.caption("Underlying")

    st.subheader(
        selected_symbol
    )


with right_top:

    st.caption("Spot Price")

    if spot is not None and np.isfinite(spot):

        st.subheader(
            f"{float(spot):,.2f}"
        )

    else:

        st.subheader("N/A")


# ============================================================
# SECOND ROW
# ============================================================

st.markdown(
    "<div style='height:0.3rem'></div>",
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3 = st.columns(3)

with metric_1:

    st.caption("Expiry")

    st.subheader(
        str(selected_expiry)
    )


with metric_2:

    st.caption("P/C OI Ratio")

    if oi_pcr is not None and np.isfinite(oi_pcr):

        st.subheader(
            f"{float(oi_pcr):.3f}"
        )

    else:

        st.subheader("N/A")


with metric_3:

    st.caption("ATM IV")

    if atm_iv is not None and np.isfinite(atm_iv):

        st.subheader(
            f"{float(atm_iv) * 100:.2f}%"
        )

    else:

        st.subheader("N/A")


# ============================================================
# SNAPSHOT INFO
# ============================================================

if (
    iv_completeness is not None
    and np.isfinite(iv_completeness)
):

    completeness = (
        f"{float(iv_completeness) * 100:.1f}%"
    )

else:

    completeness = "N/A"


st.caption(
    f"Latest snapshot: {snapshot_ts} "
    f" | IV completeness: {completeness}"
)

st.divider()


# ============================================================
# OPTION CHAIN TITLE
# ============================================================

st.subheader(
    f"Option Chain · {selected_symbol} · {selected_expiry}"
)


# ============================================================
# CALLS / PUTS
# ============================================================

calls = chain[
    chain["option_type"] == "CE"
].copy()

puts = chain[
    chain["option_type"] == "PE"
].copy()

calls = calls.set_index("strike")
puts = puts.set_index("strike")

strikes = sorted(
    chain["strike"]
    .dropna()
    .unique()
)


# ============================================================
# FORMATTERS
# ============================================================

def missing(value):

    try:
        return pd.isna(value)
    except Exception:
        return True


def fmt_price(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value):,.2f}"


def fmt_integer(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value):,.0f}"


def fmt_iv(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value) * 100:.2f}%"


def fmt_delta(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value):.4f}"


def fmt_gamma(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value):.6f}"


def fmt_theta(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value):.4f}"


def fmt_vega(value):

    if missing(value):
        return '<span class="na-value">—</span>'

    return f"{float(value):.4f}"


# ============================================================
# HTML TABLE
# ============================================================

html = []

html.append(
    '<div class="chain-wrapper">'
)

html.append(
    '<table class="option-chain">'
)

html.append(
    """
<thead>

<tr>

<th class="group-header" colspan="8">
CALLS
</th>

<th class="group-header">
STRIKE
</th>

<th class="group-header" colspan="8">
PUTS
</th>

</tr>

<tr>

<th class="column-header">LTP</th>
<th class="column-header">Volume</th>
<th class="column-header">OI</th>
<th class="column-header">IV</th>
<th class="column-header">Delta</th>
<th class="column-header">Gamma</th>
<th class="column-header">Theta</th>
<th class="column-header">Vega</th>

<th class="column-header">Strike</th>

<th class="column-header">Vega</th>
<th class="column-header">Theta</th>
<th class="column-header">Gamma</th>
<th class="column-header">Delta</th>
<th class="column-header">IV</th>
<th class="column-header">OI</th>
<th class="column-header">Volume</th>
<th class="column-header">LTP</th>

</tr>

</thead>

<tbody>
"""
)


# ============================================================
# TABLE ROWS
# ============================================================

for strike in strikes:

    call = (
        calls.loc[strike]
        if strike in calls.index
        else None
    )

    put = (
        puts.loc[strike]
        if strike in puts.index
        else None
    )

    is_atm = np.isclose(
        float(strike),
        float(atm_strike),
    )

    row_class = (
        "atm-row"
        if is_atm
        else ""
    )


    # --------------------------------------------------------
    # CALLS
    # --------------------------------------------------------

    if call is not None:

        call_ltp = fmt_price(
            call.get("ltp")
        )

        call_volume = fmt_integer(
            call.get("volume")
        )

        call_oi = fmt_integer(
            call.get("oi")
        )

        call_iv = fmt_iv(
            call.get("iv")
        )

        call_delta = fmt_delta(
            call.get("delta")
        )

        call_gamma = fmt_gamma(
            call.get("gamma")
        )

        call_theta = fmt_theta(
            call.get("theta")
        )

        call_vega = fmt_vega(
            call.get("vega")
        )

    else:

        call_ltp = '<span class="na-value">—</span>'
        call_volume = '<span class="na-value">—</span>'
        call_oi = '<span class="na-value">—</span>'
        call_iv = '<span class="na-value">—</span>'
        call_delta = '<span class="na-value">—</span>'
        call_gamma = '<span class="na-value">—</span>'
        call_theta = '<span class="na-value">—</span>'
        call_vega = '<span class="na-value">—</span>'


    # --------------------------------------------------------
    # PUTS
    # --------------------------------------------------------

    if put is not None:

        put_vega = fmt_vega(
            put.get("vega")
        )

        put_theta = fmt_theta(
            put.get("theta")
        )

        put_gamma = fmt_gamma(
            put.get("gamma")
        )

        put_delta = fmt_delta(
            put.get("delta")
        )

        put_iv = fmt_iv(
            put.get("iv")
        )

        put_oi = fmt_integer(
            put.get("oi")
        )

        put_volume = fmt_integer(
            put.get("volume")
        )

        put_ltp = fmt_price(
            put.get("ltp")
        )

    else:

        put_vega = '<span class="na-value">—</span>'
        put_theta = '<span class="na-value">—</span>'
        put_gamma = '<span class="na-value">—</span>'
        put_delta = '<span class="na-value">—</span>'
        put_iv = '<span class="na-value">—</span>'
        put_oi = '<span class="na-value">—</span>'
        put_volume = '<span class="na-value">—</span>'
        put_ltp = '<span class="na-value">—</span>'


    # --------------------------------------------------------
    # STRIKE
    # --------------------------------------------------------

    atm_label = (
        '<span class="atm-label">ATM</span>'
        if is_atm
        else ""
    )

    strike_display = (
        f"{float(strike):,.0f}"
        f"{atm_label}"
    )


    # --------------------------------------------------------
    # ROW
    # --------------------------------------------------------

    html.append(
        f"""
<tr class="{row_class}">

<td>{call_ltp}</td>
<td>{call_volume}</td>
<td>{call_oi}</td>
<td>{call_iv}</td>
<td>{call_delta}</td>
<td>{call_gamma}</td>
<td>{call_theta}</td>
<td>{call_vega}</td>

<td class="strike-cell">
{strike_display}
</td>

<td>{put_vega}</td>
<td>{put_theta}</td>
<td>{put_gamma}</td>
<td>{put_delta}</td>
<td>{put_iv}</td>
<td>{put_oi}</td>
<td>{put_volume}</td>
<td>{put_ltp}</td>

</tr>
"""
    )


# ============================================================
# CLOSE TABLE
# ============================================================

html.append(
    """
</tbody>
</table>
</div>
"""
)


# ============================================================
# RENDER TABLE
# ============================================================

st.html(
    "".join(html)
)


# ============================================================
# FOOTER SUMMARY
# ============================================================

st.markdown(
    "<div style='height:1rem'></div>",
    unsafe_allow_html=True,
)

footer_1, footer_2, footer_3 = st.columns(3)

with footer_1:

    st.metric(
        label="Strikes Displayed",
        value=len(strikes),
    )


with footer_2:

    st.metric(
        label="Call Contracts",
        value=len(calls),
    )


with footer_3:

    st.metric(
        label="Put Contracts",
        value=len(puts),
    )


# ============================================================
# FOOTER NOTE
# ============================================================

st.caption(
    "QuantOS uses live market snapshots stored locally in SQLite. "
    "Missing market fields are displayed as unavailable values."
)