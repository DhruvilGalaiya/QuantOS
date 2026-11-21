"""
QuantOS dashboard entry point (page router).

    python -m streamlit run backend/app/regime_dashboard_v10.py

Data is produced by scheduled jobs (python -m quantos_platform.jobs ...); pages only read it.
Set QUANTOS_MODE=public for the read-only public deployment.

Pages:
    views/today.py            market brief (default): price context, volatility premium, events this week
    views/market_regimes.py   Indian index regimes, V10 HMM + V11 forecast, prediction log
    views/volatility_edge.py  volatility risk premium, vol cone, GARCH-t, rough vol, straddle decay
    views/event_lab.py        event studies: what happened after 52-week highs, streaks, VIX spikes, ...
    views/volatility_surface.py  IV surface (SVI/SABR), smiles, skew, term structure
    views/us_stocks.py        US stock price history
"""

import sys
from pathlib import Path

import streamlit as st

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:   # fallback when the package isn't pip-installed
    sys.path.insert(0, _ROOT)

import theme  # noqa: E402  (backend/app/theme.py)

st.set_page_config(
    page_title="QuantOS",
    page_icon=str(theme.ASSETS / "mark.svg"),
    layout="wide",
)
theme.apply()

navigation = st.navigation(
    [
        st.Page(
            "views/today.py",
            title="Today",
            icon=":material/today:",
            default=True,
        ),
        st.Page(
            "views/market_regimes.py",
            title="Market Regimes",
            icon=":material/insights:",
        ),
        st.Page(
            "views/volatility_edge.py",
            title="Volatility & Edge",
            icon=":material/ssid_chart:",
        ),
        st.Page(
            "views/event_lab.py",
            title="Event Lab",
            icon=":material/science:",
        ),
        st.Page(
            "views/volatility_surface.py",
            title="Volatility Surface",
            icon=":material/stacked_line_chart:",
        ),
        st.Page(
            "views/us_stocks.py",
            title="US Stocks",
            icon=":material/public:",
        ),
    ],
    position="top",
)
navigation.run()
