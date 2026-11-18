"""
QuantOS dashboard entry point (page router).

    python -m streamlit run backend/app/regime_dashboard_v10.py

Data is produced by scheduled jobs (python -m quantos_platform.jobs ...); pages only read it.
Set QUANTOS_MODE=public for the read-only public deployment.

Pages:
    views/market_regimes.py   Indian index regimes, V10 HMM + V11 forecast, prediction log
    views/volatility_surface.py  IV surface (SVI/SABR), smiles, skew, term structure
    views/expiry_playbook.py  expiry-week overnight option signal + paper log
    views/us_stocks.py        US stock price history
"""

import sys
from pathlib import Path

import streamlit as st

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:   # fallback when the package isn't pip-installed
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="QuantOS",
    page_icon="◈",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page(
            "views/market_regimes.py",
            title="Market Regimes",
            icon=":material/insights:",
            default=True,
        ),
        st.Page(
            "views/volatility_surface.py",
            title="Volatility Surface",
            icon=":material/stacked_line_chart:",
        ),
        st.Page(
            "views/expiry_playbook.py",
            title="Expiry Playbook",
            icon=":material/event_available:",
        ),
        st.Page(
            "views/us_stocks.py",
            title="US Stocks",
            icon=":material/public:",
        ),
    ]
)
navigation.run()
