"""
QuantOS | Volatility dashboard, standalone entry point.

The page now lives in the main dashboard (backend/app/views/volatility_surface.py):

    python -m streamlit run backend/app/regime_dashboard_v10.py      # all pages (recommended)
    python -m streamlit run src/quantos_volatility/volatility_surface_dashboard.py   # this page only
"""

import runpy
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="QuantOS Volatility", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")

import sys  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "backend" / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
PAGE = _ROOT / "backend" / "app" / "views" / "volatility_surface.py"
runpy.run_path(str(PAGE), run_name="__main__")
