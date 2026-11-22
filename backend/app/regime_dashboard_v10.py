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


def _drop_stale_modules() -> None:
    """Streamlit reruns page scripts after a deploy but keeps already-imported modules in memory, so a page
    can meet an old copy of quantos_platform / theme / ui (e.g. a missing settings attribute). Forget any of
    our modules whose file changed since it was imported, so the next import loads the new code."""
    seen = sys.__dict__.setdefault("_quantos_module_mtimes", {})
    first_run = not seen
    for name, mod in list(sys.modules.items()):
        if not (name == "quantos_platform" or name.startswith("quantos_platform.") or name in ("theme", "ui")):
            continue
        path = getattr(mod, "__file__", None)
        try:
            mtime = Path(path).stat().st_mtime if path else None
        except OSError:
            mtime = None
        if first_run or (mtime is not None and seen.get(name) not in (None, mtime)):
            sys.modules.pop(name, None)
            seen.pop(name, None)
        elif mtime is not None:
            seen.setdefault(name, mtime)
    seen["__init__"] = True


_drop_stale_modules()

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
