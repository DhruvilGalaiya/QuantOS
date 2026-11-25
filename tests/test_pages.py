"""Smoke tests: every dashboard page renders without exceptions (needs local data)."""

from pathlib import Path

import pytest

from tests.conftest import has_local_data

ROUTER = str(Path(__file__).resolve().parents[1] / "backend" / "app" / "regime_dashboard_v10.py")
PAGES = ["views/today.py", "views/volatility_edge.py", "views/event_lab.py", "views/market_states.py", "views/risk_monitor.py", "views/market_regimes.py", "views/volatility_surface.py", "views/us_stocks.py"]

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not has_local_data(), reason="no local data directory")]


@pytest.fixture
def public_app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from quantos_platform.core import settings
    monkeypatch.setenv("QUANTOS_MODE", "public")
    monkeypatch.setattr(settings, "PUBLIC", True)
    monkeypatch.syspath_prepend(str(Path(ROUTER).parent))   # streamlit adds the app dir too
    import ui
    monkeypatch.setattr(ui, "PUBLIC", True)
    at = AppTest.from_file(ROUTER, default_timeout=300)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_read_only(public_app, page):
    public_app.switch_page(page).run()
    assert not public_app.exception, [e.value for e in public_app.exception]
    # public mode: no buttons that trigger data refreshes
    labels = [b.label for b in public_app.button]
    assert not any(w in " ".join(labels).lower() for w in ("pull", "refresh")), labels
