"""Market Regimes, Volatility & Edge and US Stocks: services, routes and the US fundamentals snapshot."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantos_platform.api.app import app
from quantos_platform.services import regimes, us_stocks
from tests.conftest import has_local_data

client = TestClient(app)
local = pytest.mark.skipif(not has_local_data(), reason="no local data directory")


def test_vix_regimes_read_as_volatility_levels():
    assert regimes._disp("INDIA_VIX", "BULL") == "HIGH"
    assert regimes._disp("INDIA_VIX", "bear") == "LOW"
    assert regimes._disp("NIFTY_50", "SIDE") == "SIDE"
    assert regimes.states_for("INDIA_VIX") == ["LOW", "MID", "HIGH"]


def test_vix_matrix_is_reordered_low_to_high():
    d = {"transition_matrix": {"index": ["BULL", "SIDE", "BEAR"], "columns": ["BULL", "SIDE", "BEAR"],
                               "data": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]}}
    m = regimes._matrix("INDIA_VIX", d)
    assert m[0] == [0.8, 0.15, 0.05]          # LOW row = old BEAR row, columns BEAR, SIDE, BULL
    assert regimes._matrix("NIFTY_50", d)[0] == [0.7, 0.2, 0.1]


def test_ratio_sentences_use_the_numbers():
    p = pd.Series([100.0 + i for i in range(30)], index=pd.date_range("2026-01-01", periods=30))
    f = {"currency": "USD", "price": 129.0, "pe": 25.0, "net_margin": 0.21, "revenue_growth": -0.05,
         "debt_to_equity": 0.5, "dividend_yield": 0.0, "high_52w": 150.0, "low_52w": 90.0}
    r = {x["key"]: x for x in us_stocks._ratios(f, p)}
    assert r["pe"]["plain"].startswith("Investors pay $25 for every $1")
    assert "Keeps $21" in r["net_margin"]["plain"]
    assert "5% lower" in r["revenue_growth"]["plain"]
    assert r["dividend_yield"]["plain"] == "Pays no regular dividend."
    assert "below its 52-week high" in r["range52"]["plain"]


def test_summary_splits_after_two_sentences():
    s = us_stocks._summary("Acme makes rockets. It sells to NASA. It was founded in 1990. Its HQ is in Ohio.")
    assert s["lead"] == "Acme makes rockets. It sells to NASA."
    assert s["rest"] == "It was founded in 1990. Its HQ is in Ohio."


def test_bad_ids_are_rejected():
    assert client.get("/internal/v1/regimes/NOPE").status_code == 404
    assert client.get("/internal/v1/volatility/NIFTY_IT").status_code == 404
    assert client.get("/internal/v1/regimes/nifty;x").status_code == 422
    assert client.get("/internal/v1/us/ZZZZZZ").status_code == 404


def test_snapshot_round_trip(tmp_path, monkeypatch):
    from quantos_platform.data import us_snapshot
    monkeypatch.setattr(us_snapshot, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr("quantos_platform.data.us_fundamentals.fetch_fundamentals", lambda t: {"name": "Test Co", "pe": 10.0})
    monkeypatch.setattr("quantos_platform.data.us_fundamentals.fetch_financials", lambda t: {"annual": {"columns": [], "rows": []}})
    assert us_snapshot.is_stale("TEST")
    res = us_snapshot.refresh(["TEST"])
    assert res == {"updated": 1, "failed": {}}
    assert us_snapshot.load("test")["fundamentals"]["name"] == "Test Co"
    assert not us_snapshot.is_stale("TEST")
    assert us_snapshot.refresh(["TEST"])["updated"] == 0      # fresh files are not refetched


@pytest.mark.slow
@local
@pytest.mark.parametrize("symbol", ["NIFTY_50", "INDIA_VIX", "NIFTY_IT"])
def test_regimes_route(symbol):
    r = client.get(f"/internal/v1/regimes/{symbol}")
    assert r.status_code == 200
    b = r.json()
    assert b["current"]["regime"] in b["states"]
    assert all(p["regime"] in b["states"] for p in b["timeline"])
    assert all(abs(sum(row) - 1) < 1e-6 for row in b["matrix"])
    assert {o["symbol"] for o in b["overview"]} >= {"NIFTY_50", "INDIA_VIX"}


@pytest.mark.slow
@local
def test_volatility_route():
    b = client.get("/internal/v1/volatility/NIFTY_50").json()
    assert b["vrp"]["series"] and b["cone"] and b["garch"]["term_structure"]
    assert all(r["tdte"] <= 5 for r in b["decay"]["rows"])
    assert len(json.dumps(b)) < 400_000        # thinned for the browser
    assert client.get("/internal/v1/volatility/NIFTY_BANK").json()["vrp"] is None


@pytest.mark.slow
@local
def test_us_routes():
    u = client.get("/internal/v1/us").json()
    assert len(u["stocks"]) > 50 and u["stocks"][0]["market_cap"] >= u["stocks"][-1]["market_cap"]
    c = client.get("/internal/v1/us/aapl").json()
    assert c["ticker"] == "AAPL" and c["history"] and c["returns"]["y1"] is not None


# ------------------------------------------------------------------ states, events, risk, surface
def test_unknown_ids_on_new_routes():
    assert client.get("/internal/v1/states/SENSEX").status_code == 404
    assert client.get("/internal/v1/events/NIFTY_IT").status_code == 404
    assert client.get("/internal/v1/surface/NIFTY_IT").status_code == 404


def test_events_unusual_rule(monkeypatch):
    from quantos_platform.services import events as ev
    summary = {"as_of": "2026-01-02", "since": "2007-01-02", "sessions": 10, "horizons": [21], "n_boot": 100, "recent": [],
               "events": {"a": {"id": "a", "name": "A", "description": "d", "group": "g", "min_gap": 1, "count": 30, "last": None,
                                "stats": {"h21": {"n": 30, "p_value": 0.01}, "h1": {"n": 5, "p_value": 0.01}},
                                "path": {"offsets": [0], "mean": [0], "ci_low": [0], "ci_high": [0], "baseline": [0]}}}}
    monkeypatch.setattr(ev, "read_json", lambda name: summary)
    monkeypatch.setattr(ev, "read_frame", lambda name: pd.DataFrame(columns=["event", "date", "close", "fwd_1", "fwd_5", "fwd_21", "fwd_63"]))
    out = ev.build("NIFTY_50")
    st = out["events"][0]["stats"]
    assert st["h21"]["unusual"] is True and st["h1"]["unusual"] is False      # too few cases at 1 day


@pytest.mark.slow
@local
def test_states_route():
    b = client.get("/internal/v1/states/NIFTY_50").json()
    assert b["current"]["state"] in b["states"]
    assert {x["state"] for x in b["behaviour"]} == {"CALM", "NORMAL", "STRESSED", "ALL"}
    assert all(abs(sum(r) - 1) < 1e-6 for r in b["transition"]["p"])
    assert b["drawdown"]["base_rates"][0]["from"] == 0.05                  # alias kept in the JSON
    assert all(f["depth"] <= -0.10 for f in b["drawdown"]["falls"])


@pytest.mark.slow
@local
def test_risk_route_has_replayable_crashes():
    b = client.get("/internal/v1/risk").json()
    assert b["now"]["avg_corr"] is not None and set(b["tail"]) >= {"NIFTY_50"}
    c = b["crashes"][0]
    assert c["dates"] and all(len(v) == len(c["dates"]) for v in c["returns"].values())


@pytest.mark.slow
@local
def test_surface_route():
    b = client.get("/internal/v1/surface/NIFTY_50").json()
    g = b["grid"]
    assert len(g["svi"]) == len(g["days"]) and len(g["svi"][0]) == len(g["k"])
    assert b["ref"]["days"] >= 7 or b["ref"]["expiry"] == b["expiries"][-1]["expiry"]
    assert {p["expiry"] for p in b["points"]} <= {e["expiry"] for e in b["expiries"]}


@pytest.mark.slow
@local
def test_events_route():
    b = client.get("/internal/v1/events/NIFTY_50").json()
    assert len(b["events"]) >= 10
    e = b["events"][0]
    assert set(e["stats"]) == {"h1", "h5", "h21", "h63"} and len(e["path"]["offsets"]) == len(e["path"]["mean"])
