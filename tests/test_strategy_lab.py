"""Strategy Lab engine on a tiny synthetic expiry week (no local data needed)."""

import numpy as np
import pandas as pd
import pytest

from quantos_platform.strategies import strategy_lab as L


def _week():
    days = pd.bdate_range("2025-01-06", periods=5)          # expiry = last day
    ts = [d + pd.Timedelta(hours=h, minutes=m) for d in days for h, m in ((9, 15), (15, 15))]
    cols = pd.MultiIndex.from_tuples([("PE", 9900.0), ("PE", 9850.0), ("CE", 10100.0), ("CE", 10000.0), ("PE", 10000.0)])
    o = pd.DataFrame(10.0, index=ts, columns=cols)
    o.loc[ts[-3], ("PE", 9900.0)] = 8.0                        # entry 1 session before expiry, 15:15
    o.loc[ts[-1], ("PE", 9900.0)] = 3.0                        # exit expiry day 15:15
    c = o.copy()
    spot_open = pd.Series(10000.0, index=ts)
    day_close = pd.Series(10000.0, index=days)
    return {days[-1]: (o, c, list(days))}, spot_open, day_close


def test_short_put_pnl_and_costs():
    chains, so, dc = _week()
    spec = L.Spec(kind="put", distance=0.01, entry_dte=1, entry_time="15:15", exit_dte=0, exit_time="15:15")
    t = L.run("NIFTY_50", spec, chains, so, dc)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["credit_pts"] == pytest.approx(8.0) and r["pnl_pts"] == pytest.approx(5.0)
    costs = 2 * L.BROKERAGE_PER_ORDER + (L.STT_SELL + L.OTHER_CHARGES) * 8.0 * L.LOT["NIFTY_50"]
    assert r["pnl_rs"] == pytest.approx(5.0 * L.LOT["NIFTY_50"] - costs)


def test_settlement_uses_intrinsic_value():
    chains, so, dc = _week()
    dc.iloc[-1] = 9898.0                                      # index closes 2 pts below the 9900 put
    spec = L.Spec(kind="put", distance=0.01, entry_dte=1, entry_time="15:15", exit_dte=0, exit_time=L.SETTLE)
    r = L.run("NIFTY_50", spec, chains, so, dc).iloc[0]
    assert r["settled"] and r["exit_cost_pts"] == pytest.approx(2.0)


def test_stop_loss_triggers_on_close():
    chains, so, dc = _week()
    (o, c, days), = chains.values()
    c.loc[c.index[-2], ("PE", 9900.0)] = 20.0                   # position cost hits 2.5x credit
    spec = L.Spec(kind="put", distance=0.01, entry_dte=1, entry_time="15:15", exit_dte=0, exit_time="15:15",
                  stop=2.0)
    r = L.run("NIFTY_50", spec, chains, so, dc).iloc[0]
    assert r["stopped"] and r["pnl_pts"] == pytest.approx(8.0 - 20.0)


def test_exit_before_entry_gives_no_trades():
    chains, so, dc = _week()
    spec = L.Spec(entry_dte=0, entry_time="15:15", exit_dte=0, exit_time="09:15")
    assert L.run("NIFTY_50", spec, chains, so, dc).empty
    assert L.summary(pd.DataFrame())["n"] == 0
    assert np.isfinite(L.summary(L.run("NIFTY_50", L.Spec(distance=0.01), chains, so, dc))["avg"])
