"""
QuantOS | Overnight option-selling backtest (real Upstox option candles)

Entry  : SELL at 15:15 IST on session T (close of the 15:00 bar)
Exit   : BUY back on the next session at
           '0915' = open of the 09:15 bar
           '0930' = close of the 09:15 bar   (more realistic fill)
           '1515' = 15:15 on T+1             (hold the day; used by Plan C)
Expiry : nearest expiry strictly after T (never hold into an expiry close)

Strategies (pre-registered, 1 lot per leg):
    straddle_atm     sell ATM call + ATM put              (pure decay, no view)
    strangle_1pct    sell 1% OTM call + 1% OTM put
    put_otm          sell 1% OTM put every night          (rides the up-drift)
    put_atm          sell ATM put every night
    dir_otm          model view: bullish -> sell OTM put, bearish -> sell OTM call
    dir_atm          model view: bullish -> sell ATM put, bearish -> sell ATM call

P&L is in index points per unit (premium sold - premium bought - costs),
and in rupees per lot using each contract's own lot size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.core import settings  # noqa: E402

PROJECT_ROOT = settings.ROOT
OPT_DIR = settings.DATA_DIR / "derivatives" / "options"

# Costs per unit, per leg, round trip.
SLIPPAGE_PTS = {"NIFTY_50": 0.5, "NIFTY_BANK": 1.5}   # per side
BROKERAGE_PER_ORDER = 20.0                              # Rs, flat
STT_SELL = 0.001                                        # 0.1% of sell premium
EXCH_TXN = 0.000355                                     # per side, of premium
GST = 0.18                                              # on brokerage + txn
STAMP_BUY = 0.00003

LEGS = {
    "straddle_atm": ["atm_ce", "atm_pe"],
    "strangle_1pct": ["otm_ce", "otm_pe"],
    "put_otm": ["otm_pe"],
    "put_atm": ["atm_pe"],
}


def load_option_bars(symbol: str) -> pd.DataFrame:
    d = OPT_DIR / symbol.lower()
    frames = [pd.read_parquet(p) for p in sorted(d.glob("20*.parquet"))]
    bars = pd.concat(frames, ignore_index=True)
    bars["date"] = bars["timestamp"].dt.tz_localize(None).dt.normalize()
    bars["hm"] = bars["timestamp"].dt.strftime("%H:%M")
    return bars


def leg_prices(symbol: str) -> pd.DataFrame:
    """One row per (session, leg) with entry and exit prices."""
    plan = pd.read_parquet(OPT_DIR / symbol.lower() / "plan.parquet")
    bars = load_option_bars(symbol)
    key = ["strike", "type", "expiry"]
    entry = bars[bars["hm"] == "15:00"][key + ["date", "close"]].rename(columns={"close": "entry"})
    first = bars[bars["hm"] == "09:15"][key + ["date", "open", "close"]].rename(
        columns={"date": "next_session", "open": "x0915", "close": "x0930"})
    late = bars[bars["hm"] == "15:00"][key + ["date", "close"]].rename(
        columns={"date": "next_session", "close": "x1515"})
    p = plan.merge(entry, on=key + ["date"], how="left")
    p = p.merge(first, on=key + ["next_session"], how="left").merge(late, on=key + ["next_session"], how="left")
    lots = bars.groupby(key)["lot_size"].first().rename("lot_size").reset_index()
    p = p.merge(lots, on=key, how="left")
    p["dte"] = (p["expiry"] - p["date"]).dt.days
    return p


def leg_pnl(p: pd.DataFrame, symbol: str, exit_col: str) -> pd.Series:
    """Short-option P&L per unit after costs (points)."""
    sell, buy = p["entry"], p[exit_col]
    lot = p["lot_size"].fillna(75)
    brokerage = 2 * BROKERAGE_PER_ORDER / lot
    txn = EXCH_TXN * (sell + buy)
    costs = (
        2 * SLIPPAGE_PTS[symbol]
        + STT_SELL * sell
        + txn
        + GST * (brokerage + txn)
        + brokerage
        + STAMP_BUY * buy
    )
    return sell - buy - costs


def strategy_book(symbol: str, view: pd.Series | None, exit_col: str = "x0930") -> pd.DataFrame:
    """
    view: per-session bullish(+1)/bearish(-1) signal (index = session date).
    Returns per-session P&L (points/unit and Rs/lot) for each strategy.
    """
    p = leg_prices(symbol)
    p = p.dropna(subset=["entry", exit_col])
    p["pnl_pts"] = leg_pnl(p, symbol, exit_col)
    p["pnl_rs"] = p["pnl_pts"] * p["lot_size"]
    p["premium"] = p["entry"]
    wide = p.pivot_table(index="date", columns="leg", values=["pnl_pts", "pnl_rs", "premium"])
    meta = p.groupby("date").agg(dte=("dte", "first"), spot=("spot", "first"), expiry=("expiry", "first"))

    out = pd.DataFrame(index=wide.index)
    for name, legs in LEGS.items():
        if all(("pnl_pts", leg) in wide.columns for leg in legs):
            out[f"{name}_pts"] = sum(wide[("pnl_pts", leg)] for leg in legs)
            out[f"{name}_rs"] = sum(wide[("pnl_rs", leg)] for leg in legs)
            out[f"{name}_prem"] = sum(wide[("premium", leg)] for leg in legs)
    if view is not None:
        v = view.reindex(out.index)
        for name, (bull, bear) in {"dir_otm": ("otm_pe", "otm_ce"), "dir_atm": ("atm_pe", "atm_ce")}.items():
            for unit in ("pts", "rs"):
                out[f"{name}_{unit}"] = np.where(v > 0, wide[(f"pnl_{unit}", bull)], wide[(f"pnl_{unit}", bear)])
            out[f"{name}_prem"] = np.where(v > 0, wide[("premium", bull)], wide[("premium", bear)])
            out.loc[v.isna(), [f"{name}_pts", f"{name}_rs", f"{name}_prem"]] = np.nan
    return out.join(meta)


def summarise(book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in [c[:-4] for c in book.columns if c.endswith("_pts")]:
        pts = book[f"{name}_pts"].dropna()
        rs = book[f"{name}_rs"].dropna()
        prem = book[f"{name}_prem"].reindex(pts.index)
        if pts.empty:
            continue
        cum = rs.cumsum()
        rows.append({
            "strategy": name,
            "nights": len(pts),
            "win_rate": (pts > 0).mean(),
            "avg_pts": pts.mean(),
            "avg_rs_per_lot": rs.mean(),
            "kept_of_premium": pts.sum() / prem.sum(),
            "worst_night_rs": rs.min(),
            "best_night_rs": rs.max(),
            "total_rs_per_lot": rs.sum(),
            "max_drawdown_rs": (cum - cum.cummax()).min(),
            "t_stat": pts.mean() / pts.std() * np.sqrt(len(pts)),
        })
    return pd.DataFrame(rows).set_index("strategy")
