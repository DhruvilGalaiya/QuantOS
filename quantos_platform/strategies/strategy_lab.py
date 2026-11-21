"""
QuantOS | Expiry Strategy Lab: backtest simple option-selling setups in the last week before expiry.

The user picks what to sell (put, call or both), how far from the index (at the money to 2% away), an optional
protective option further out, when to sell (sessions before expiry + time) and when to buy back (or let it
expire), plus an optional stop-loss. Every past expiry cycle is replayed on real 15-minute option prices
(data/derivatives/expiry_week, see quantos_platform.data.expiry_week).

Conventions:
    entry / exit price  = the 15-minute bar OPEN at the chosen time (what you could trade at that moment)
    strike              = nearest listed strike to index × (1 ± distance), using the index price at entry
    stop-loss           = checked on 15-minute closes of the whole position; exit at that close
    expiry settlement   = intrinsic value at the index's final close on expiry day
    rupees              = points × today's lot size (so old and new trades compare like for like)
    costs               = ₹20 brokerage per order + 0.1% STT on premium sold + ~0.05% exchange/other charges
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from quantos_platform.core import settings

CHAIN_DIR = settings.DATA_DIR / "derivatives" / "expiry_week"
STEP = {"NIFTY_50": 50, "NIFTY_BANK": 100}
LOT = {"NIFTY_50": 65, "NIFTY_BANK": 30}
BROKERAGE_PER_ORDER = 20.0
STT_SELL = 0.001
OTHER_CHARGES = 0.0005
SETTLE = "settle"          # exit_time value meaning "let it expire"
TIMES = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in (0, 15, 30, 45)
         if "09:15" <= f"{h:02d}:{m:02d}" <= "15:15"]


@dataclass(frozen=True)
class Spec:
    kind: str = "put"            # put | call | both
    distance: float = 0.01       # 0 = at the money
    wing: float | None = None    # buy protection this much further out (e.g. 0.01), None = naked
    entry_dte: int = 1           # sessions before expiry at entry (0 = expiry day)
    entry_time: str = "15:15"
    exit_dte: int = 0            # sessions before expiry at exit
    exit_time: str = "15:15"     # or SETTLE
    stop: float | None = None    # exit if the position costs this multiple of the credit (e.g. 2.0)


# ── data ───────────────────────────────────────────────────────────────────────
def load_chains(symbol: str) -> dict:
    """{expiry: (opens, closes, days)} with opens/closes indexed by naive IST timestamp, columns (type, strike)."""
    out = {}
    for path in sorted((CHAIN_DIR / symbol.lower()).glob("*.parquet")):
        df = pd.read_parquet(path, columns=["timestamp", "open", "close", "strike", "type"])
        df["ts"] = df["timestamp"].dt.tz_localize(None)
        o = df.pivot_table(index="ts", columns=["type", "strike"], values="open", aggfunc="first")
        c = df.pivot_table(index="ts", columns=["type", "strike"], values="close", aggfunc="first")
        days = sorted(o.index.normalize().unique())
        out[pd.Timestamp(path.stem)] = (o, c, days)
    return out


def load_spot(symbol: str) -> tuple[pd.Series, pd.Series]:
    from quantos_platform.data.intraday import load_intraday

    b = load_intraday(symbol)
    ts = b["timestamp"].dt.tz_localize(None)
    opens = pd.Series(b["open"].to_numpy(), index=ts)
    day_close = pd.Series(b["close"].to_numpy(), index=ts.dt.normalize()).groupby(level=0).last()
    return opens, day_close


# ── engine ─────────────────────────────────────────────────────────────────────
def _legs(spec: Spec, spot: float, step: int) -> list[tuple[str, float, int]]:
    """(type, strike, sign): sign -1 = sold, +1 = bought."""
    def k(x):
        return float(round(x / step) * step)
    legs = []
    if spec.kind in ("put", "both"):
        legs.append(("PE", k(spot * (1 - spec.distance)), -1))
        if spec.wing:
            legs.append(("PE", k(spot * (1 - spec.distance - spec.wing)), +1))
    if spec.kind in ("call", "both"):
        legs.append(("CE", k(spot * (1 + spec.distance)), -1))
        if spec.wing:
            legs.append(("CE", k(spot * (1 + spec.distance + spec.wing)), +1))
    return legs


def describe(spec: Spec) -> str:
    where = "at the money" if spec.distance == 0 else f"{spec.distance:.1%} out of the money"
    what = {"put": "a put", "call": "a call", "both": "a put and a call"}[spec.kind]
    return f"Sell {what} {where}" + (f", buy protection {spec.wing:.1%} further out" if spec.wing else "")


def run(symbol: str, spec: Spec, chains: dict, spot_open: pd.Series, day_close: pd.Series) -> pd.DataFrame:
    step, lot = STEP[symbol], LOT[symbol]
    rows = []
    for exp, (o, c, days) in chains.items():
        if not days or days[-1] != exp or len(days) <= spec.entry_dte or spec.exit_dte > spec.entry_dte:
            continue
        ed, xd = days[-1 - spec.entry_dte], days[-1 - spec.exit_dte]
        t0 = ed + pd.Timedelta(spec.entry_time + ":00")
        settle = spec.exit_time == SETTLE
        t1 = xd + pd.Timedelta(("15:15" if settle else spec.exit_time) + ":00")
        if t1 <= t0 or t0 not in o.index or t0 not in spot_open.index or (not settle and t1 not in o.index):
            continue
        s0 = float(spot_open[t0])
        legs = _legs(spec, s0, step)
        cols = [(t, k) for t, k, _ in legs]
        if any(col not in o.columns for col in cols):
            continue
        sign = np.array([g for _, _, g in legs], dtype=float)
        p0 = o.loc[t0, cols].to_numpy(dtype=float)
        if not np.isfinite(p0).all():
            continue
        credit = float(-(sign * p0).sum())                     # premium collected (net)
        if credit <= 0:
            continue
        path = c.loc[(c.index >= t0) & (c.index < t1), cols]
        stopped, exit_ts = False, t1
        if spec.stop and len(path):
            cost_path = (-(path.to_numpy(dtype=float) * sign)).sum(axis=1)
            hit = np.where(cost_path >= spec.stop * credit)[0]
            if len(hit):
                stopped, exit_ts = True, path.index[hit[0]]
                p1 = path.iloc[hit[0]].to_numpy(dtype=float)
        if not stopped:
            if settle:
                if exp not in day_close.index:
                    continue
                s1 = float(day_close[exp])
                p1 = np.array([max(s1 - k, 0.0) if t == "CE" else max(k - s1, 0.0) for t, k, _ in legs])
            else:
                p1 = o.loc[t1, cols].to_numpy(dtype=float)
        if not np.isfinite(p1).all():
            continue
        pnl_pts = credit + float((sign * p1).sum())
        sold_prem = float(p0[sign < 0].sum())
        n_orders = len(legs) * (1 if settle and not stopped else 2)
        costs = BROKERAGE_PER_ORDER * n_orders + (STT_SELL + OTHER_CHARGES) * sold_prem * lot
        rows.append({
            "expiry": exp, "entry_ts": t0, "exit_ts": exit_ts, "spot_entry": s0,
            "spot_exit": float(day_close[exp]) if settle and not stopped else float(spot_open.get(exit_ts, np.nan)),
            "legs": " + ".join(f"{'Sell' if g < 0 else 'Buy'} {k:,.0f} {t}" for t, k, g in legs),
            "entry_prices": " / ".join(f"{p:,.1f}" for p in p0), "exit_prices": " / ".join(f"{p:,.1f}" for p in p1),
            "credit_pts": credit, "exit_cost_pts": credit - pnl_pts, "pnl_pts": pnl_pts,
            "pnl_rs": pnl_pts * lot - costs, "credit_rs": credit * lot, "costs_rs": costs,
            "kept_pct": pnl_pts / credit, "stopped": stopped, "settled": settle and not stopped,
        })
    return pd.DataFrame(rows)


def summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    p = trades["pnl_rs"]
    eq = p.cumsum()
    dd = (eq - eq.cummax()).min()
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    return {"n": len(p), "win_rate": float((p > 0).mean()), "avg": float(p.mean()), "median": float(p.median()),
            "total": float(p.sum()), "best": float(p.max()), "worst": float(p.min()),
            "max_drawdown": float(dd), "profit_factor": float(wins / losses) if losses > 0 else np.inf,
            "avg_credit": float(trades["credit_rs"].mean()), "stops": int(trades["stopped"].sum()),
            "start": trades["entry_ts"].min(), "end": trades["entry_ts"].max()}


def compare_distances(symbol: str, spec: Spec, chains, spot_open, day_close,
                      distances=(0.0, 0.005, 0.01, 0.015, 0.02)) -> pd.DataFrame:
    rows = []
    for d in distances:
        s = summary(run(symbol, replace(spec, distance=d), chains, spot_open, day_close))
        rows.append({"distance": d, **s})
    return pd.DataFrame(rows)
