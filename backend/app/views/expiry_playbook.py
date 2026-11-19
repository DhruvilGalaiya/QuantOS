"""
QuantOS | Expiry Playbook page (experimental, paper-tracked)

Last 2 sessions before an expiry: at ~15:15 sell the ~1% OTM option on the
side the 15:15 model favours (bullish -> put, bearish -> call), buy it back at
15:15 on the next session. Logic lives in src/quantos_expiry_playbook.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)
from quantos_platform import jobs  # noqa: E402
from quantos_platform.strategies import expiry_playbook as pb  # noqa: E402

UP, DOWN = theme.BULL, theme.BEAR
NAMES = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "BANK NIFTY"}
EXITS = {"x1515": "Hold to 15:15 next day", "x0930": "Exit 09:30 next day"}
STRATS = {
    "dir_otm": "Model side · 1% OTM (playbook)",
    "put_otm": "Always sell 1% OTM put",
    "strangle_1pct": "Sell 1% OTM strangle",
    "straddle_atm": "Sell ATM straddle",
}


# ============================================================
# DATA (cached)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_backtest(symbol: str) -> pd.DataFrame:
    return pb.load_backtest(symbol)


def rs(x: float, signed: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{'+' if signed and x > 0 else ''}{'−' if x < 0 else ''}₹{abs(x):,.0f}"


def nice_date(s: str | None) -> str:
    return pd.Timestamp(s).strftime("%a %d %b") if s else "—"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.subheader("Playbook")
    symbol = st.selectbox("Index", list(NAMES), format_func=NAMES.get)
    ui.refresh_button("playbook", "Refresh signal")
    st.caption(
        "The signal is **final between 15:10 and 15:29 IST**. Before that "
        "it's a preview based on the session so far."
    )

    st.caption(ui.freshness_caption("playbook", "Signal computed",
                                    fallback=(jobs.load_tonight(symbol) or {}).get("as_of")))

with st.spinner("Checking the 15:15 signal…"):
    ui.refresh_on_launch("playbook")
ui.job_health("playbook")
# the scheduled `playbook` job writes tonight's signal and scores the paper log
sig = jobs.load_tonight(symbol)
sig_error = None if sig else "no signal computed yet (runs with the `playbook` job)"
log = pb.load_signals(symbol)
backtest = get_backtest(symbol)


# ============================================================
# TONIGHT
# ============================================================

theme.hero(
    "Expiry playbook",
    "On the last two sessions before an expiry, sell the roughly 1% out-of-the-money option on the side "
    "the 15:15 model favours and buy it back at 15:15 the next session.",
    [f"<b>{NAMES[symbol]}</b>", f"Paper-tracked since <b>{pb.TRACK_START:%d %b %Y}</b>", "Experimental, not advice"],
    eyebrow="Derivatives strategy",
)

if sig_error:
    theme.status(f"Could not compute tonight's signal ({sig_error}). Showing stored results.", "side")

if sig:
    phase, active = sig.get("phase"), sig.get("active")
    contract = sig.get("trading_symbol") or (
        f"{sig.get('strike', 0):.0f} {sig.get('side', '')}" if sig.get("side") else "—")
    lot = sig.get("lot_size")
    prem = sig.get("premium")
    c = theme.esc(contract)

    if phase == "closed":
        theme.signal("Market closed today",
                     f"Next signal night: <b>{nice_date(sig.get('next_window') or sig.get('session'))}</b>.", "muted")
    elif not active:
        theme.signal("No trade tonight",
                     f"{sig['tdte']} sessions to the {nice_date(sig['expiry'])} expiry. "
                     f"Next signal night: <b>{nice_date(sig.get('next_window'))}</b>.", "muted")
    elif phase in ("preview", "pre-open"):
        theme.signal("Signal night, preview",
                     f"Current lean: sell <b>{c}</b>. The final call is made between 15:10 and 15:29.", "side")
    elif phase == "final":
        theme.signal(f"Sell 1 lot {contract}",
                     (f"At about <b>₹{prem:,.2f}</b>. " if prem else "")
                     + "Buy it back at 15:15 tomorrow.", "bull")
    else:  # after the window
        theme.signal("Today was a signal night",
                     f"Sold <b>{c}</b> at 15:15. It is logged below once scored.", "accent")

    if "view" in sig:
        bullish = sig["view"] > 0
        theme.stats([
            theme.stat("Sessions to expiry", str(sig.get("tdte", "—")), sub=f"Expiry {nice_date(sig.get('expiry'))}",
                       help=f"Signal nights are the last {pb.MAX_TDTE} sessions before expiry."),
            theme.stat("Model view", "Bullish" if bullish else "Bearish", tone="bull" if bullish else "bear",
                       sub=f"P(gap up) {sig['p_up']:.0%} vs base {sig['drift']:.0%}",
                       help="Probability the next open is above the 15:15 price, vs the usual up-rate. "
                            "Bullish → sell the put, bearish → sell the call."),
            theme.stat("Option to sell" if active else "Would sell (not tonight)",
                       f"{sig['strike']:,.0f} {sig['side']}", sub=f"Spot {sig['spot']:,.0f}, about 1% OTM"),
            theme.stat("Premium per lot", rs(prem * lot, signed=False) if prem and lot else "—",
                       sub=f"₹{prem:,.2f} × {lot}" if prem and lot else None,
                       help="Last traded price × lot size: the most you can keep if it expires worthless."),
        ])
    theme.note(f"As of {sig.get('as_of', '—')} IST, session {sig.get('session', '—')}.")

st.space("small")
with st.expander("The rule and its risks"):
    st.markdown(
        f"""
- **When:** only on the last {pb.MAX_TDTE} trading sessions before an expiry (NIFTY weekly, BANK NIFTY monthly).
- **What:** at ~15:15, sell 1 lot of the ~1% out-of-the-money option on the side the model favours.
  Bullish → **put**, bearish → **call**.
- **Exit:** buy it back at **15:15 on the next session**, or at 09:30 for a quicker exit with roughly half the edge.
- **Why it works:** in the last 1–2 days an option loses value quickly. Overnight, option prices usually
  get *more expensive* (implied vol 13.5% → 15% by 9:30), which kills the trade on other nights. Close to expiry, the decay wins.
- **Risk:** a big move against you. Worst backtest night was about −₹19.5k per NIFTY lot.
  NSE charges extra margin on expiry day for short options.
- **Caveats:** ~2 years of data. The "last 2 sessions" filter was found after looking at results.
  Treat it as unproven until the paper log below agrees.
"""
    )


# ============================================================
# PAPER LOG
# ============================================================

theme.section("Paper-trade log", "Every signal night since tracking began, scored against real option prices.")

scored = log.dropna(subset=["pnl_1515_rs"]) if len(log) else log
if len(log) == 0:
    theme.note(
        f"No signal nights since {pb.TRACK_START:%d %b %Y} yet. Each signal night is logged from "
        "the 15:15 snapshot and scored against real option prices after the exit."
    )
else:
    total = scored["pnl_1515_rs"].sum() if len(scored) else np.nan
    theme.stats([
        theme.stat("Signal nights", f"{len(log)}"),
        theme.stat("Scored", f"{len(scored)}"),
        theme.stat("Win rate, hold", f"{(scored['pnl_1515_rs'] > 0).mean():.0%}" if len(scored) else "—"),
        theme.stat("Total per lot, hold", rs(total) if len(scored) else "—",
                   tone=("bull" if total > 0 else "bear") if len(scored) and total != 0 else None),
    ], enter=False)
    show = log.sort_values("date", ascending=False).assign(
        Night=lambda d: d["date"].dt.strftime("%a %d %b %Y"),
        Exit=lambda d: d["next_session"].dt.strftime("%a %d %b"),
        View=lambda d: np.where(d["view"] > 0, "Bullish", "Bearish"),
    )
    st.dataframe(
        show[["Night", "tdte", "View", "trading_symbol", "entry", "exit_0930", "exit_1515",
              "pnl_0930_rs", "pnl_1515_rs", "Exit"]],
        hide_index=True, width="stretch",
        column_config={
            "tdte": st.column_config.NumberColumn("Sessions to expiry", format="%d"),
            "trading_symbol": "Sold",
            "entry": st.column_config.NumberColumn("Sold at 15:15", format="%.2f"),
            "exit_0930": st.column_config.NumberColumn("09:30 price", format="%.2f"),
            "exit_1515": st.column_config.NumberColumn("15:15 price", format="%.2f"),
            "pnl_0930_rs": st.column_config.NumberColumn("₹/lot (exit 09:30)", format="%+.0f"),
            "pnl_1515_rs": st.column_config.NumberColumn("₹/lot (hold)", format="%+.0f"),
            "Exit": "Exit session",
        },
    )
    theme.note("P&L is after brokerage, STT, exchange fees, GST and slippage, for one lot.")


# ============================================================
# BACKTEST
# ============================================================

theme.section("Backtest", "Signal nights only, with the playbook compared against simpler short-option rules.")

if backtest.empty:
    theme.note("Backtest not built yet. Run `python -m quantos_platform.strategies.expiry_playbook --backtest`.")
else:
    exit_col = st.segmented_control(
        "Exit", list(EXITS), format_func=EXITS.get, default="x1515", label_visibility="collapsed",
    ) or "x1515"
    near = backtest[backtest["tdte"] <= pb.MAX_TDTE]

    rows = []
    for key, label in STRATS.items():
        x = near[f"{key}_{exit_col}"].dropna()
        last12 = x[x.index >= (x.index.max().to_period("M") - 11).to_timestamp()]
        rows.append({
            "Strategy": label,
            "Nights": len(x),
            "Avg ₹/lot": x.mean(),
            "Win rate": (x > 0).mean(),
            "Worst night": x.min(),
            "Last 12 months": last12.sum(),
            "Luck check (t)": x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan,
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Avg ₹/lot": st.column_config.NumberColumn(format="%+.0f"),
            "Win rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Worst night": st.column_config.NumberColumn(format="%+.0f"),
            "Last 12 months": st.column_config.NumberColumn(format="%+.0f"),
            "Luck check (t)": st.column_config.NumberColumn(format="%.1f",
                help="Above ~2 means the average is unlikely to be pure luck."),
        },
    )

    pl = near[f"dir_otm_{exit_col}"].dropna()
    start = (pl.index.max().to_period("M") - 11).to_timestamp()  # 12 calendar months
    recent = pl[pl.index >= start]
    monthly = recent.groupby(recent.index.to_period("M")).agg(["sum", "count", lambda s: (s > 0).mean()])
    monthly.columns = ["pnl", "nights", "win"]
    labels = [p.strftime("%b %y") for p in monthly.index]

    theme.section("Playbook P&L by month", f"Last 12 months, {EXITS[exit_col].lower()}, ₹ per lot.")
    fig = go.Figure(go.Bar(
        x=labels, y=monthly["pnl"],
        marker=dict(color=[UP if v >= 0 else DOWN for v in monthly["pnl"]], line=dict(width=0),
                    cornerradius=5, opacity=0.92),
        customdata=np.c_[monthly["nights"], monthly["win"] * 100],
        hovertemplate="%{x}<br>₹%{y:+,.0f} per lot<br>%{customdata[0]:.0f} nights, "
                      "%{customdata[1]:.0f}% won<extra></extra>",
        text=[rs(v) for v in monthly["pnl"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.MUTED, size=11),
    ))
    fig.add_hline(y=0, line=dict(color=theme.FAINT, width=1))
    fig.update_layout(
        height=340, margin=dict(l=8, r=8, t=24, b=8), showlegend=False, bargap=0.38,
        yaxis=dict(tickprefix="₹", tickformat=",.0f", zeroline=False),
        xaxis=dict(showgrid=False),
    )
    with theme.card("pnl"):
        theme.chart(fig)
    theme.note(
        f"Total {rs(recent.sum())} per lot over {len(recent)} nights "
        f"({(recent > 0).mean():.0%} won). The backtest uses real Upstox option candles "
        f"from {near.index.min():%b %Y} to {near.index.max():%d %b %Y}, a walk-forward model "
        "(trained only on data before each night) and full trading costs."
    )
