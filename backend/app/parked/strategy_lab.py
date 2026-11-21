"""
QuantOS | Expiry Strategy Lab (PARKED: not in the navigation; to be redesigned in a later phase)

To bring it back: move this file to backend/app/views/ and add an st.Page entry in regime_dashboard_v10.py.

Build a simple option-selling trade for expiry week and replay it on every past expiry cycle with real
15-minute option prices: what to sell (put / call / both), how far from the index, optional protection,
when to sell and when to buy back (or let it expire), and an optional stop-loss. Engine:
quantos_platform/strategies/strategy_lab.py; data: python -m quantos_platform.data.expiry_week.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st
import theme  # noqa: E402  (backend/app/theme.py)

from quantos_platform.strategies import strategy_lab as L  # noqa: E402

NAMES = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "BANK NIFTY"}
KINDS = {"put": "Sell a put (you profit if the index does not fall much)",
         "call": "Sell a call (you profit if the index does not rise much)",
         "both": "Sell both, put and call (you profit if the index stays in a range)"}
DISTANCES = {0.0: "At the money", 0.005: "0.5% away", 0.01: "1% away", 0.015: "1.5% away", 0.02: "2% away"}
WINGS = {None: "No protection", 0.005: "Buy one 0.5% further out", 0.01: "Buy one 1% further out"}
STOPS = {None: "No stop-loss", 1.5: "Exit if it costs 1.5× what I collected", 2.0: "Exit at 2×", 3.0: "Exit at 3×"}


def dte_label(d: int) -> str:
    return "Expiry day" if d == 0 else ("1 session before expiry" if d == 1 else f"{d} sessions before expiry")


def rs(x, signed: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    sign = ("+" if x > 0 else "−" if x < 0 else "") if signed else ("−" if x < 0 else "")
    return f"{sign}₹{abs(x):,.0f}"


def _mtime(symbol: str) -> float:
    d = L.CHAIN_DIR / symbol.lower()
    return max((p.stat().st_mtime for p in d.glob("*.parquet")), default=0.0)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def data(symbol: str, _mtime: float):
    return L.load_chains(symbol), *L.load_spot(symbol)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def backtest(symbol: str, spec: L.Spec, _mtime: float) -> pd.DataFrame:
    return L.run(symbol, spec, *data(symbol, _mtime))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def summary_of(symbol: str, spec: L.Spec, _mtime: float) -> dict:
    return L.summary(backtest(symbol, spec, _mtime))


# ============================================================
# HEADER + BUILDER
# ============================================================

theme.hero(
    "Expiry Strategy Lab",
    "Design an option-selling trade for expiry week and see how it would have done on every past expiry, "
    "using real option prices. Mix and match what you sell, how far from the index, when you enter and when "
    "you exit.",
    ["Real 15-minute option prices", "Costs included", "History, not a recommendation"],
)

with theme.card("builder"):
    st.markdown("##### Build your trade")
    c1, c2, c3 = st.columns([1, 1.6, 1])
    symbol = c1.selectbox("Index", list(NAMES), format_func=NAMES.get, key="sl_symbol")
    kind = c2.selectbox("What to sell", list(KINDS), format_func=KINDS.get, key="sl_kind")
    distance = c3.selectbox("Strike: how far from the index", list(DISTANCES), index=2,
                            format_func=DISTANCES.get, key="sl_dist",
                            help="At the money = strike closest to the index. Further away = cheaper option, "
                                 "less premium, but the index has to move more before you lose.")
    c4, c5, c6, c7 = st.columns(4)
    entry_dte = c4.selectbox("Sell on", [4, 3, 2, 1, 0], index=3, format_func=dte_label, key="sl_edte")
    entry_time = c5.selectbox("Sell at (IST)", L.TIMES, index=L.TIMES.index("15:15"), key="sl_etime")
    exit_choices = list(range(entry_dte, -1, -1))
    exit_dte = c6.selectbox("Buy back on", exit_choices, index=len(exit_choices) - 1 if entry_dte else 0,
                            format_func=dte_label, key=f"sl_xdte_{entry_dte}")
    exit_opts = ([L.SETTLE] if exit_dte == 0 else []) + L.TIMES
    exit_time = c7.selectbox("Buy back at (IST)", exit_opts, index=exit_opts.index("15:15"),
                             format_func=lambda t: "Let it expire" if t == L.SETTLE else t, key=f"sl_xtime_{exit_dte}")
    c8, c9 = st.columns(2)
    wing = c8.selectbox("Protection (limits the worst loss)", list(WINGS), format_func=WINGS.get, key="sl_wing",
                        help="Also buy a cheaper option further away. It costs part of the premium but caps how "
                             "much a big move can hurt.")
    stop = c9.selectbox("Stop-loss", list(STOPS), format_func=STOPS.get, key="sl_stop",
                        help="Buy back early if the position costs this multiple of the premium you collected. "
                             "Checked every 15 minutes.")

spec = L.Spec(kind=kind, distance=distance, wing=wing, entry_dte=entry_dte, entry_time=entry_time,
              exit_dte=exit_dte, exit_time=exit_time, stop=stop)
mt = _mtime(symbol)
if not mt:
    theme.status("No expiry-week option data yet. Run: python -m quantos_platform.data.expiry_week", "side")
    st.stop()
with st.spinner("Replaying every past expiry…"):
    trades = backtest(symbol, spec, mt)
S = L.summary(trades)
if not S["n"]:
    theme.status("No trades for this setup: the buy-back time must come after the sell time.", "side")
    st.stop()


# ============================================================
# EXAMPLE + RESULTS
# ============================================================

lot = L.LOT[symbol]
ex = trades.iloc[-1]
exit_word = ("let it expire; it settled" if ex["settled"] else
             "the stop-loss bought it back" if ex["stopped"] else "bought it back")
theme.section("An example from the most recent expiry")
theme.insight_html(
    f"On <b>{ex['entry_ts']:%a %d %b %Y} at {ex['entry_ts']:%H:%M}</b>, {NAMES[symbol]} was at "
    f"<b>{ex['spot_entry']:,.0f}</b>. The trade: <b>{theme.esc(ex['legs'])}</b> at "
    f"{theme.esc(ex['entry_prices'])}, collecting <b>{rs(ex['credit_rs'], False)}</b> for 1 lot of {lot}. "
    f"On {ex['exit_ts']:%a %d %b at %H:%M} you {exit_word} at {theme.esc(ex['exit_prices'])}. "
    f"Result after costs: <b class=\"t-{'bull' if ex['pnl_rs'] > 0 else 'bear'}\">{rs(ex['pnl_rs'])}</b>.",
    "Example")

theme.section("How this trade did on every past expiry",
              f"{L.describe(spec)}; sell {dte_label(entry_dte).lower()} at {entry_time}, "
              + ("let it expire." if exit_time == L.SETTLE else
                 f"buy back {dte_label(exit_dte).lower()} at {exit_time}.")
              + f" {S['n']} expiries from {S['start']:%b %Y} to {S['end']:%b %Y}, 1 lot ({lot} units).")
theme.stats([
    theme.stat("Trades won", f"{S['win_rate']:.0%}", sub=f"{S['n']} trades"),
    theme.stat("Average per trade", rs(S["avg"]), tone="bull" if S["avg"] > 0 else "bear",
               sub=f"typical (median) {rs(S['median'])}"),
    theme.stat("Total profit", rs(S["total"]), tone="bull" if S["total"] > 0 else "bear"),
    theme.stat("Worst trade", rs(S["worst"]), tone="bear", sub=f"best {rs(S['best'])}"),
    theme.stat("Deepest losing stretch", rs(S["max_drawdown"]), tone="bear",
               help="The biggest fall from a running high in total profit: how much you would have been down "
                    "before recovering."),
    theme.stat("Premium collected", rs(S["avg_credit"], False), sub="average per trade"),
])

eq = trades.set_index("entry_ts")["pnl_rs"]
with theme.card("equity"):
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                        subplot_titles=("Running total of profit, ₹ (1 lot)", "Each trade, ₹"))
    total = pd.concat([pd.Series([0.0], index=[eq.index[0] - pd.Timedelta(days=1)]), eq.cumsum()])
    fig.add_scatter(x=total.index, y=total, mode="lines", fill="tozeroy", line=dict(color=theme.ACCENT, width=2.2),
                    fillcolor=theme.alpha(theme.ACCENT, 0.08), showlegend=False, row=1, col=1,
                    hovertemplate="%{x|%d %b %Y}<br>Running total <b>₹%{y:,.0f}</b><extra></extra>")
    fig.add_bar(x=eq.index, y=eq, marker_color=np.where(eq >= 0, theme.BULL, theme.BEAR), showlegend=False,
                customdata=np.c_[trades["legs"], trades["entry_prices"], trades["exit_prices"]], row=2, col=1,
                hovertemplate="<b>%{x|%d %b %Y}</b><br>%{customdata[0]}<br>Sold at %{customdata[1]}, bought back "
                              "at %{customdata[2]}<br>Result <b>₹%{y:,.0f}</b><extra></extra>")
    fig.add_hline(y=0, line=dict(color=theme.FAINT, width=1), row=1, col=1)
    fig.update_layout(height=440, hovermode="closest", margin=dict(t=30, l=8, r=8, b=8), bargap=0.25)
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    fig.update_annotations(font=dict(size=12.5, color=theme.MUTED), x=0, xanchor="left")
    theme.chart(fig, key="equity")

avg_win = trades.loc[trades["pnl_rs"] > 0, "pnl_rs"].mean()
avg_loss = trades.loc[trades["pnl_rs"] < 0, "pnl_rs"].mean()
verdict = ("made money overall" if S["total"] > 0 else "lost money overall")
theme.insight_html(
    f"This setup won <b>{S['win_rate']:.0%}</b> of the time and <b>{verdict}</b> ({rs(S['total'])} on 1 lot). "
    + (f"An average win was {rs(avg_win)} and an average loss {rs(avg_loss)}, so one loss erased about "
       f"{abs(avg_loss) / avg_win:.1f} wins. " if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_win > 0
       else "")
    + "Option selling usually wins often and loses rarely but big: judge a setup by its worst trade and its "
      "losing stretch, not just by how often it wins. Try adding protection or a stop-loss to see the trade-off.")


# ============================================================
# COMPARISONS
# ============================================================

theme.section("Same timing, different strikes", "Your entry and exit, with the strike moved closer or further.")
rows = []
for d, lab in DISTANCES.items():
    s = summary_of(symbol, replace(spec, distance=d), mt)
    if not s["n"]:
        continue
    rows.append([f'<span class="strong">{lab}</span>' + (" ← yours" if d == distance else ""), str(s["n"]),
                 f"{s['win_rate']:.0%}", rs(s["avg_credit"], False), theme.rupee_html(s["avg"]),
                 f'<span class="t-bear">{rs(s["worst"])}</span>', theme.rupee_html(s["total"])])
theme.table([("Strike", ""), ("Trades", "num"), ("Won", "num"), ("Premium collected", "num"),
             ("Average result", "num"), ("Worst trade", "num"), ("Total", "num")], rows,
            current=list(DISTANCES).index(distance) if len(rows) == len(DISTANCES) else None)

theme.section("Same trade, different entry days", "Your strike and times, entered earlier or later in "
              "expiry week (buying back as you chose, or on expiry day if that comes first).")
rows = []
for d in [4, 3, 2, 1, 0]:
    alt = replace(spec, entry_dte=d, exit_dte=min(exit_dte, d))
    s = summary_of(symbol, alt, mt)
    if not s["n"]:
        continue
    rows.append([f'<span class="strong">{dte_label(d)}</span>' + (" ← yours" if d == entry_dte else ""),
                 str(s["n"]), f"{s['win_rate']:.0%}", rs(s["avg_credit"], False),
                 theme.rupee_html(s["avg"]), f'<span class="t-bear">{rs(s["worst"])}</span>',
                 theme.rupee_html(s["total"])])
theme.table([("Sell on", ""), ("Trades", "num"), ("Won", "num"), ("Premium collected", "num"),
             ("Average result", "num"), ("Worst trade", "num"), ("Total", "num")], rows)


# ============================================================
# TRADE LOG (on demand)
# ============================================================

theme.section("Every trade")
show = st.segmented_control("Show trades", ["Last 10", "Last 25", "All"], default=None, key="sl_log",
                            label_visibility="collapsed")
if show:
    n = {"Last 10": 10, "Last 25": 25}.get(show, len(trades))
    t = trades.sort_values("entry_ts", ascending=False).head(n)
    theme.table([("Sold", ""), ("Index then", "num"), ("Position", ""), ("Sold at", "num"), ("Bought back", ""),
                 ("At", "num"), ("Result", "num")],
                [[f"{r.entry_ts:%d %b %Y %H:%M}", f"{r.spot_entry:,.0f}", theme.esc(r.legs), theme.esc(r.entry_prices),
                  ("Expired" if r.settled else ("Stop-loss " if r.stopped else "") + f"{r.exit_ts:%d %b %H:%M}"),
                  theme.esc(r.exit_prices), theme.rupee_html(r.pnl_rs)] for r in t.itertuples()])
else:
    theme.note("Pick how many trades to list.")

theme.method(
    "- **Prices**: real 15-minute option candles for every expired contract cycle (Upstox expired-instruments "
    "data, from late 2024). Entry and exit use the bar's opening price at the chosen time.\n"
    "- **Strike**: the listed strike nearest to index × (1 ± distance), using the index price at entry.\n"
    "- **Let it expire**: options settle at their value against the index's final close on expiry day.\n"
    "- **Stop-loss**: checked on 15-minute closing prices; real fills in a fast market can be worse.\n"
    f"- **Rupees**: points × today's lot size ({lot}), after estimated costs (₹{L.BROKERAGE_PER_ORDER:.0f} per "
    "order, 0.1% STT on premium sold, ~0.05% other charges). Margin needs are not modelled.\n"
    "- **Sample**: weekly NIFTY expiries give about 110 trades; BANK NIFTY has monthly expiries, so far fewer. "
    "Few trades and one market period: treat results as a description of history, not a promise.")
