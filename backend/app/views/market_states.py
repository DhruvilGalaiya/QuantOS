"""
QuantOS | Market States

A description of the market's state, not a forecast: which volatility state the index is in (Calm, Normal,
Stressed, from a walk-forward HMM with forward filtering only), how long states usually last, how the market
behaved in each, and the full drawdown history with base rates. Reads data/analytics/states_*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)

from quantos_platform.analytics import read_frame, read_json  # noqa: E402

SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK"}
WORD = {"CALM": "Calm", "NORMAL": "Normal", "STRESSED": "Stressed"}
TONE = {"CALM": "bull", "NORMAL": "side", "STRESSED": "bear"}
COLOR = {"CALM": theme.BULL, "NORMAL": theme.SIDE, "STRESSED": theme.BEAR}
MEANING = {"CALM": "small, steady daily moves", "NORMAL": "ordinary day-to-day swings",
           "STRESSED": "large, fast moves in both directions"}


def pct(x, d: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def chip(state: str) -> str:
    return f'<span class="q-chip t-{TONE[state]}">{WORD[state]}</span>'


@st.cache_data(ttl=600, show_spinner=False)
def load(symbol: str, _mtime: float) -> tuple[dict, pd.DataFrame]:
    s = symbol.lower()
    return read_json(f"states_{s}.json"), read_frame(f"states_{s}_series.parquet")


def _mtime(symbol: str) -> float:
    from quantos_platform.core import settings
    p = settings.ANALYTICS_DIR / f"states_{symbol.lower()}.json"
    return p.stat().st_mtime if p.exists() else 0.0


with st.sidebar:
    st.subheader("Market States")
    symbol = st.selectbox("Index", list(SYMBOLS), format_func=SYMBOLS.get, key="ms_symbol")
    if ui.refresh_button("analytics", "Rebuild analytics", key="ms_refresh"):
        load.clear()

with st.spinner("Checking analytics…"):
    ui.refresh_on_launch("analytics")
ui.job_health("analytics")
S, ser = load(symbol, _mtime(symbol))
with st.sidebar:
    st.caption(ui.freshness_caption("analytics", "Analytics built from data", fallback=S.get("as_of")))

theme.hero(
    "Market states",
    "Markets switch between calm stretches and stormy ones. This page shows which state the market is in now, "
    "how long such states usually last and how the market behaved in each, plus every major fall since 2007. "
    "A description of the present and of history, not a forecast.",
    [f"<b>{SYMBOLS[symbol]}</b>", f"Data to <b>{pd.Timestamp(S['as_of']):%d %b %Y}</b>" if S else "No data"],
)
if not S:
    theme.status("No analytics yet. Run: python -m quantos_platform.jobs analytics", "side")
    st.stop()

cur, cond, T = S["current"], S["conditional"], S["transition"]
state = cur["state"]
c = cond[state]


# ============================================================
# NOW
# ============================================================

daily_move = c["vol_annual"] / np.sqrt(252)
theme.signal(
    f"{SYMBOLS[symbol]} is in a {WORD[state]} state",
    f"For <b>{cur['streak']} session{'s' if cur['streak'] != 1 else ''}</b> (since "
    f"{pd.Timestamp(cur['since']):%d %b %Y}). In this state the market has typically seen {MEANING[state]}: "
    f"a typical day moves about <b>±{daily_move * 100:.1f}%</b>.",
    TONE[state],
)
probs = cur["probs"]
still5, still21 = cur.get("still_after_5"), cur.get("still_after_21")
theme.stats([
    theme.stat("Model confidence", f"{probs[state]:.0%}", sub=" · ".join(f"{WORD[s]} {probs[s]:.0%}" for s in WORD),
               help="The model's probability for each state today, using data up to today only."),
    theme.stat("Typical length of this state", f"{S['episodes'][state]['median_days']:.0f} sessions"
               if S["episodes"][state]["median_days"] else "—",
               sub=f"median of {S['episodes'][state]['n']} past episodes"),
    theme.stat("Still running a week later", f"{still5:.0%}" if still5 is not None else "—",
               sub=f"of past {WORD[state]} spells that reached {cur['streak']} sessions"),
    theme.stat("Still running a month later", f"{still21:.0%}" if still21 is not None else "—",
               sub="21 more sessions"),
])

theme.section("How the market behaves in each state",
              "Measured on the day after each labelled day, so no future information is used.")
rows = []
for s in WORD:
    x = cond[s]
    rows.append([
        chip(s) + (" ← now" if s == state else ""), f"{x['share_of_days']:.0%}",
        f"±{x['vol_annual'] / np.sqrt(252) * 100:.1f}%", pct(x["vol_annual"]),
        f"{x['big_down_share'] * 100:.1f}%",
        theme.signed_html(x["fwd21_mean"] * 100, "{:+.1f}%"),
        f"{pct(x['fwd21_p10'], 0, True)} to {pct(x['fwd21_p90'], 0, True)}",
    ])
a = cond["ALL"]
rows.append(['<span class="t-muted">All days</span>', "100%", f"±{a['vol_annual'] / np.sqrt(252) * 100:.1f}%",
             pct(a["vol_annual"]), f"{a['big_down_share'] * 100:.1f}%",
             theme.signed_html(a["fwd21_mean"] * 100, "{:+.1f}%"),
             f"{pct(a['fwd21_p10'], 0, True)} to {pct(a['fwd21_p90'], 0, True)}"])
theme.table([("State", ""), ("Share of time", "num"), ("Typical day", "num"), ("Yearly volatility", "num"),
             ("Days falling 2%+", "num"), ("Next month, average", "num"),
             ("Next month, 8 in 10 within", "num")], rows, current=list(WORD).index(state))
st_ = cond["STRESSED"]
theme.insight_html(
    f"Stressed spells are rare ({st_['share_of_days']:.0%} of days) but carry most of the risk: a 2%+ fall "
    f"happened on {st_['big_down_share'] * 100:.0f}% of the days that followed, against "
    f"{cond['CALM']['big_down_share'] * 100:.1f}% in Calm spells. Their average next month was "
    f"{pct(st_['fwd21_mean'], 1, True)}, because sharp rebounds also happen in stressed markets. The state tells "
    "you how big the swings tend to be, which is what sizing and stop-losses should respond to.")


# ============================================================
# TIMELINE
# ============================================================

theme.section("State timeline", "The index with the state the model assigned each day.")
period = st.segmented_control("Show", ["1 year", "3 years", "5 years", "Since 2011"], default="5 years",
                              key="ms_period", label_visibility="collapsed")
yrs = {"1 year": 1, "3 years": 3, "5 years": 5}.get(period or "5 years")
sv = ser.dropna(subset=["state"])
if yrs:
    sv = sv[sv.index >= sv.index.max() - pd.DateOffset(years=yrs)]
with theme.card("timeline"):
    fig = go.Figure()
    runs = (sv["state"] != sv["state"].shift()).cumsum()
    lo_y, hi_y = float(sv["close"].min()), float(sv["close"].max())
    segs = [seg for _, seg in sv.groupby(runs)]
    for i, seg in enumerate(segs):           # each band runs to the next band's start, so no weekend gaps
        x1 = segs[i + 1].index[0] if i + 1 < len(segs) else seg.index[-1]
        fig.add_vrect(x0=seg.index[0], x1=x1, fillcolor=COLOR[seg["state"].iloc[0]],
                      opacity=0.14, line_width=0, layer="below")
    fig.add_scatter(x=sv.index, y=sv["close"], mode="lines", line=dict(color=theme.TEXT, width=1.4),
                    customdata=sv["state"].map(WORD), showlegend=False,
                    hovertemplate="%{x|%d %b %Y}<br><b>%{y:,.0f}</b><br>%{customdata} state<extra></extra>")
    for s in WORD:
        fig.add_scatter(x=[None], y=[None], mode="markers", name=WORD[s], hoverinfo="skip",
                        marker=dict(size=11, symbol="square", color=COLOR[s], opacity=0.5))
    fig.update_layout(height=380, hovermode="closest", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis=dict(title=f"{SYMBOLS[symbol]}", tickformat=",.0f",
                                 range=[lo_y * 0.97, hi_y * 1.03]))
    theme.chart(fig, key="timeline")


# ============================================================
# HOW LONG STATES LAST
# ============================================================

theme.section("How long states last", "Each line shows the share of past spells still running after a "
              "given number of sessions.")
c1, c2 = st.columns([1.05, 1])
with c1:
    with theme.card("survival"):
        fig = go.Figure()
        for s, curve in S["survival"].items():
            k = np.arange(1, len(curve) + 1)
            fig.add_scatter(x=k, y=np.array(curve) * 100, name=WORD[s], line=dict(color=COLOR[s], width=2.2),
                            hovertemplate=f"{WORD[s]}: %{{y:.0f}}% still running after %{{x}} sessions"
                                          "<extra></extra>")
        fig.add_vline(x=cur["streak"], line=dict(color=COLOR[state], dash="dot"),
                      annotation_text=f"today: {cur['streak']} sessions", annotation_font_color=COLOR[state])
        fig.update_layout(height=320, hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0),
                          xaxis=dict(title="Sessions since the state began",
                                     unifiedhovertitle=dict(text="After %{x} sessions")),
                          yaxis=dict(title="Spells still running, %", range=[0, 102]))
        theme.chart(fig, key="survival")
with c2:
    trows = []
    for i, s in enumerate(WORD):
        e = S["episodes"][s]
        nxt = e["next"]
        total = sum(nxt.values())
        goes = "<br>".join(f"{WORD[t]} {nxt[t] / total:.0%}" for t in WORD if t != s and total) if total else "—"
        trows.append([chip(s), str(e["n"]), f"{e['median_days']:.0f}" if e["median_days"] else "—",
                      str(e["longest"] or "—"), goes])
    theme.table([("State", ""), ("Spells", "num"), ("Typical", "num"), ("Longest", "num"),
                 ("Then turned", "")], trows)
    theme.note("Lengths in trading sessions. Completed spells only; the current one is still running.")

st.markdown("##### Chance of staying in the same state tomorrow")
P, lo, hi = np.array(T["p"]), np.array(T["lo"]), np.array(T["hi"])
mrows = []
for i, s in enumerate(WORD):
    mrows.append([chip(s)] + [
        (f'<span class="strong">{P[i, j]:.1%}</span>' if i == j else f"{P[i, j]:.1%}")
        + f'<br><span class="dim">{lo[i, j]:.1%} – {hi[i, j]:.1%}</span>' for j in range(3)])
theme.table([("From today's state", "")] + [(f"Tomorrow {WORD[s]}", "num") for s in WORD], mrows,
            current=list(WORD).index(state))
d = T["duration"][list(WORD).index(state)]
theme.insight_html(
    f"From a {WORD[state]} day, the market stayed {WORD[state]} the next day "
    f"<b>{P[list(WORD).index(state), list(WORD).index(state)]:.0%}</b> of the time, which implies spells of about "
    f"<b>{d:.0f} sessions</b> on average (90% band {T['dur_lo'][list(WORD).index(state)]:.0f}–"
    f"{T['dur_hi'][list(WORD).index(state)]:.0f}). States are sticky: today's state is the best single description "
    "of the next few days, but it says little about the next few months.")


# ============================================================
# DRAWDOWNS
# ============================================================

DD = S["drawdown"]
theme.section("Every major fall since 2007", "How far the index has been below its previous high, and how long "
              "each fall took to recover.")
theme.stats([
    theme.stat("Now", DD["stage"].split(" (")[0].replace("At or near the high", "Near the high"),
               sub=f"{pct(DD['now'], 1, True)} from the high of {DD['peak']:,.0f}",
               tone="bear" if DD["now"] < -0.1 else None),
    *[theme.stat(f"Falls of {int(b['from'] * 100)}% that reached {int(b['to'] * 100)}%",
                 f"{b['went_on']} of {b['n']}", sub=(f"typical recovery {b['median_recover_days']:.0f} sessions"
                                                     if b["median_recover_days"] else None))
      for b in DD["base_rates"]],
], enter=False)
dd = ser["drawdown"].dropna()
with theme.card("underwater"):
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", mode="lines",
                               line=dict(color=theme.BEAR, width=1), fillcolor=theme.alpha(theme.BEAR, 0.18),
                               hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f}%</b> below the previous high"
                                             "<extra></extra>"))
    for lvl, txt in ((-10, "correction"), (-20, "bear market")):
        fig.add_hline(y=lvl, line=dict(color=theme.FAINT, dash="dot"), annotation_text=txt,
                      annotation_font_color=theme.FAINT, annotation_position="bottom left")
    fig.update_layout(height=280, yaxis=dict(title="Below previous high, %", ticksuffix="%"), showlegend=False)
    theme.chart(fig, key="underwater")
eps = pd.DataFrame(DD["episodes"])
big = eps[eps["depth"] <= -0.10].sort_values("peak", ascending=False)
theme.table([("Peak", ""), ("Bottom", ""), ("Fall", "num"), ("Sessions falling", "num"),
             ("Sessions to recover", "num")],
            [[pd.Timestamp(r.peak).strftime("%d %b %Y"), pd.Timestamp(r.trough).strftime("%d %b %Y"),
              f'<span class="t-bear">{r.depth * 100:.1f}%</span>', str(r.days_down),
              ("still recovering" if r.ongoing else f"{int(r.days_to_recover)}")] for r in big.itertuples()])
b10 = next((b for b in DD["base_rates"] if b["from"] == 0.10), None)
if b10 and b10["n"]:
    theme.insight_html(
        f"Since 2007, {SYMBOLS[symbol]} had {b10['n']} falls of 10% or more; {b10['went_on']} of them "
        f"({b10['went_on'] / b10['n']:.0%}) deepened into a 20% bear market. The typical 10% fall took "
        f"{b10['median_recover_days']:.0f} sessions to get back to the old high. Most corrections do not become "
        "bear markets, but the ones that do take far longer to heal.")

theme.method(
    "- **States**: a 3-state Gaussian hidden Markov model on daily return, 20-day volatility and the daily change "
    "in India VIX. It is refit every January using only earlier years, and each day's state uses the forward "
    "filter (data up to that day), never the smoothed history, so there is no look-ahead.\n"
    f"- **Switching rule**: the label changes only when another state's probability reaches {S['switch_prob']:.0%}, "
    "which stops one-day flickers. States are named by volatility (Calm, Normal, Stressed), not direction.\n"
    "- **Behaviour by state**: returns of the session after each labelled day, and the 21 sessions after it.\n"
    "- **Durations**: Kaplan-Meier survival of completed spells (the current spell is treated as still running). "
    "Bands on the transition table: 90% moving-block bootstrap (63-session blocks, 1,000 resamples).\n"
    "- **Falls**: peak-to-trough on daily closes since 2007; recovery = first close back at the old peak.")
