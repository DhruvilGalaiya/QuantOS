"""
QuantOS | Risk Monitor (systemic & tail risk)

How fragile the market is and how bad bad days get: sector co-movement, turbulence, dispersion, a sector
scoreboard, 1-day loss levels (VaR / expected shortfall) with their track record, a one-month range of outcomes
from a GARCH-t Monte Carlo, and replays of past crashes on today's money. Reads data/analytics/risk*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)

from quantos_platform.analytics import read_frame, read_json  # noqa: E402


def pct(x, d: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def ordinal(p) -> str:
    n = int(round(p * 100))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf} percentile"


def level(p) -> str:
    return "very low" if p < 0.1 else "low" if p < 0.35 else "normal" if p <= 0.65 else "high" if p <= 0.9 else "very high"


@st.cache_data(ttl=600, show_spinner=False)
def load(_mtime: float):
    return read_json("risk.json"), read_frame("risk_series.parquet"), read_frame("risk_returns.parquet")


def _mtime() -> float:
    from quantos_platform.core import settings
    p = settings.ANALYTICS_DIR / "risk.json"
    return p.stat().st_mtime if p.exists() else 0.0


with st.sidebar:
    st.subheader("Risk Monitor")
    amount = st.number_input("Money at risk (₹)", min_value=10_000, max_value=100_000_000, value=1_000_000,
                             step=100_000, format="%d", key="rm_amount",
                             help="Used to turn percentages into rupees throughout the page.")
    st.caption(f"= ₹{amount:,.0f}")
    if ui.refresh_button("analytics", "Rebuild analytics", key="rm_refresh"):
        load.clear()

with st.spinner("Checking analytics…"):
    ui.refresh_on_launch("analytics")
ui.job_health("analytics")
R, sysdf, rets = load(_mtime())
with st.sidebar:
    st.caption(ui.freshness_caption("analytics", "Analytics built from data", fallback=R.get("as_of")))

theme.hero(
    "Risk monitor",
    "How fragile the market is right now and how bad bad days can get: whether sectors are moving as one, "
    "whether today's moves are unusual, how much a normal bad day costs, and what past crashes would do to "
    "your money today.",
    [f"Data to <b>{pd.Timestamp(R['as_of']):%d %b %Y}</b>" if R else "No data", "Measurements, not forecasts"],
)
if not R:
    theme.status("No analytics yet. Run: python -m quantos_platform.jobs analytics", "side")
    st.stop()

N = R["now"]
SECT = R["sectors"]
theme.stats([
    theme.stat("Sectors moving together", f"{N['avg_corr']:.2f}", sub=f"{level(N['avg_corr_pct'])} · "
               f"{ordinal(N['avg_corr_pct'])}", help="Average correlation between the 11 sector indices over the "
                                                     "last 3 months (0 = independent, 1 = move as one)."),
    theme.stat("Market turbulence", ordinal(N["turbulence_pct"]).replace(" percentile", " pct"),
               sub=f"{N['turbulent_days_21']} turbulent day(s) in the last month",
               tone="bear" if N["turbulence_pct"] > 0.9 else None,
               help="How unusual today's sector moves are compared with the past year."),
    theme.stat("Sector spread", pct(N["dispersion"]), sub=f"{level(N['dispersion_pct'])} · gap between sectors",
               help="How far apart sector returns are over the last month (cross-sectional spread)."),
    theme.stat("Risk concentration", pct(N["absorption"], 0), sub=f"{level(N['absorption_pct'])} · top 2 drivers",
               help="Share of all sector risk explained by the two biggest common drivers over the past year."),
])


# ============================================================
# 1. MOVING TOGETHER
# ============================================================

theme.section("Are sectors moving together?",
              "When everything moves as one, diversifying across sectors protects less, and shocks spread faster.")
period = st.segmented_control("Show", ["1 year", "3 years", "5 years", "Since 2008"], default="5 years",
                              key="rm_period", label_visibility="collapsed")
yrs = {"1 year": 1, "3 years": 3, "5 years": 5}.get(period or "5 years")
sv = sysdf.dropna(subset=["avg_corr"])
x0 = sv.index.max() - pd.DateOffset(years=yrs) if yrs else sv.index.min()
sv = sv[sv.index >= x0]
c1, c2 = st.columns([1.5, 1])
with c1:
    with theme.card("corr"):
        fig = go.Figure()
        fig.add_scatter(x=sv.index, y=sv["avg_corr"], name="Average correlation between sectors (3 months)",
                        line=dict(color=theme.ACCENT, width=1.8),
                        hovertemplate="%{x|%d %b %Y}<br>Average correlation <b>%{y:.2f}</b><extra></extra>")
        fig.add_scatter(x=sv.index, y=sv["absorption"], name="Risk concentration (top 2 drivers, 1 year)",
                        line=dict(color=theme.WARM, width=1.4, dash="dot"),
                        hovertemplate="%{x|%d %b %Y}<br>Risk concentration <b>%{y:.0%}</b><extra></extra>")
        fig.update_layout(height=330, hovermode="closest", legend=dict(orientation="h", y=1.12, x=0),
                          yaxis=dict(title="0 = independent, 1 = one market", range=[0, 1]))
        theme.chart(fig, key="corr")
with c2:
    cm = R["corr_matrix"]
    with theme.card("heat"):
        fig = go.Figure(go.Heatmap(z=cm["matrix"], x=cm["names"], y=cm["names"], zmin=-0.2, zmax=1,
                                   colorscale=[[0, "#F5F3EE"], [0.5, theme.alpha(theme.ACCENT, 0.45)],
                                               [1, theme.ACCENT]],
                                   hovertemplate="%{y} vs %{x}<br>Correlation <b>%{z:.2f}</b><extra></extra>",
                                   colorbar=dict(thickness=10, len=0.8)))
        fig.update_layout(height=330, xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"),
                          margin=dict(l=8, r=8, t=8, b=8))
        theme.chart(fig, key="heat")
    theme.note("Last 3 months. Sectors that move alike are placed next to each other.")
theme.insight_html(
    f"Sectors are moving together at an average correlation of <b>{N['avg_corr']:.2f}</b>, the "
    f"{ordinal(N['avg_corr_pct'])} since 2008 ({level(N['avg_corr_pct'])}). "
    + ("Low co-movement means stock and sector choice matters more than the overall market right now, and "
       "spreading money across sectors genuinely diversifies." if N["avg_corr_pct"] < 0.35 else
       "High co-movement means most sectors rise and fall together, so diversifying across sectors gives less "
       "protection than usual." if N["avg_corr_pct"] > 0.65 else
       "Co-movement is around its usual level."))


# ============================================================
# 2. TURBULENCE
# ============================================================

theme.section("Turbulence: are today's moves unusual?",
              "Compares each day's moves across all sectors with the past year. A high reading means sectors "
              "are moving by unusual amounts or in unusual combinations.")
TB = R["turbulence_base"]
with theme.card("turb"):
    fig = go.Figure()
    fig.add_scatter(x=sv.index, y=sv["turbulence"], mode="lines", line=dict(color=theme.alpha(theme.TEXT, 0.25),
                    width=0.8), name="Each day", hovertemplate="%{x|%d %b %Y}<br>That day <b>%{y:.1f}</b><extra></extra>")
    fig.add_scatter(x=sv.index, y=sv["turbulence"].rolling(10).mean(), mode="lines",
                    line=dict(color=theme.TEXT, width=1.8), name="10-day average",
                    hovertemplate="%{x|%d %b %Y}<br>10-day average <b>%{y:.1f}</b><extra></extra>")
    fig.add_hline(y=N["turbulence_p95"], line=dict(color=theme.BEAR, dash="dot"),
                  annotation_text="turbulent day (top 5%)", annotation_font_color=theme.BEAR,
                  annotation_position="top left")
    hot = sv[sv["turbulence"] > N["turbulence_p95"]]
    fig.add_scatter(x=hot.index, y=hot["turbulence"], mode="markers", marker=dict(color=theme.BEAR, size=5),
                    name="Turbulent day", hovertemplate="%{x|%d %b %Y}<br>Turbulent: <b>%{y:.1f}</b><extra></extra>")
    fig.update_layout(height=300, legend=dict(orientation="h", y=1.1, x=0), hovermode="closest",
                      yaxis=dict(title="Turbulence (1 = typical)",
                                 range=[0, float(np.nanpercentile(sv["turbulence"], 99.5)) * 1.15]))
    theme.chart(fig, key="turb")
theme.insight_html(
    f"After the {TB['n_turbulent']} turbulent days since 2009, NIFTY's volatility over the next month averaged "
    f"<b>{pct(TB['fwd_vol_turbulent'])}</b> against <b>{pct(TB['fwd_vol_all'])}</b> normally: rough patches tend "
    f"to continue. Returns were not worse on average ({pct(TB['fwd_ret_turbulent'], 1, True)} vs "
    f"{pct(TB['fwd_ret_all'], 1, True)}), so turbulence is a warning about the size of moves, not their direction.")


# ============================================================
# 3. SECTOR SCOREBOARD
# ============================================================

theme.section("Sector scoreboard", f"Returns and risk of each sector. The gap between sectors over the last month "
              f"is {pct(N['dispersion'])}, the {ordinal(N['dispersion_pct'])}.")
sb = sorted(R["scoreboard"], key=lambda r: r["m1"], reverse=True)
theme.table([("Sector", ""), ("1 week", "num"), ("1 month", "num"), ("3 months", "num"), ("1 year", "num"),
             ("From 52-week high", "num"), ("52-week range", ""), ("Volatility", "num")],
            [[f'<span class="strong">{theme.esc(r["name"])}</span>',
              theme.signed_html(r["w1"] * 100, "{:+.1f}%"), theme.signed_html(r["m1"] * 100, "{:+.1f}%"),
              theme.signed_html(r["m3"] * 100, "{:+.1f}%"), theme.signed_html(r["y1"] * 100, "{:+.1f}%"),
              pct(r["off_high"], 1, True) if r["off_high"] < -0.001 else "At the high",
              theme.mini_bar(r["range52"] or 0, f"{(r['range52'] or 0):.0%}"), pct(r["vol21"], 0)]
             for r in sb])
theme.note("52-week range: where the sector sits between its lowest (0%) and highest (100%) close of the past year. "
           "Volatility: last 21 sessions, yearly terms.")


# ============================================================
# 4. HOW BAD CAN A BAD DAY GET
# ============================================================

TAIL = R["tail"]
theme.section("How bad can a bad day get?", "Loss levels for a single day, in percent and in rupees on the amount "
              "set in the sidebar.")
tsym = st.segmented_control("Index", list(TAIL), default="NIFTY_50", format_func=lambda s: TAIL[s]["name"],
                            key="rm_tail", label_visibility="collapsed") or "NIFTY_50"
t = TAIL[tsym]
f95, f99, bt = t["fhs"]["0.95"], t["fhs"]["0.99"], t["backtest"]


def rs(x):
    v = x * amount
    return f"{'−' if v < 0 else ''}₹{abs(v):,.0f}"


theme.stats([
    theme.stat("1 day in 20, a fall of at least", pct(f95["var"], 2), sub=f"{rs(f95['var'])} on your amount",
               tone="bear", help="Value at Risk (95%), from the current volatility level (GARCH-t filtered history)."),
    theme.stat("On those bad days, the average fall", pct(f95["cvar"], 2), sub=rs(f95["cvar"]), tone="bear",
               help="Expected shortfall (95%): the average loss on the worst 5% of days."),
    theme.stat("1 day in 100, a fall of at least", pct(f99["var"], 2), sub=rs(f99["var"]), tone="bear"),
    theme.stat("Worst day on record", pct(t["worst_days"][0]["ret"], 1, True),
               sub=f"{pd.Timestamp(t['worst_days'][0]['date']):%d %b %Y} · {rs(-t['worst_days'][0]['ret'])}",
               tone="bear"),
])
h95 = t["hist"]["all_95"]
alpha = t["hill_alpha"]
theme.insight_html(
    f"With volatility where it is today, {t['name']} loses <b>{pct(f95['var'], 1)}</b> or more on about one day in "
    f"twenty (roughly once a month) and <b>{pct(f99['var'], 1)}</b> or more on one day in a hundred. Over all history "
    f"since 2007 the one-in-twenty level was {pct(h95['var'], 1)}. Track record: tested on {bt['n']:,} days since "
    f"{pd.Timestamp(bt['start']):%Y} using only data available at the time, the one-in-twenty line was crossed on "
    f"<b>{bt['rate95']:.1%}</b> of days (target 5%) and the one-in-a-hundred line on {bt['rate99']:.1%} "
    f"(target 1%). Big falls follow a power law: a fall twice as large is about <b>{2 ** alpha:.0f} times rarer</b>, "
    "far more common than a bell curve would suggest.", "In plain words")

c1, c2 = st.columns([1.3, 1])
mc = t["mc"]
with c1:
    with theme.card("mc"):
        edges = np.array(mc["edges"]) * 100
        mids = (edges[:-1] + edges[1:]) / 2
        cnt = np.array(mc["hist"])
        fig = go.Figure(go.Bar(x=mids, y=cnt / cnt.sum() * 100, marker_color=np.where(mids < 0, theme.BEAR,
                                                                                         theme.BULL),
                               hovertemplate="One month later: %{x:+.1f}%<br>%{y:.1f}% of simulations<extra></extra>"))
        for q, txt in ((mc["p05"], "1 in 20 worse"), (mc["p95"], "1 in 20 better")):
            fig.add_vline(x=q * 100, line=dict(color=theme.TEXT, dash="dot"), annotation_text=txt,
                          annotation_font_color=theme.MUTED)
        fig.update_layout(height=290, bargap=0.05, xaxis=dict(title="Change over the next 21 sessions, %",
                                                              ticksuffix="%",
                                                              range=[mc["p01"] * 130, mc["p95"] * 160]),
                          yaxis_title="Share of simulations, %", title="Range of outcomes over the next month",
                          margin=dict(t=44))
        theme.chart(fig, key="mc")
with c2:
    theme.table([("Over the next month", ""), ("Index change", "num"), ("On your amount", "num")],
                [["1 in 20 worse than", pct(mc["p05"], 1, True), rs(mc["p05"])],
                 ["1 in 4 worse than", pct(mc["p25"], 1, True), rs(mc["p25"])],
                 ["Middle outcome", pct(mc["p50"], 1, True), rs(mc["p50"])],
                 ["1 in 4 better than", pct(mc["p75"], 1, True), rs(mc["p75"])],
                 ["1 in 20 better than", pct(mc["p95"], 1, True), rs(mc["p95"])],
                 ["Chance of a 10%+ fall", f"{mc['down10']:.1%}", ""]])
    theme.note("10,000 simulated months from a GARCH-t model fitted to the last 10 years. A risk range for "
               "today's volatility, not a forecast of direction.")


# ============================================================
# 5. CRASH REPLAY
# ============================================================

theme.section("Crash replay: what past crashes would do to your money today",
              "Pick a past crash and what you hold. We apply that period's actual daily moves to your amount.")
CR = R["crashes"]
names_all = {**R["indices"], **{k: v for k, v in SECT.items() if k not in R["indices"]}}
c1, c2 = st.columns([1, 1.4])
crash = c1.selectbox("Crash", range(len(CR)), key="rm_crash",
                     format_func=lambda i: f"{pd.Timestamp(CR[i]['peak']):%b %Y} – "
                                           f"{pd.Timestamp(CR[i]['trough']):%b %Y} (NIFTY {CR[i]['depth'] * 100:.0f}%)")
held = c2.multiselect("What you hold (equal weights)", [k for k in names_all if k in rets],
                      default=["NIFTY_50"], format_func=names_all.get, key="rm_held")
if held:
    cr = CR[crash]
    w = rets.loc[(rets.index > pd.Timestamp(cr["peak"])) & (rets.index <= pd.Timestamp(cr["trough"]) +
                                                              pd.Timedelta(days=90)), held]
    have = [h for h in held if w[h].notna().mean() > 0.9]
    if not have:
        theme.status("No data for these holdings in that period.", "side")
    else:
        port = np.expm1(w[have].fillna(0)).mean(axis=1)          # daily rebalanced equal weight
        value = amount * (1 + port).cumprod()
        bottom = value.idxmin()
        with theme.card("replay"):
            fig = go.Figure()
            fig.add_scatter(x=value.index, y=value, mode="lines", line=dict(color=theme.BEAR, width=2),
                            fill="tozeroy", fillcolor=theme.alpha(theme.BEAR, 0.06), name="Your money",
                            hovertemplate="%{x|%d %b %Y}<br>Your money <b>₹%{y:,.0f}</b><extra></extra>")
            fig.add_hline(y=amount, line=dict(color=theme.FAINT, dash="dot"), annotation_text="starting amount",
                          annotation_font_color=theme.FAINT)
            fig.add_vline(x=pd.Timestamp(cr["trough"]), line=dict(color=theme.FAINT, dash="dash"),
                          annotation_text="NIFTY bottom", annotation_font_color=theme.FAINT)
            fig.update_layout(height=300, showlegend=False,
                              yaxis=dict(title="Value, ₹", tickprefix="₹", tickformat=",.0f",
                                         range=[value.min() * 0.9, max(value.max(), amount) * 1.05]))
            theme.chart(fig, key="replay")
        theme.stats([
            theme.stat("Lowest point", f"₹{value.min():,.0f}", sub=f"{value.min() / amount - 1:+.1%} on "
                       f"{bottom:%d %b %Y}", tone="bear"),
            theme.stat("Worst single day", pct(port.min(), 1, True), sub=f"{rs(port.min())} · "
                       f"{port.idxmin():%d %b %Y}", tone="bear"),
            theme.stat("Days from start to bottom", f"{value.index.get_loc(bottom) + 1}"),
            theme.stat("Three months after NIFTY's bottom", f"₹{value.iloc[-1]:,.0f}",
                       sub=f"{value.iloc[-1] / amount - 1:+.1%} from start"),
        ], enter=False)
        if len(have) < len(held):
            theme.note("Left out (no data for that period): " + ", ".join(names_all[h] for h in held if h not in have))
        theme.insight_html(
            f"Had this crash started today, {', '.join(names_all[h] for h in have)} would have taken "
            f"<b>₹{amount:,.0f}</b> down to <b>₹{value.min():,.0f}</b> at the worst point. Ask whether you could "
            "hold through that without selling at the bottom; that, more than any forecast, decides how much "
            "risk is right for you.")

theme.method(
    "- **Sectors moving together**: average pairwise correlation of daily returns of 11 NIFTY sector indices, "
    "rolling 63 sessions. **Risk concentration** (absorption ratio, Kritzman et al. 2011): share of the variance "
    "explained by the top 2 principal components, rolling 252 sessions.\n"
    "- **Turbulence** (Kritzman & Li 2010): Mahalanobis distance of today's sector returns from the trailing "
    "year's mean and covariance (today excluded), divided by the number of sectors so 1 is typical. Turbulent = "
    "above the 95th percentile known at the time.\n"
    "- **Sector spread**: cross-sectional standard deviation of 21-session sector returns.\n"
    "- **Bad-day levels**: filtered historical simulation. A GARCH(1,1) Student-t model is fitted to the last "
    "10 years; past standardised shocks are rescaled by tomorrow's volatility. Track record: refit every January "
    "on the previous 10 years and tested on the following year (Kupiec test p-value "
    f"{bt['p95']:.2f} for the 95% level; above 0.05 means consistent with 5%).\n"
    "- **Power law**: Hill estimator on the largest 5% of daily losses (tail index α ≈ "
    f"{alpha:.1f}; a loss twice as large is 2^α times rarer).\n"
    "- **Range of outcomes**: 10,000 simulated 21-session paths from the same GARCH-t model.\n"
    "- **Crash replay**: actual daily returns from the NIFTY peak to 90 calendar days after its bottom, applied to "
    "your amount with equal daily-rebalanced weights. Past crashes differ; the next one will too.")
