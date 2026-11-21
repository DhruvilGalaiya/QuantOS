"""
QuantOS | Volatility & Edge page

Measurements of volatility and of the option seller's edge: the volatility risk premium (India VIX vs the
realized volatility that followed), the volatility cone, GARCH(1,1)-t, one-session ATM straddle outcomes by
days to expiry, and a research corner (rough volatility).
Everything is read from data/analytics/ (python -m quantos_platform.jobs analytics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)

from quantos_platform.analytics import INDICES, read_frame, read_json  # noqa: E402


def pct(x, d: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def pts(x, d: int = 1) -> str:
    """Volatility difference in vol points (0.031 -> +3.1 pts)."""
    return "—" if x is None else f"{x * 100:+.{d}f} pts"


def ordinal(p) -> str:
    if p is None:
        return "—"
    n = int(round(p * 100))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf} pct"


@st.cache_data(ttl=600, show_spinner=False)
def load(symbol: str, _mtime: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    s = symbol.lower()
    return (read_json(f"volatility_{s}.json"), read_frame(f"volatility_{s}_series.parquet"),
            read_frame(f"straddle_decay_{s}.parquet"))


def _mtime(symbol: str) -> float:
    from quantos_platform.core import settings
    p = settings.ANALYTICS_DIR / f"volatility_{symbol.lower()}.json"
    return p.stat().st_mtime if p.exists() else 0.0


# ============================================================
# SIDEBAR + DATA
# ============================================================

with st.sidebar:
    st.subheader("Volatility & Edge")
    symbol = st.selectbox("Index", list(INDICES), format_func=INDICES.get, key="ve_symbol")
    if ui.refresh_button("analytics", "Rebuild analytics", key="ve_refresh"):
        load.clear()

with st.spinner("Checking analytics…"):
    ui.refresh_on_launch("analytics")
ui.job_health("analytics")

V, series, decay = load(symbol, _mtime(symbol))
with st.sidebar:
    st.caption(ui.freshness_caption("analytics", "Analytics built from data", fallback=V.get("as_of")))

theme.hero(
    "Volatility & the house edge",
    "How much the market is moving, how much option prices say it will move, and who has historically "
    "been paid for the difference. Measurements and history, not forecasts of direction.",
    [f"<b>{INDICES[symbol]}</b>", f"Data to <b>{pd.Timestamp(V['as_of']):%d %b %Y}</b>" if V else "No data"],
)
if not V:
    theme.status("No analytics yet. Run: python -m quantos_platform.jobs analytics", "side")
    st.stop()

G, H, vrp = V["garch"], V["hurst"], V.get("vrp")
yz, yz_p = V["realized_now"]["yang_zhang"], V["realized_percentile"]["yang_zhang"]

head = [
    theme.stat("Realized vol, 21 sessions", pct(yz), sub=f"Yang-Zhang · {ordinal(yz_p)} since 2007",
               help="Annualised volatility of the last 21 sessions from open, high, low and close."),
    theme.stat("GARCH expected vol, 21 sessions", pct(G["vol_21d"]), sub=f"Long-run level {pct(G['long_run_vol'])}",
               help="Average volatility over the next 21 sessions implied by today's GARCH(1,1)-t state. "
                    "A risk measure: it says how big moves tend to be, not which way."),
]
if vrp:
    head += [
        theme.stat("India VIX", pct(vrp["vix_now"]), sub=f"{ordinal(vrp['vix_percentile_1y'])} of the past year",
                   help="Implied volatility of NIFTY options over the next 30 calendar days."),
        theme.stat("Premium now (VIX − realized)", pts(vrp["spread_trailing_now"]),
                   tone="bull" if vrp["spread_trailing_now"] > 0 else "bear",
                   sub=f"{ordinal(vrp['spread_trailing_percentile'])} of history",
                   help="Implied minus trailing 21-session realized volatility, in volatility points."),
    ]
theme.stats(head)


# ============================================================
# 1. VOLATILITY RISK PREMIUM
# ============================================================

theme.section("The volatility risk premium",
              "Option prices embed an insurance premium: implied volatility usually sits above the volatility "
              "that actually follows. Option sellers are the house that collects it, and occasionally pays out "
              "in a crash.")
if vrp:
    from plotly.subplots import make_subplots

    s = series.dropna(subset=["vix"])
    vix_s, rv_s = s["vix"] * 100, s["rv_forward"] * 100
    known = rv_s.notna()
    roll = (s["spread_realized"] * 100).rolling(63, min_periods=40).mean()
    period = st.segmented_control("Show", ["1 year", "3 years", "5 years", "Since 2011"], default="3 years",
                                  key="vrp_period", label_visibility="collapsed")
    years = {"1 year": 1, "3 years": 3, "5 years": 5}.get(period or "3 years")
    x0 = s.index.max() - pd.DateOffset(years=years) if years else s.index.min()
    with theme.card("vrp"):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
        # shaded gap: green where VIX ended above what followed (sellers paid), red where it fell short
        up, lo = np.maximum(vix_s, rv_s).where(known), np.minimum(vix_s, rv_s).where(known)
        nohover = dict(mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=s.index, y=rv_s.where(known), **nohover, row=1, col=1)
        fig.add_scatter(x=s.index, y=up, fill="tonexty", fillcolor=theme.alpha(theme.BULL, 0.28),
                        name="Sellers were paid (VIX above what followed)", mode="lines", line=dict(width=0),
                        hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=s.index, y=lo, **nohover, row=1, col=1)
        fig.add_scatter(x=s.index, y=rv_s.where(known), fill="tonexty", fillcolor=theme.alpha(theme.BEAR, 0.35),
                        name="Buyers were paid (market moved more)", mode="lines", line=dict(width=0),
                        hoverinfo="skip", row=1, col=1)
        gap = vix_s - rv_s
        verdict_txt = np.where(~known, "not known yet (needs 21 more sessions)",
                               np.where(gap >= 0, "sellers were paid", "buyers were paid"))
        fig.add_scatter(x=s.index, y=vix_s, name="India VIX: the move options priced in",
                        line=dict(color=theme.ACCENT, width=1.7), row=1, col=1,
                        customdata=np.array(list(zip(rv_s, gap, verdict_txt)), dtype=object),
                        hovertemplate="<b>%{x|%d %b %Y}</b><br>VIX (priced in): <b>%{y:.1f}%</b><br>"
                                      "Actual move next month: <b>%{customdata[0]:.1f}%</b><br>"
                                      "Gap: <b>%{customdata[1]:+.1f} pts</b>, %{customdata[2]}<extra></extra>")
        fig.add_scatter(x=s.index, y=rv_s, name="What the market actually moved over the next month",
                        line=dict(color=theme.TEXT, width=1.1), hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=roll.index, y=roll.clip(lower=0), fill="tozeroy", mode="lines", line=dict(width=0),
                        fillcolor=theme.alpha(theme.BULL, 0.45), hoverinfo="skip", showlegend=False, row=2, col=1)
        fig.add_scatter(x=roll.index, y=roll.clip(upper=0), fill="tozeroy", mode="lines", line=dict(width=0),
                        fillcolor=theme.alpha(theme.BEAR, 0.55), hoverinfo="skip", showlegend=False, row=2, col=1)
        fig.add_scatter(x=roll.index, y=roll, mode="lines", line=dict(color=theme.TEXT, width=1),
                        name="3-month average gap", showlegend=False, row=2, col=1,
                        hovertemplate="%{x|%d %b %Y}<br>3-month average gap: <b>%{y:+.1f} pts</b><extra></extra>")
        fig.add_hline(y=0, line=dict(color=theme.FAINT, width=1), row=2, col=1)
        fig.update_layout(height=520, hovermode="x", legend=dict(orientation="h", y=1.1, x=0),
                          margin=dict(l=8, r=8, t=40, b=8))
        fig.update_xaxes(range=[x0, s.index.max()], showspikes=True, spikemode="across", spikethickness=1,
                         spikecolor=theme.FAINT, spikedash="dot")
        vis = s.index >= x0
        top = float(np.nanmax(np.r_[vix_s[vis], rv_s[vis]])) * 1.1
        bot = float(np.nanmin(np.r_[vix_s[vis], rv_s[vis]])) * 0.85
        g_abs = float(np.nanmax(np.abs(roll[vis]))) * 1.25 if roll[vis].notna().any() else 5
        fig.update_yaxes(title_text="Volatility, % a year", range=[bot, top], row=1, col=1)
        fig.update_yaxes(title_text="Avg gap, pts", range=[-g_abs, g_abs], row=2, col=1, zeroline=False)
        theme.chart(fig, key="vrp_lines")
    win = s.loc[s.index >= x0, "spread_realized"].dropna()
    if len(win):
        theme.stats([
            theme.stat("Days sellers were paid", f"{(win > 0).mean():.0%}", sub=f"in the period shown ({len(win):,} days)"),
            theme.stat("Average gap", pts(win.mean()), tone="bull" if win.mean() > 0 else "bear",
                       sub="VIX minus the move that followed"),
            theme.stat("Biggest seller win", pts(win.max()), sub=f"{win.idxmax():%d %b %Y}"),
            theme.stat("Biggest seller loss", pts(win.min()), tone="bear", sub=f"{win.idxmin():%d %b %Y}"),
        ], enter=False)
    theme.note("How to read it: the blue line is the yearly move option prices were charging for (India VIX); "
               "the black line is how much NIFTY actually moved over the following month. Green shading means "
               "options were priced for more movement than happened, so sellers kept the premium; red means the "
               "market moved more than priced and buyers won. The lower panel smooths the gap over three months. "
               "Hover any day for the exact numbers.")
    theme.insight_html(
        f"Since {pd.Timestamp(vrp['since']):%Y}, India VIX was above the volatility that followed on "
        f"<b>{vrp['share_implied_above']:.0%}</b> of days, by <b>{vrp['mean_spread'] * 100:.1f}</b> volatility "
        f"points on average (median {vrp['median_spread'] * 100:.1f}). The worst miss started on "
        f"<b>{pd.Timestamp(vrp['worst_date']):%d %b %Y}</b> ({vrp['worst_spread'] * 100:+.0f} pts): the premium is "
        "earned in small, steady amounts and given back in rare, large ones.")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        theme.table([("VIX level", ""), ("Range", "num"), ("Implied > realized", "num"),
                     ("Avg premium", "num")],
                    [[theme.esc(b["bucket"]), f"{b['vix_range'][0] * 100:.0f}–{b['vix_range'][1] * 100:.0f}",
                      f"{b['share_implied_above']:.0%}", f"{b['mean_spread'] * 100:+.1f} pts"]
                     for b in vrp["by_vix_level"]])
        theme.note("Buckets use the VIX level known on the day, so there is no look-ahead in the grouping.")
    with c2:
        yr = pd.DataFrame(vrp["yearly"])
        fig3 = go.Figure(go.Bar(x=yr["year"], y=yr["mean_spread"] * 100,
                                marker_color=np.where(yr["mean_spread"] >= 0, theme.BULL, theme.BEAR)))
        fig3.update_layout(height=230, yaxis_title="Avg premium, pts", xaxis=dict(dtick=2))
        theme.chart(fig3, key="vrp_year")
    theme.method(
        "- **Implied**: India VIX close on day *t* (30-calendar-day NIFTY implied volatility).\n"
        "- **Realized that followed**: close-to-close volatility of sessions *t+1 … t+21*, annualised with √252. "
        "It is only known 21 sessions later, so the last month has no value yet.\n"
        "- **Premium now** uses *trailing* realized vol (known today), so it is a proxy for the premium being "
        "priced, not a measure of what will be earned.\n"
        "- VIX averages a 30-day window of strikes and is not the same contract as an ATM straddle; the spread "
        "overstates what a seller nets after costs, slippage and margin.")
else:
    imp = (read_json("today.json").get("symbols", {}).get(symbol, {}) or {}).get("implied", {})
    if imp.get("atm_iv"):
        theme.stats([theme.stat("Front ATM implied vol", pct(imp["atm_iv"]), sub=f"Expiry {imp['expiry']}"),
                     theme.stat("GARCH expected vol, 21 sessions", pct(G["vol_21d"])),
                     theme.stat("Implied − GARCH", pts(imp["atm_iv"] - G["vol_21d"]))], cols=3)
    theme.note("A long implied-volatility history (India VIX) exists only for NIFTY 50, so the full premium "
               "history is shown on the NIFTY 50 view.")


# ============================================================
# 2. VOLATILITY CONE
# ============================================================

theme.section("Volatility cone", "Where today's realized volatility sits in the full history for each "
              "look-back window. A quick way to see whether the market is unusually calm or unusually wild.")
cone = pd.DataFrame(V["cone"])
with theme.card("cone"):
    fig = go.Figure()
    x = cone["horizon"]
    band = dict(mode="lines", line=dict(width=0), hoverinfo="skip")
    fig.add_scatter(x=x, y=cone["p90"] * 100, showlegend=False, **band)
    fig.add_scatter(x=x, y=cone["p10"] * 100, fill="tonexty", fillcolor=theme.alpha(theme.ACCENT, 0.08),
                    name="10th–90th percentile", **band)
    fig.add_scatter(x=x, y=cone["p75"] * 100, showlegend=False, **band)
    fig.add_scatter(x=x, y=cone["p25"] * 100, fill="tonexty", fillcolor=theme.alpha(theme.ACCENT, 0.16),
                    name="25th–75th percentile", **band)
    fig.add_scatter(x=x, y=cone["p50"] * 100, name="Median", mode="lines",
                    line=dict(color=theme.ACCENT, dash="dot", width=1.4))
    fig.add_scatter(x=x, y=cone["current"] * 100, name="Today", mode="lines+markers",
                    line=dict(color=theme.WARM, width=2.4), marker=dict(size=9),
                    customdata=np.c_[cone["percentile"] * 100, cone["max"] * 100],
                    hovertemplate="Today %{y:.1f}% · %{customdata[0]:.0f}th pct<br>"
                                  "Record high %{customdata[1]:.0f}%<extra></extra>")
    top = float(max(cone["p90"].max(), cone["current"].max()) * 100 * 1.25)
    fig.update_layout(height=340, xaxis=dict(title="Look-back window (sessions)", tickvals=list(x), type="log",
                                             unifiedhovertitle=dict(text="Last %{x} sessions")),
                      yaxis=dict(title="Realized vol, %", range=[0, top]),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    theme.chart(fig, key="cone")
c21 = cone.set_index("horizon").loc[21] if 21 in set(cone["horizon"]) else None
if c21 is not None:
    calm = "calmer than usual" if c21["percentile"] < 0.4 else ("more volatile than usual"
                                                                 if c21["percentile"] > 0.6 else "about normal")
    theme.insight_html(f"21-session close-to-close realized volatility is <b>{c21['current'] * 100:.1f}%</b>, the "
                       f"<b>{ordinal(c21['percentile'])}</b> of all 21-session windows since 2007: {calm}. "
                       f"The historical median is {c21['p50'] * 100:.1f}%.")
theme.method("Close-to-close log returns, rolling standard deviation × √252 for each window, percentiles over "
             "every available window since 2007 (the 2008 and 2020 extremes are in the hover, not the axis). Overlapping windows mean the percentiles are smooth but not "
             "independent observations.")


# ============================================================
# 4. GARCH
# ============================================================

theme.section("GARCH(1,1) with fat tails",
              "Volatility clusters: calm follows calm, storms follow storms. GARCH measures how strongly a shock "
              "feeds into tomorrow's variance and how fast it fades back to normal.")
c1, c2 = st.columns([1, 1.25])
with c1:
    theme.stats([
        theme.stat("Shock weight α", f"{G['alpha']:.3f}", help="How much yesterday's surprise moves today's variance."),
        theme.stat("Memory β", f"{G['beta']:.3f}", help="How much of yesterday's variance carries over."),
        theme.stat("Persistence α+β", f"{G['persistence']:.3f}"),
        theme.stat("Shock half-life", f"{G['half_life_days']:.0f} sessions" if G.get("half_life_days") else "—"),
        theme.stat("Tail thickness ν", f"{G['nu']:.1f}", help="Student-t degrees of freedom. Lower = fatter tails; "
                                                          "a normal distribution is ν = ∞."),
        theme.stat("Current conditional vol", pct(G["current_vol"])),
    ], cols=2)
with c2:
    ts = pd.DataFrame(G["term_structure"])
    with theme.card("garch_term"):
        fig = go.Figure()
        fig.add_scatter(x=ts["days"], y=ts["vol"] * 100, mode="lines+markers", name="Expected average vol",
                        line=dict(color=theme.ACCENT, width=2.2))
        if G.get("long_run_vol"):
            fig.add_hline(y=G["long_run_vol"] * 100, line=dict(color=theme.FAINT, dash="dot"),
                          annotation_text="long-run", annotation_font_color=theme.FAINT)
        fig.update_layout(height=260, xaxis_title="Horizon (sessions)", yaxis_title="Annualised vol, %")
        theme.chart(fig, key="garch_term")
with theme.card("garch_hist"):
    gh = series.dropna(subset=["garch_vol"]).iloc[-5 * 252:]
    fig = go.Figure()
    fig.add_scatter(x=gh.index, y=gh["rv21_close"] * 100, name="Realized (21 sessions)",
                    line=dict(color=theme.FAINT, width=1.2))
    fig.add_scatter(x=gh.index, y=gh["garch_vol"] * 100, name="GARCH conditional vol",
                    line=dict(color=theme.ACCENT, width=1.6))
    fig.update_layout(height=280, hovermode="x unified", yaxis_title="Annualised vol, %",
                      legend=dict(orientation="h", y=1.1, x=0))
    theme.chart(fig, key="garch_hist")
theme.insight_html(
    f"A volatility shock loses half its effect in about <b>{G['half_life_days']:.0f} sessions</b>. "
    f"Tails are fat (ν ≈ {G['nu']:.1f}): days several standard deviations from the mean happen far more often "
    f"than a normal distribution allows. Today's conditional volatility ({G['current_vol'] * 100:.1f}%) is "
    + ("above" if G["current_vol"] > (G.get("long_run_vol") or 0) else "below")
    + f" its long-run level ({(G.get('long_run_vol') or 0) * 100:.1f}%), so the model expects it to drift back.")
theme.method(
    r"$r_t = \mu + \varepsilon_t$, $\varepsilon_t = \sigma_t z_t$, $z_t \sim t_\nu$, "
    r"$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$."
    f"\n\nFitted by maximum likelihood on daily log returns from {G['fit_start']} ({G['n_obs']:,} sessions) with "
    "the `arch` package. The term structure averages the forecast daily variances over each horizon. GARCH "
    "describes the *size* of moves, never their direction.")


# ============================================================
# 4. OVERNIGHT STRADDLE: WHAT A SELLER KEPT, BY DAYS TO EXPIRY
# ============================================================

LOT = {"NIFTY_50": 65, "NIFTY_BANK": 30}   # current exchange lot sizes, so rupee figures compare like for like
theme.section("Selling options overnight: what the seller actually kept",
              "The test: at 15:15 sell both the call and the put at the strike closest to the index (an "
              "at-the-money straddle) and buy them back at 15:15 on the next trading day. The seller keeps the "
              "premium if the index stays put and loses if it moves a lot.")
sd = V.get("straddle_decay", {})
if decay.empty or not sd.get("summary"):
    theme.note("Expired option data is stored for NIFTY 50 and NIFTY BANK only.")
else:
    lot = LOT.get(symbol, 1)
    d = decay.assign(rs100=decay["seller_pnl_pct"] * 100, rupees=(decay["entry"] - decay["exit"]) * lot,
                     prem_rs=decay["entry"] * lot)

    def day_label(t: int) -> str:
        return "1 day (day before expiry)" if t == 1 else f"{t} days"

    def verdict_of(g: pd.DataFrame) -> str:
        if g["rupees"].mean() <= 0:
            return '<span class="t-bear">Sellers lost on average</span>'
        if abs(g["rupees"].min()) > 20 * g["rupees"].mean():   # one bad night erases 20+ average nights
            return '<span class="t-side">Wins often, but one bad night wipes out many</span>'
        return '<span class="t-bull">Steady small edge</span>'

    d = d[d["tdte"] <= 5]   # longer gaps (holiday weeks) are too rare to measure
    rows, by = [], d.groupby("tdte")
    for t, g in by:
        rows.append([
            f'<span class="strong">{day_label(int(t))}</span>', f"{len(g)}",
            f"{(g['rs100'] > 0).mean():.0%}",
            theme.rupee_html(g["rupees"].mean()),
            theme.rupee_html(g["rs100"].median()),
            f'<span class="t-bear">{theme.rupees(g["rupees"].min())}</span>',
            verdict_of(g),
        ])
    theme.table([("Days left to expiry", ""), ("Nights", "num"), ("Won", "num"), ("Average (1 lot)", "num"),
                 ("Typical per ₹100", "num"), ("Worst (1 lot)", "num"), ("What history says", "")], rows)
    theme.note(f"Average and worst are rupees for 1 lot at today's lot size ({lot} units), using the option prices "
               "at the time, before brokerage, taxes and slippage. 'Typical per ₹100' is what the median night kept "
               "for every ₹100 of premium collected, so one huge night cannot distort it.")

    summ = by.agg(win=("rs100", lambda x: (x > 0).mean() * 100), avg=("rupees", "mean")).reset_index()
    labels = [day_label(int(t)) for t in summ["tdte"]]
    c1, c2 = st.columns(2)
    with c1:
        with theme.card("decay_win"):
            fig = go.Figure(go.Bar(x=labels, y=summ["win"], marker_color=theme.ACCENT,
                                   text=[f"{v:.0f}%" for v in summ["win"]], textposition="outside",
                                   hovertemplate="%{x}<br>Profitable on <b>%{y:.0f}%</b> of nights<extra></extra>"))
            fig.add_hline(y=50, line=dict(color=theme.FAINT, dash="dot"), annotation_text="coin flip",
                          annotation_font_color=theme.FAINT)
            fig.update_layout(height=270, yaxis=dict(title="Nights in profit, %", range=[0, 100]),
                              xaxis_title="Days left to expiry", title="How often the seller made money",
                              margin=dict(t=44))
            theme.chart(fig, key="decay_win")
    with c2:
        with theme.card("decay_avg"):
            fig = go.Figure(go.Bar(x=labels, y=summ["avg"],
                                   marker_color=np.where(summ["avg"] >= 0, theme.BULL, theme.BEAR),
                                   text=[theme.rupees(v) for v in summ["avg"]], textposition="outside",
                                   hovertemplate="%{x}<br>Average result: <b>₹%{y:+,.0f}</b> per lot<extra></extra>"))
            fig.update_layout(height=270, yaxis_title="₹ per lot", xaxis_title="Days left to expiry",
                              title="Average profit per night, wins and losses together", margin=dict(t=44))
            theme.chart(fig, key="decay_avg")

    pick = st.segmented_control("See every night for", list(summ["tdte"]), default=int(summ["tdte"].min()),
                                format_func=day_label, key="ve_tdte")
    sub = d[d["tdte"] == (pick or summ["tdte"].min())]
    with theme.card("decay_hist"):
        bins = dict(start=-300, end=110, size=10)
        fig = go.Figure()
        fig.add_histogram(x=sub.loc[sub["rs100"] >= 0, "rs100"], xbins=bins, marker_color=theme.BULL,
                          name="Profitable nights",
                          hovertemplate="Kept ₹%{x} per ₹100 collected<br><b>%{y} nights</b><extra></extra>")
        fig.add_histogram(x=sub.loc[sub["rs100"] < 0, "rs100"].clip(lower=-299), xbins=bins,
                          marker_color=theme.BEAR, name="Losing nights",
                          hovertemplate="Result ₹%{x} per ₹100 collected<br><b>%{y} nights</b><extra></extra>")
        fig.update_layout(height=260, barmode="overlay", bargap=0.05, legend=dict(orientation="h", y=1.12, x=0),
                          xaxis=dict(title="Profit or loss per ₹100 of premium collected (₹; losses beyond −300 "
                                           "grouped at the left edge)", ticksuffix=""),
                          yaxis_title="Number of nights")
        theme.chart(fig, key="decay_hist")

    best_t = int(summ.loc[summ["avg"].idxmax(), "tdte"])
    g = d[d["tdte"] == best_t]
    avg_rs, worst_rs = g["rupees"].mean(), g["rupees"].min()
    wins_lost = abs(worst_rs) / avg_rs if avg_rs > 0 else np.nan
    theme.insight_html(
        f"Selling the overnight straddle with <b>{day_label(best_t).split(' (')[0]}</b> left to expiry worked best: "
        f"profitable on <b>{(g['rs100'] > 0).mean():.0%}</b> of {len(g)} nights, earning <b>₹{avg_rs:,.0f}</b> per lot "
        f"on average. But the worst night lost <b>₹{abs(worst_rs):,.0f}</b>"
        + (f", about {wins_lost:.0f} average nights of profit in one go" if np.isfinite(wins_lost) else "")
        + ". The edge is real but thin: sellers win often and small, and lose rarely and big. If you trade it, "
          "size for the worst night, not the average one; a hedge (also buying a cheaper option further out) or "
          "a stop-loss caps that tail at the cost of some of the premium.",
        "What this means for a seller")
    theme.method(
        "Entry and exit are the 15:15 bar open of the call and put at the strike nearest the index on the entry "
        "day. Each trade is held for one session. Per ₹100 collected = (premium sold − premium bought back) ÷ "
        f"premium sold × 100. Sample: {sd['n']} nights from {pd.Timestamp(sd['start']):%b %Y} to "
        f"{pd.Timestamp(sd['end']):%b %Y}; weekly NIFTY gives about 110 nights per row, monthly BANK NIFTY about "
        "28, so BANK rows are much less certain. For reference, textbook time decay (index unchanged, same "
        r"implied volatility) would keep $1-\sqrt{\tau_1/\tau_0}$ of the premium, far more than sellers actually "
        "kept: the overnight move eats most of the decay. Try other strikes and holding times on the Expiry "
        "Playbook page.")


# ============================================================
# 5. RESEARCH CORNER
# ============================================================

theme.section("Research corner", "Ideas from current quantitative-finance research, measured on Indian data. "
              "Interesting for the curious; not needed for day-to-day decisions.")

st.markdown("#### Is volatility rough?")
theme.note("Gatheral, Jaisson and Rosenbaum (2018) showed that log-volatility behaves like fractional Brownian "
              "motion with a Hurst exponent near 0.1: much rougher than the smooth diffusions in classic models.")
c1, c2 = st.columns(2)
with c1:
    with theme.card("rough_scaling"):
        fig = go.Figure()
        for i, (q, m) in enumerate(H["m"].items()):
            fig.add_scatter(x=H["lags"], y=m, mode="markers+lines", name=f"q = {q}",
                            line=dict(width=1.2, color=theme.SERIES[i]), marker=dict(size=4))
        fig.update_layout(height=300, xaxis=dict(type="log", title="Lag Δ (sessions)"),
                          yaxis=dict(type="log", title="E|Δ log σ|^q"), legend=dict(orientation="h", y=1.12, x=0))
        theme.chart(fig, key="rough_scaling")
with c2:
    with theme.card("rough_zeta"):
        z = {float(k): v for k, v in H["zeta"].items()}
        qs = np.array(sorted(z))
        fig = go.Figure()
        fig.add_scatter(x=qs, y=[z[q] for q in qs], mode="markers", name="Measured ζ_q",
                        marker=dict(size=10, color=theme.ACCENT))
        fig.add_scatter(x=qs, y=H["H"] * qs, name=f"q·H, H = {H['H']:.2f}", line=dict(color=theme.ACCENT))
        fig.add_scatter(x=qs, y=0.5 * qs, name="Brownian (H = 0.5)", line=dict(color=theme.FAINT, dash="dot"))
        fig.update_layout(height=300, xaxis_title="Moment q", yaxis_title="Scaling exponent ζ_q",
                          legend=dict(orientation="h", y=1.12, x=0))
        theme.chart(fig, key="rough_zeta")
if H.get("rolling"):
    rh = pd.DataFrame(H["rolling"])
    with theme.card("rough_rolling"):
        fig = go.Figure(go.Scatter(x=pd.to_datetime(rh["end"]), y=rh["H"], mode="lines+markers",
                                   line=dict(color=theme.ACCENT, width=1.6), name="H, 3-year window"))
        fig.add_hline(y=0.5, line=dict(color=theme.FAINT, dash="dot"), annotation_text="Brownian 0.5",
                      annotation_font_color=theme.FAINT)
        fig.update_layout(height=220, yaxis_title="Hurst exponent H", yaxis_range=[0, 0.6])
        theme.chart(fig, key="rough_rolling")
theme.insight_html(
    f"The measured Hurst exponent of {INDICES[symbol]} log-volatility is <b>H ≈ {H['H']:.2f}</b>, far below the "
    "0.5 of a random walk, and it stays low in every 3-year window. Volatility is rough: it jitters sharply "
    "over days, so models built on smooth volatility understate short-dated option risk.")
theme.method(
    f"Volatility proxy: {H['proxy']}. For lags Δ we compute m(q,Δ) = mean |log σ(t+Δ) − log σ(t)|^q, fit "
    "log m ≈ ζ_q log Δ beyond the smoothing window, and H is the slope of ζ_q against q (mono-fractal "
    "scaling means ζ_q = qH). The original study used 5-minute realized variance; daily ranges add "
    "measurement noise, so read the level as indicative and the comparison over time as the robust part.")
