"""
QuantOS | Today (default page)

One-screen market brief for NIFTY 50 and NIFTY BANK: price context, regime, volatility and its premium,
the range option prices are paying for by expiry, price signals that fired this week with their base rates,
and the coming week's scheduled news. Reads data/analytics/today.json and news_calendar.json.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)

from quantos_platform.analytics import read_json  # noqa: E402


def pct(x, d: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def ordinal(p) -> str:
    if p is None:
        return "—"
    n = int(round(p * 100))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf} percentile"


def level_word(p) -> str:
    if p is None:
        return "—"
    return "very low" if p < 0.1 else "low" if p < 0.35 else "normal" if p <= 0.65 else "high" if p <= 0.9 else "very high"


@st.cache_data(ttl=300, show_spinner=False)
def load(_mtime: float) -> dict:
    return read_json("today.json")


@st.cache_data(ttl=300, show_spinner=False)
def load_news(_mtime: float) -> dict:
    return read_json("news_calendar.json")


def _mtime(name: str = "today.json") -> float:
    from quantos_platform.core import settings
    p = settings.ANALYTICS_DIR / name
    return p.stat().st_mtime if p.exists() else 0.0


with st.sidebar:
    st.subheader("Today")
    if ui.refresh_button("analytics", "Rebuild brief", key="td_refresh"):
        load.clear()

with st.spinner("Checking analytics…"):
    ui.refresh_on_launch("analytics")
ui.job_health("analytics")
B = load(_mtime())
syms = B.get("symbols", {})
as_of = max((v["market"]["as_of"] for v in syms.values()), default=None)
with st.sidebar:
    st.caption(ui.freshness_caption("analytics", "Brief built from data", fallback=as_of))

theme.hero(
    "Today in the market",
    "The state of Indian index markets in one screen: where prices sit, how much they are moving, what option "
    "sellers are being paid, and what history says about this week's events. Analysis, not advice.",
    [f"Session <b>{pd.Timestamp(as_of):%a %d %b %Y}</b>" if as_of else "No data yet",
     "Measurements and base rates, no forecasts"],
)
if not syms:
    theme.status("No brief yet. Run: python -m quantos_platform.jobs analytics", "side")
    st.stop()


# ============================================================
# INDEX CARDS
# ============================================================

cols = st.columns(len(syms), gap="large")
for col, (sym, d) in zip(cols, syms.items()):
    m, r, v, imp = d["market"], d.get("regime", {}), d.get("vol", {}), d.get("implied", {})
    with col:
        regime = r.get("regime")
        chip = theme.chip(regime) if regime else ""
        streak = f" · {r['streak']} sessions" if r.get("streak") else ""
        theme.signal(
            d["name"],
            f'<span style="font-family:var(--q-serif);font-size:2.1rem;color:var(--q-text)">{m["close"]:,.2f}</span>'
            f'&nbsp;&nbsp;{theme.signed_html(m["change_pct"] * 100)}<br>'
            f'<span class="t-muted">Regime</span> {chip}<span class="t-muted">{streak}</span>',
            theme.tone_of(regime) if regime else "muted",
        )
        stt = read_json(f"states_{sym.lower()}.json").get("current")
        items = [
            *([theme.stat("Market state", {"CALM": "Calm", "NORMAL": "Normal", "STRESSED": "Stressed"}[stt["state"]],
                          tone={"CALM": "bull", "NORMAL": "side", "STRESSED": "bear"}[stt["state"]],
                          sub=f"for {stt['streak']} sessions · volatility-based",
                          help="Calm, Normal or Stressed, from how violently the market is moving. See Market States.")]
              if stt else []),
            theme.stat("From all-time high", pct(m["drawdown"], 1, True) if m["drawdown"] < 0 else "At the high",
                       tone="bear" if m["drawdown"] < -0.1 else None, sub=f"High {m['ath']:,.0f}"),
            theme.stat("52-week range position", f"{(m['range52_pos'] or 0):.0%}",
                       sub=f"{m['lo52']:,.0f} – {m['hi52']:,.0f}"),
            theme.stat("1 week / 1 month", f"{pct(m['week_pct'], 1, True)} / {pct(m['month_pct'], 1, True)}"),
            theme.stat("Realized vol, 21 sessions", pct(v.get("rv21")),
                       sub=f"{level_word(v.get('rv21_percentile'))} · {ordinal(v.get('rv21_percentile'))}"),
        ]
        if "vix" in v:
            items += [
                theme.stat("India VIX", pct(v["vix"]), sub=f"{ordinal(v['vix_percentile_1y'])} of past year"),
                theme.stat("Volatility premium", f"{v['vrp_now'] * 100:+.1f} pts",
                           tone="bull" if v["vrp_now"] > 0 else "bear",
                           sub=f"VIX − realized · {ordinal(v['vrp_percentile'])}"),
            ]
        elif imp.get("atm_iv"):
            items += [
                theme.stat("Front ATM implied vol", pct(imp["atm_iv"]), sub=f"expiry {imp['expiry']}"),
                theme.stat("Implied − GARCH", f"{(v.get('iv_vs_garch') or 0) * 100:+.1f} pts",
                           sub=f"GARCH 21-session {pct(v.get('garch_21d'))}"),
            ]
        if imp.get("move_points"):
            items.append(theme.stat("Expected range by expiry", f"{imp['lower']:,.0f} – {imp['upper']:,.0f}"
                                    if imp.get("lower") else f"±{imp['move_points']:,.0f}",
                                    sub=f"±{imp['move_points']:,.0f} pts (±{imp['move_pct'] * 100:.2f}%) by "
                                        f"{pd.Timestamp(imp['expiry']):%d %b}",
                                    help="The range option prices are paying for until this expiry; explained "
                                         "in the section below."))
        theme.stats(items, cols=2)


# ============================================================
# WHAT OPTIONS ARE PRICING FOR THIS EXPIRY
# ============================================================

imp_rows = [(d, d.get("implied", {})) for d in syms.values() if d.get("implied", {}).get("move_points")]
if imp_rows:
    theme.section("How far the market is expected to move by expiry",
                  "Option prices reveal how big a move traders are paying for. No forecast of ours is involved.")
    theme.insight_html(
        "<b>In plain words:</b> the price of the at-the-money call and put tells you how much movement the market "
        "is insuring against. From that we get a range around today's level. By option pricing the index finishes "
        "<b>inside the range about 2 times in 3</b> (68%) and outside it about 1 time in 3. Option sellers profit if "
        "the index stays inside; option buyers need a move bigger than this to profit.", "What this is")
    rows = []
    for d, imp in imp_rows:
        rows.append([
            f'<span class="strong">{theme.esc(d["name"])}</span>',
            f"{imp.get('spot', d['market']['close']):,.0f}",
            f'<span class="strong">{imp["lower"]:,.0f} – {imp["upper"]:,.0f}</span>' if imp.get("lower") else "—",
            f"±{imp['move_points']:,.0f} (±{imp['move_pct'] * 100:.2f}%)",
            f"{pd.Timestamp(imp['expiry']):%a %d %b}",
            f"{imp['atm_strike']:,.0f}" if imp.get("atm_strike") else "—",
            (f"₹{imp['atm_call']:,.0f} + ₹{imp['atm_put']:,.0f} = ₹{imp['straddle']:,.0f}"
             if imp.get("straddle") else "—"),
            pct(imp["atm_iv"]),
        ])
    theme.table([("Index", ""), ("Last level", "num"), ("Expected range", "num"), ("Move", "num"),
                 ("Expiry", "num"), ("Options used (strike)", "num"), ("Call + put price", "num"),
                 ("Implied vol", "num")], rows)
    snap = imp_rows[0][1].get("snapshot_ts")
    theme.note("Options used: the call and put at the strike closest to the index on the nearest expiry. Their "
               "implied volatility (the yearly move they price in) is scaled to the time left: range = level ± "
               "level × implied vol × √(years to expiry). Call and put prices are valued at that implied volatility."
               + (f" Option prices as of {pd.Timestamp(snap):%d %b %Y, %H:%M} IST." if snap else ""))


# ============================================================
# READING THE NUMBERS
# ============================================================

theme.section("Reading the numbers")
n50 = syms.get("NIFTY_50", {})
v50 = n50.get("vol", {})
lines = []
if v50.get("vix") is not None:
    lines.append(
        f"India VIX is <b>{v50['vix'] * 100:.1f}</b>, {level_word(v50['vix_percentile_1y'])} for the past year, while "
        f"NIFTY's realized volatility is <b>{v50['rv21'] * 100:.1f}%</b>. Option buyers are paying "
        f"<b>{v50['vrp_now'] * 100:+.1f}</b> volatility points over recent movement, the "
        f"{ordinal(v50['vrp_percentile'])} of history. Historically implied exceeded what followed on "
        f"{v50['vrp_share_positive']:.0%} of days.")
if v50.get("garch_21d"):
    lines.append(f"GARCH puts NIFTY's expected volatility over the next 21 sessions at "
                 f"<b>{v50['garch_21d'] * 100:.1f}%</b> against a long-run level of "
                 f"{(v50.get('garch_long_run') or 0) * 100:.1f}%.")
risk = read_json("risk.json")
if risk.get("tail", {}).get("NIFTY_50"):
    f95 = risk["tail"]["NIFTY_50"]["fhs"]["0.95"]
    rn = risk["now"]
    lines.append(f"At today's volatility, NIFTY falls <b>{f95['var'] * 100:.1f}%</b> or more on about one day in "
                 f"twenty (₹{f95['var'] * 1e5:,.0f} per ₹1 lakh). Sectors are moving together at a "
                 f"{'low' if rn['avg_corr_pct'] < 0.35 else 'high' if rn['avg_corr_pct'] > 0.65 else 'normal'} level "
                 f"and market turbulence is at the {rn['turbulence_pct'] * 100:.0f}th percentile.")
for line in lines:
    theme.insight_html(line, "Note")


# ============================================================
# EVENTS THIS WEEK
# ============================================================

theme.section("Events this week", "Price-based signals that fired in the last five sessions (with what "
              "followed them historically over the next month), and the scheduled news coming up.")
ev_rows = []
for sym, d in syms.items():
    for e in d.get("events", []):
        sig = e.get("p_value") is not None and e["p_value"] < 0.05 and (e.get("n") or 0) >= 10
        ev_rows.append([
            f'<span class="strong">{theme.esc(e["name"])}</span>', theme.esc(d["name"]),
            pd.Timestamp(e["date"]).strftime("%a %d %b"), str(e.get("n") or "—"),
            theme.signed_html((e.get("mean") or np.nan) * 100, "{:+.2f}%"),
            pct(e.get("baseline_mean"), 2, True),
            f"{e['hit_rate']:.0%}" if e.get("hit_rate") is not None else "—",
            ('<span class="q-chip t-accent">Unusual</span>' if sig else '<span class="t-muted">Like a normal day</span>'),
        ])
st.markdown("##### Price-based signals")
if ev_rows:
    theme.table([("Event", ""), ("Index", ""), ("Date", ""), ("Past cases", "num"), ("Avg next month", "num"),
                 ("All days", "num"), ("Up share", "num"), ("History says", "")], ev_rows)
    theme.note("Full statistics, confidence intervals and paths are on the Event Lab page.")
else:
    theme.note("No tracked price signals fired in the last five sessions.")

st.markdown("##### Scheduled news in the next 7 days")
news = load_news(_mtime("news_calendar.json"))
nev = news.get("events", [])
if nev:
    IMPACT = {"High": '<span class="q-chip t-bear">High</span>', "Medium": '<span class="q-chip t-side">Medium</span>'}
    theme.table([("When", ""), ("Event", ""), ("Where", ""), ("Impact", ""), ("Why it matters", "")],
                [[f'{pd.Timestamp(e["date"]):%a %d %b}' + (f'<br><span class="dim">{e["time"]} IST</span>'
                                                            if e.get("time") else ""),
                  f'<span class="strong">{theme.esc(e["title"])}</span>', theme.esc(e["country"]),
                  IMPACT.get(e["impact"], theme.esc(e["impact"])), f'<span class="q-wrap">{theme.esc(e["why"])}</span>'] for e in nev])
else:
    theme.note("No major scheduled events found for the next 7 days.")
if news:
    theme.note(f"Sources: Forex Factory (global releases; next week's list appears from Sunday), RBI and Federal "
               f"Reserve meeting calendars, MoSPI release dates, NSE corporate calendar (NIFTY 50 results) and "
               f"the option chain (expiries). Updated {pd.Timestamp(news['built_at']):%d %b %H:%M} IST.")


# ============================================================
# NAVIGATION
# ============================================================

theme.section("Go deeper")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.page_link("views/volatility_edge.py", label="Volatility & Edge", icon=":material/ssid_chart:")
c2.page_link("views/event_lab.py", label="Event Lab", icon=":material/science:")
c3.page_link("views/market_regimes.py", label="Market Regimes", icon=":material/insights:")
c4.page_link("views/volatility_surface.py", label="Volatility Surface", icon=":material/stacked_line_chart:")
c5.page_link("views/market_states.py", label="Market States", icon=":material/waves:")
c6.page_link("views/risk_monitor.py", label="Risk Monitor", icon=":material/shield:")
