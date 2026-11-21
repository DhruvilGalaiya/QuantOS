"""
QuantOS | Event Lab page

"What happened after…": historical base rates for market events (52-week highs, losing streaks, VIX spikes,
gaps, drawdowns, Budget days) with bootstrap confidence intervals and a test against ordinary days.
Reads data/analytics/events_*.{json,parquet} (python -m quantos_platform.jobs analytics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)

from quantos_platform.analytics import INDICES, read_frame, read_json  # noqa: E402

HORIZON_LABEL = {1: "1 session", 5: "1 week (5)", 21: "1 month (21)", 63: "3 months (63)"}
ALPHA = 0.05


def pct(x, d: int = 2, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


@st.cache_data(ttl=600, show_spinner=False)
def load(symbol: str, _mtime: float) -> tuple[dict, pd.DataFrame]:
    s = symbol.lower()
    return read_json(f"events_summary_{s}.json"), read_frame(f"events_{s}.parquet")


def _mtime(symbol: str) -> float:
    from quantos_platform.core import settings
    p = settings.ANALYTICS_DIR / f"events_summary_{symbol.lower()}.json"
    return p.stat().st_mtime if p.exists() else 0.0


def verdict(st_: dict) -> str:
    p = st_.get("p_value")
    if not st_.get("n") or p is None:
        return '<span class="t-muted">—</span>'
    if st_["n"] < 10:
        return '<span class="t-muted">Too few</span>'
    if p < ALPHA:
        tone = "bull" if st_["excess"] > 0 else "bear"
        word = "Stronger" if st_["excess"] > 0 else "Weaker"
        return f'<span class="q-chip t-{tone}">{word}</span>'
    return '<span class="t-muted">Normal</span>'


# ============================================================
# SIDEBAR + DATA
# ============================================================

with st.sidebar:
    st.subheader("Event Lab")
    symbol = st.selectbox("Index", list(INDICES), format_func=INDICES.get, key="el_symbol")
    horizon = st.radio("Look ahead", list(HORIZON_LABEL), index=2, format_func=HORIZON_LABEL.get, key="el_h")
    if ui.refresh_button("analytics", "Rebuild analytics", key="el_refresh"):
        load.clear()

with st.spinner("Checking analytics…"):
    ui.refresh_on_launch("analytics")
ui.job_health("analytics")
M, rows = load(symbol, _mtime(symbol))
with st.sidebar:
    st.caption(ui.freshness_caption("analytics", "Analytics built from data", fallback=M.get("as_of")))

theme.hero(
    "Event Lab",
    "What actually happened after the market events traders talk about. Every result is compared with "
    "ordinary days and comes with a confidence interval, so you can tell a real tendency from noise.",
    [f"<b>{INDICES[symbol]}</b>", f"{M.get('sessions', 0):,} sessions since <b>{M.get('since', '—')[:4]}</b>",
     f"Look ahead <b>{HORIZON_LABEL[horizon]}</b>"],
)
if not M:
    theme.status("No analytics yet. Run: python -m quantos_platform.jobs analytics", "side")
    st.stop()

E = M["events"]
hkey = f"h{horizon}"

if M.get("recent"):
    items = ", ".join(f"<b>{theme.esc(r['name'])}</b> ({pd.Timestamp(r['date']):%d %b})" for r in M["recent"])
    theme.signal("Fired in the last five sessions", items, "accent")


# ============================================================
# LIBRARY
# ============================================================

theme.section("Event library", f"Average {INDICES[symbol]} return over the next {HORIZON_LABEL[horizon].lower()} "
              "after each event, against the average for all days.")
order = sorted(E.values(), key=lambda e: (e["stats"][hkey].get("p_value") is None,
                                          e["stats"][hkey].get("p_value") or 1))
lib_rows = []
for e in order:
    s = e["stats"][hkey]
    lib_rows.append([
        f'<span class="strong">{theme.esc(e["name"])}</span><br><span class="dim">{theme.esc(e["group"])}</span>',
        str(e["count"]),
        theme.signed_html((s.get("mean") or np.nan) * 100, "{:+.2f}%"),
        pct(s.get("baseline_mean")),
        f"{s['hit_rate']:.0%}" if s.get("n") else "—",
        f"{pct(s.get('ci_low'), 1)} / {pct(s.get('ci_high'), 1)}" if s.get("n") else "—",
        f"{s['p_value']:.2f}" if s.get("p_value") is not None else "—",
        verdict(s),
    ])
theme.table([("Event", ""), ("Times", "num"), ("Avg after", "num"), ("All days", "num"),
             ("Up share", "num"), ("95% CI", "num"), ("p", "num"), ("Verdict", "")], lib_rows)
n_sig = sum(1 for e in E.values() if (e["stats"][hkey].get("p_value") or 1) < ALPHA and e["stats"][hkey].get("n", 0) >= 10)
expected = len(E) * ALPHA
theme.insight_html(
    f"<b>{n_sig}</b> of {len(E)} events differ from an ordinary day at the 5% level over this horizon. With "
    f"{len(E)} tests, about <b>{expected:.1f}</b> would pass by luck alone, so treat single hits with caution. "
    "Most famous \"signals\" leave the market's behaviour indistinguishable from normal: the edge is usually "
    "in risk (how wide the outcomes are), not in direction.")


# ============================================================
# ONE EVENT IN DEPTH
# ============================================================

theme.section("One event in depth")
names = {e["id"]: e["name"] for e in order}
eid = st.selectbox("Event", list(names), format_func=names.get, key="el_event")
e = E[eid]
s = e["stats"][hkey]
theme.note(e["description"] + (f" Occurrences closer than {e['min_gap']} sessions are counted once."
                               if e["min_gap"] > 1 else ""))
if not s.get("n"):
    theme.status("This event has not occurred in the data, or not long enough ago to measure.", "side")
    st.stop()

theme.stats([
    theme.stat("Occurrences", f"{s['n']}", sub=f"last {pd.Timestamp(e['last']):%d %b %Y}" if e.get("last") else None),
    theme.stat(f"Average after {HORIZON_LABEL[horizon].lower()}", pct(s["mean"]),
               tone="bull" if s["mean"] > 0 else "bear", sub=f"all days {pct(s['baseline_mean'])}"),
    theme.stat("Median", pct(s["median"]), sub=f"up {s['hit_rate']:.0%} of the time (all days "
                                              f"{s['baseline_hit_rate']:.0%})"),
    theme.stat("Worst / best", f"{s['worst'] * 100:+.0f}% / {s['best'] * 100:+.0f}%", sub="range of outcomes"),
    theme.stat("p-value vs normal days", f"{s['p_value']:.2f}", sub="below 0.05 = unusual",
               help="Share of random samples of ordinary days (same size) whose average is at least as far from "
                    "the all-days average as this event's average."),
])

path = e["path"]
off = np.array(path["offsets"])
with theme.card("path"):
    fig = go.Figure()
    fig.add_scatter(x=off, y=np.array(path["ci_high"], dtype=float) * 100, line=dict(width=0), showlegend=False,
                    hoverinfo="skip")
    ci_hi = np.array(path["ci_high"], dtype=float) * 100
    fig.add_scatter(x=off, y=np.array(path["ci_low"], dtype=float) * 100, fill="tonexty",
                    fillcolor=theme.alpha(theme.ACCENT, 0.13), line=dict(width=0), name="95% confidence band",
                    customdata=ci_hi, hovertemplate="Likely range: %{y:+.2f}% to %{customdata:+.2f}%<extra></extra>")
    fig.add_scatter(x=off, y=np.array(path["baseline"], dtype=float) * 100, name="Typical drift (all days)",
                    line=dict(color=theme.FAINT, dash="dot", width=1.5),
                    hovertemplate="Typical day: %{y:+.2f}%<extra></extra>")
    fig.add_scatter(x=off, y=np.array(path["mean"], dtype=float) * 100, name="Average path after the event",
                    line=dict(color=theme.ACCENT, width=2.4),
                    hovertemplate="After this event: <b>%{y:+.2f}%</b><extra></extra>")
    fig.add_vline(x=0, line=dict(color=theme.WARM, width=1.2), annotation_text="event close",
                  annotation_font_color=theme.WARM)
    fig.add_vline(x=horizon, line=dict(color=theme.LINE, dash="dash"))
    fig.update_layout(height=360, hovermode="x unified", xaxis=dict(title="Sessions from the event",
                                                                    unifiedhovertitle=dict(text="Session %{x} (0 = event day)")),
                      yaxis_title="Return vs event-day close, %", legend=dict(orientation="h", y=1.08, x=0))
    theme.chart(fig, key="path")

c1, c2 = st.columns([1.15, 1])
with c1:
    ev = rows[rows["event"] == eid].sort_values("date", ascending=False)
    col = f"fwd_{horizon}"
    with theme.card("dist"):
        fig = go.Figure()
        fig.add_histogram(x=ev[col].dropna() * 100, nbinsx=25, name="After this event",
                          marker_color=theme.ACCENT, opacity=0.85, histnorm="percent")
        fig.add_vline(x=s["mean"] * 100, line=dict(color=theme.ACCENT, width=2))
        fig.add_vline(x=s["baseline_mean"] * 100, line=dict(color=theme.FAINT, dash="dot", width=2))
        fig.update_layout(height=280, xaxis_title=f"Return over {HORIZON_LABEL[horizon].lower()}, %",
                          yaxis_title="% of occurrences", showlegend=False)
        theme.chart(fig, key="dist")
    theme.note("Solid line: this event's average. Dotted line: the average of all days.")
with c2:
    show = ev.head(12)
    theme.table([("Date", ""), ("+1", "num"), ("+5", "num"), ("+21", "num"), ("+63", "num")],
                [[pd.Timestamp(r.date).strftime("%d %b %Y"),
                  *[theme.signed_html(getattr(r, f'fwd_{h}') * 100, "{:+.1f}%") for h in (1, 5, 21, 63)]]
                 for r in show.itertuples()])
    theme.note(f"Most recent {len(show)} of {len(ev)} occurrences. Blank = not enough sessions yet.")

lo, hi = s["ci_low"], s["ci_high"]
unusual = s["n"] >= 10 and s["p_value"] < ALPHA
theme.insight_html(
    f"After <b>{theme.esc(e['name'].lower())}</b>, {INDICES[symbol]} returned <b>{pct(s['mean'])}</b> on average "
    f"over the next {HORIZON_LABEL[horizon].lower()} (95% CI {pct(lo, 1)} to {pct(hi, 1)}), versus "
    f"{pct(s['baseline_mean'])} for all days. "
    + (f"Random samples of ordinary days rarely land this far from normal (p = {s['p_value']:.2f}), so the "
       f"difference is unusual, though {len(E)} events are tested at once." if unusual else
       f"Random samples of ordinary days often land this far from normal (p = {s['p_value']:.2f}), so history "
       "does not show a reliable tendency here.")
    + f" Outcomes ranged from {pct(s['worst'], 1)} to {pct(s['best'], 1)}.")

theme.method(
    "- **Event day**: defined only with information available at that day's close. Returns run from that close "
    "to the close *h* sessions later, so there is no look-ahead.\n"
    "- **De-clustering**: repeated triggers within the event's minimum gap count once, so a single episode "
    "is not counted many times.\n"
    f"- **95% CI**: bootstrap of the event average ({M['n_boot']:,} resamples).\n"
    "- **p-value**: share of equally sized random samples of all days whose average is at least as far from "
    "the all-days average (two-sided).\n"
    "- **Caveats**: overlapping windows and changing market regimes make outcomes less independent than they "
    "look; many events are tested at once (see the library note); small samples have wide intervals. "
    "This describes history; it is not a trading signal.")
