"""
QuantOS dashboard design system ("Research desk": light, editorial).

One place for colour tokens, the Plotly template, global CSS (including motion) and the
small HTML components the pages share. Colours mirror .streamlit/config.toml.

    import theme
    theme.apply()                      # once per run, in the page router
    theme.hero("Market regimes", "…")  # page header
    theme.chart(fig)                   # st.plotly_chart with the QuantOS template
"""

from __future__ import annotations

import datetime
import html
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── tokens ────────────────────────────────────────────────────────────────────
BG = "#F5F3EE"          # warm paper
PANEL = "#FFFFFF"       # cards
RAISED = "#FAF8F4"      # table heads, hover
LINE = "#E4E0D6"
LINE_SOFT = "#EEEBE4"
TEXT = "#191B22"        # ink
MUTED = "#5C6270"
FAINT = "#8C919C"
ACCENT = "#3346D3"      # cobalt
WARM = "#D9653B"        # terracotta, secondary highlight
BULL = "#13875A"
SIDE = "#B7791F"
BEAR = "#CF404D"
GRID = "rgba(25,27,34,0.07)"
FONT = "Manrope, -apple-system, 'Segoe UI', system-ui, sans-serif"
SERIF = "'Instrument Serif', 'Iowan Old Style', Georgia, serif"
MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace"

# categorical series, tuned for the paper ground (accent first)
SERIES = ["#3346D3", "#E07A3C", "#13875A", "#C2417E", "#1F92B8", "#B8962E", "#7A55C9", "#CF404D"]
# single-hue sequential scale: pale at the low end, deep cobalt at the high end
SEQ = [[0.0, "#CDD3F7"], [0.35, "#8F9CEB"], [0.7, "#4C5FDC"], [1.0, "#1A2380"]]

TONES = {"bull": BULL, "side": SIDE, "bear": BEAR, "accent": ACCENT, "muted": MUTED}
REGIME_TONE = {"BULL": "bull", "SIDE": "side", "BEAR": "bear",
               # India VIX: calm is good, high volatility is stress
               "LOW": "bull", "MID": "side", "HIGH": "bear"}
REGIME_WORD = {"BULL": "Bull", "SIDE": "Sideways", "BEAR": "Bear",
               "LOW": "Low vol", "MID": "Mid vol", "HIGH": "High vol"}


def alpha(hex_color: str, a: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,a)' for fills and soft lines."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


ASSETS = Path(__file__).resolve().parent / "assets"
PLOT_CONFIG = {"displaylogo": False, "displayModeBar": "hover",
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def esc(x) -> str:
    return html.escape(str(x))


def tone_of(regime: str) -> str:
    return REGIME_TONE.get(str(regime).upper(), "muted")


def word_of(regime: str) -> str:
    r = str(regime).upper()
    return REGIME_WORD.get(r, r.title())


# ── plotly ────────────────────────────────────────────────────────────────────
def _axis(**kw) -> dict:
    return dict(gridcolor=GRID, linecolor=LINE, zerolinecolor=LINE, tickcolor=LINE, automargin=True,
                ticks="", showline=False, title=dict(font=dict(color=FAINT, size=12), standoff=10),
                tickfont=dict(color=MUTED, size=11.5), **kw)


def _scene_axis() -> dict:
    return dict(backgroundcolor="rgba(245,243,238,0.75)", showbackground=True, gridcolor="rgba(25,27,34,0.10)",
                zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED, size=10.5),
                title=dict(font=dict(color=MUTED, size=11.5)))


TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=12.5, color=MUTED),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=SERIES,
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=_axis(),
    yaxis=_axis(),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=FONT, color=TEXT, size=12.5),
                    align="left"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12), title=dict(text="")),
    title=dict(font=dict(family=SERIF, color=TEXT, size=18), x=0, xanchor="left"),
    scene=dict(xaxis=_scene_axis(), yaxis=_scene_axis(), zaxis=_scene_axis()),
    coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED)))),
    bargap=0.32,
))
TEMPLATE.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
TEMPLATE.data.surface = [go.Surface(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED)))]


def _axis_title(fig: go.Figure, ref: str | None, letter: str) -> str:
    name = f"{letter}axis{(ref or letter)[1:]}"
    try:
        return fig.layout[name].title.text or ""
    except (KeyError, AttributeError):
        return ""


def _unit(title: str) -> tuple[str, str]:
    """Number format + suffix for a hover value, read from the axis title ('… %' → '1.86%')."""
    t = title.lower()
    if "%" in t or "percent" in t:
        return ",.2f", "%"
    if "pts" in t or "points" in t:
        return ",.1f", " pts"
    return ",.2f", ""


def _is_date(values) -> bool:
    if values is None or len(values) == 0:
        return False
    v = values[0]
    return isinstance(v, (pd.Timestamp, np.datetime64, datetime.date)) or (
        isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-")


def _auto_hover(fig: go.Figure) -> None:
    """Readable hover labels for traces that did not set their own: trace name, axis names, rounded numbers
    with units, instead of raw coordinates like (15 - 20, 13) or 0.7559387."""
    unified = str(fig.layout.hovermode or "").endswith("unified")
    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl", "bar", "histogram"):
            continue
        if tr.hovertemplate is not None or tr.hoverinfo == "skip":
            continue
        horizontal = getattr(tr, "orientation", None) == "h"
        xt, yt = _axis_title(fig, tr.xaxis, "x"), _axis_title(fig, tr.yaxis, "y")
        if tr.type == "histogram":
            norm = getattr(tr, "histnorm", None) or ""
            count = "%{y:.1f}%" if norm == "percent" else "%{y:,.0f}"
            tr.hovertemplate = f"{xt or 'Range'}: %{{x}}<br>{yt or 'Count'}: {count}<extra></extra>"
            continue
        vt, val_axis, cat, cat_title = (xt, "x", "y", yt) if horizontal else (yt, "y", "x", xt)
        fmt, suf = _unit(vt)
        cat_vals = tr.y if horizontal else tr.x
        cat_fmt = "|%d %b %Y" if _is_date(cat_vals) else ""
        if not cat_fmt and cat_vals is not None and len(cat_vals) and isinstance(cat_vals[0], (int, float, np.number)):
            cfmt, csuf = _unit(cat_title)
            cat_fmt = f":{cfmt}"
        else:
            csuf = ""
        name = tr.name or vt or "Value"
        value = f"%{{{val_axis}:{fmt}}}{suf}"
        if unified:
            tr.hovertemplate = f"{name}: <b>{value}</b><extra></extra>"
        else:
            head = f"{cat_title}: " if cat_title and not cat_fmt.startswith("|") else ""
            tr.hovertemplate = (f"<b>{name}</b><br>{head}%{{{cat}{cat_fmt}}}{csuf}<br>"
                                f"{vt or 'Value'}: {value}<extra></extra>")
    for ax in ("xaxis", "yaxis"):
        a = fig.layout[ax]
        if a.hoverformat is None and any(_is_date(getattr(t, ax[0], None)) for t in fig.data):
            a.hoverformat = "%a %d %b %Y"


def chart(fig: go.Figure, *, config: dict | None = None, key: str | None = None) -> None:
    """Render a Plotly figure with the QuantOS template (Streamlit's own chart theme off)."""
    _auto_hover(fig)
    # Streamlit fills unset backgrounds with its secondary colour; keep charts on the card.
    if fig.layout.plot_bgcolor is None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    if fig.layout.paper_bgcolor is None:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", theme=None, config={**PLOT_CONFIG, **(config or {})}, key=key)


def card(name: str, parent=st):
    """Bordered container styled as a white card (`with theme.card("timeline"):`). Names must be unique per page."""
    return parent.container(border=True, key=f"qcard_{name}")


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
:root {
  --q-bg:#F5F3EE; --q-panel:#FFFFFF; --q-raised:#FAF8F4; --q-line:#E4E0D6; --q-line-soft:#EEEBE4;
  --q-text:#191B22; --q-muted:#5C6270; --q-faint:#8C919C;
  --q-accent:#3346D3; --q-accent-soft:rgba(51,70,211,.07); --q-warm:#D9653B;
  --q-bull:#13875A; --q-side:#B7791F; --q-bear:#CF404D;
  --q-ease:cubic-bezier(.23,1,.32,1);
  --q-font:Manrope,-apple-system,'Segoe UI',system-ui,sans-serif;
  --q-serif:'Instrument Serif','Iowan Old Style',Georgia,serif;
  --q-mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,monospace;
  --q-shadow:0 1px 2px rgba(25,27,34,.04), 0 6px 20px -10px rgba(25,27,34,.10);
  --q-shadow-hi:0 1px 2px rgba(25,27,34,.05), 0 14px 32px -14px rgba(25,27,34,.18);
}

/* ── app chrome ───────────────────────────────────────── */
[data-testid="stApp"] {
  background:
    radial-gradient(900px 380px at 88% -6%, rgba(51,70,211,.07), transparent 62%),
    radial-gradient(700px 320px at -4% 4%, rgba(217,101,59,.05), transparent 60%),
    var(--q-bg);
}
[data-testid="stMainBlockContainer"] { padding-top: 4.6rem; padding-bottom: 5rem; max-width: 1360px; }
[data-testid="stHeader"] {
  background: color-mix(in srgb, var(--q-bg) 78%, transparent);
  backdrop-filter: saturate(160%) blur(14px); -webkit-backdrop-filter: saturate(160%) blur(14px);
  border-bottom: 1px solid var(--q-line);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
[data-testid="stAppViewContainer"] { font-variant-numeric: tabular-nums; }
h1, h2, h3, h4 { font-weight: 400 !important; letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--q-faint); font-size: .82rem; line-height: 1.55; }
hr { border-color: var(--q-line) !important; }
::selection { background: rgba(51,70,211,.16); }

/* page change: the main column settles in once */
[data-testid="stMainBlockContainer"] > div { animation: q-fade .5s var(--q-ease) both; }

/* bordered containers become cards */
[class*="st-key-qcard_"] {
  background: var(--q-panel); box-shadow: var(--q-shadow); border-color: var(--q-line) !important; border-radius: 16px !important;
}

/* metrics that remain native */
[data-testid="stMetric"] { background: var(--q-panel); border: 1px solid var(--q-line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--q-shadow); }
[data-testid="stMetricLabel"] p { color: var(--q-muted); font-size: .8rem; }
[data-testid="stMetricValue"] { letter-spacing: -0.015em; }

/* controls: quick, tactile feedback */
[data-testid^="stBaseButton"] {
  transition: transform .16s var(--q-ease), background-color .18s ease, border-color .18s ease, color .18s ease, box-shadow .18s ease;
}
[data-testid^="stBaseButton"]:active { transform: scale(.97); }
[data-testid="stExpander"] details { background: var(--q-panel); border-color: var(--q-line); border-radius: 14px; box-shadow: var(--q-shadow); }
[data-testid="stExpander"] summary { transition: color .18s ease; }
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover svg { color: var(--q-accent); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; box-shadow: var(--q-shadow); }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div { transition: border-color .18s ease, box-shadow .18s ease; }
:focus-visible { outline: 2px solid var(--q-accent) !important; outline-offset: 2px; }

/* ── components ───────────────────────────────────────── */
.q, .q * { box-sizing: border-box; }
.q { font-family: var(--q-font); color: var(--q-text); font-variant-numeric: tabular-nums; }
.t-bull { color: var(--q-bull); } .t-side { color: var(--q-side); } .t-bear { color: var(--q-bear); }
.t-accent { color: var(--q-accent); } .t-muted { color: var(--q-muted); } .t-warm { color: var(--q-warm); }

/* eyebrow: small mono label that names the context above a title */
.q-eyebrow { display: inline-flex; align-items: center; gap: 8px; font-family: var(--q-mono); font-size: .72rem;
  font-weight: 500; letter-spacing: .02em; color: var(--q-accent); }
.q-eyebrow::before { content: ""; width: 18px; height: 1px; background: currentColor; }

/* page hero */
.q-hero { padding: 6px 0 28px; }
.q-hero h1 { margin: 10px 0 0; padding: 0; font-family: var(--q-serif); font-weight: 400;
  font-size: clamp(2.4rem, 1.6rem + 2.6vw, 3.6rem); line-height: 1.02; letter-spacing: -0.015em; color: var(--q-text); }
.q-lede { margin: 12px 0 0; max-width: 68ch; color: var(--q-muted); font-size: 1.04rem; line-height: 1.6; }
.q-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.q-pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border-radius: 999px;
  border: 1px solid var(--q-line); background: var(--q-panel); color: var(--q-muted); font-size: .8rem; box-shadow: 0 1px 2px rgba(25,27,34,.04); }
.q-pill b { color: var(--q-text); font-weight: 650; }

/* section heading: hairline rule, serif title, sans note */
.q-section { margin: 40px 0 14px; padding-top: 16px; border-top: 1px solid var(--q-line); }
.q-section h2 { margin: 0; padding: 0; font-family: var(--q-serif); font-size: 1.85rem; font-weight: 400;
  line-height: 1.1; letter-spacing: -0.01em; color: var(--q-text); }
.q-section p { margin: 6px 0 0; max-width: 78ch; color: var(--q-muted); font-size: .92rem; line-height: 1.55; }

/* stat cards */
.q-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 12px; }
.q-stat { position: relative; background: var(--q-panel); border: 1px solid var(--q-line); border-radius: 14px;
  padding: 15px 18px 16px; min-width: 0; box-shadow: var(--q-shadow); }
.q-stat-label { color: var(--q-muted); font-size: .8rem; font-weight: 500; line-height: 1.3; }
.q-stat-label[title] { cursor: help; text-decoration: underline dotted color-mix(in srgb, var(--q-muted) 50%, transparent); text-underline-offset: 3px; }
.q-stat-value { margin-top: 8px; font-size: 1.6rem; font-weight: 650; letter-spacing: -0.025em; line-height: 1.1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.q-stat-sub { margin-top: 6px; color: var(--q-faint); font-family: var(--q-mono); font-size: .72rem; line-height: 1.4; }

/* panel */
.q-card { background: var(--q-panel); border: 1px solid var(--q-line); border-radius: 16px; padding: 20px 22px; box-shadow: var(--q-shadow); }
.q-card + .q-note, .q-stats + .q-note { margin-top: 10px; }
.q-note { color: var(--q-faint); font-size: .84rem; line-height: 1.6; max-width: 92ch; margin: 10px 0 0; }

/* regime chip */
.q-chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px 2px 8px; border-radius: 999px;
  font-size: .78rem; font-weight: 650; line-height: 1.65; white-space: nowrap;
  background: color-mix(in srgb, currentColor 10%, white); }
.q-chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

/* tables */
.q-table-wrap { background: var(--q-panel); border: 1px solid var(--q-line); border-radius: 16px; overflow-x: auto; box-shadow: var(--q-shadow); }
.q-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.q-table th { text-align: left; padding: 11px 16px; color: var(--q-faint); background: var(--q-raised);
  font-family: var(--q-mono); font-weight: 500; font-size: .7rem; letter-spacing: .02em;
  border-bottom: 1px solid var(--q-line); white-space: nowrap; }
.q-table td { padding: 13px 16px; border-bottom: 1px solid var(--q-line-soft); white-space: nowrap; vertical-align: middle; }
.q-table tbody tr:last-child td { border-bottom: 0; }
.q-table tbody tr { transition: background-color .18s ease; }
.q-table tbody tr:hover { background: var(--q-raised); }
.q-table tr.is-current td { background: var(--q-accent-soft); }
.q-table tr.is-current td:first-child { box-shadow: inset 3px 0 0 var(--q-accent); }
.q-table .num { text-align: right; }
.q-table .strong { font-weight: 650; color: var(--q-text); }
.q-table .dim { color: var(--q-muted); }
.q-mini { display: inline-flex; align-items: center; gap: 10px; min-width: 104px; }
.q-mini .strong { min-width: 2.6em; text-align: right; }
.q-mini-track { flex: 1; height: 6px; border-radius: 3px; background: var(--q-line-soft); overflow: hidden; }
.q-mini-track > i { display: block; height: 100%; border-radius: inherit; background: currentColor;
  transform-origin: left; animation: q-grow .9s var(--q-ease) both; }

/* meter (persistence) */
.q-meter { margin-top: 4px; }
.q-meter-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.q-meter-value { font-family: var(--q-serif); font-size: 3.4rem; font-weight: 400; letter-spacing: -0.02em; line-height: 1; }
.q-meter-value small { font-family: var(--q-mono); font-size: .8rem; color: var(--q-faint); letter-spacing: 0; margin-left: 4px; }
.q-meter-level { font-size: .85rem; font-weight: 650; }
.q-meter-track { position: relative; height: 10px; margin: 18px 0 8px; border-radius: 5px; background: var(--q-line-soft); overflow: hidden; }
.q-meter-track > i { position: absolute; inset: 0 auto 0 0; border-radius: inherit;
  background: linear-gradient(90deg, #A3AEF0, var(--q-accent));
  transform-origin: left; animation: q-grow 1.1s var(--q-ease) .1s both; }
.q-meter-track > b { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--q-panel); }
.q-meter-scale { position: relative; height: 14px; color: var(--q-faint); font-family: var(--q-mono); font-size: .68rem; }
.q-meter-scale span { position: absolute; transform: translateX(-50%); }
.q-meter-scale span:first-child { transform: none; } .q-meter-scale span:last-child { transform: translateX(-100%); }

/* regime hero (market page) */
.q-regime { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px 32px; align-items: end; padding: 6px 0 24px; }
.q-regime-word { margin-top: 6px; font-family: var(--q-serif); font-size: clamp(3.6rem, 2.2rem + 5vw, 6.6rem); font-weight: 400;
  letter-spacing: -0.02em; line-height: .92; }
.q-regime-meta { margin-top: 14px; color: var(--q-muted); font-size: .96rem; }
.q-regime-meta b { color: var(--q-text); font-weight: 650; }
.q-price { text-align: right; }
.q-price-label { display: inline-flex; align-items: center; gap: 8px; color: var(--q-faint); font-family: var(--q-mono); font-size: .72rem; }
.q-price-value { margin-top: 6px; font-size: clamp(1.8rem, 1.3rem + 1.3vw, 2.5rem); font-weight: 650; letter-spacing: -0.03em; line-height: 1; }
.q-price-change { margin-top: 8px; font-size: .92rem; font-weight: 650; }
.q-live { position: relative; width: 7px; height: 7px; border-radius: 50%; background: var(--q-bull); flex: none; }
.q-live::after { content: ""; position: absolute; inset: 0; border-radius: 50%; background: var(--q-bull);
  animation: q-pulse 2.2s cubic-bezier(.4,0,.6,1) infinite; }
.q-ribbon { padding: 18px 20px 16px; background: var(--q-panel); border: 1px solid var(--q-line); border-radius: 16px; box-shadow: var(--q-shadow); }
.q-ribbon-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 16px; font-size: .88rem; color: var(--q-muted); }
.q-ribbon-head b { color: var(--q-text); font-weight: 650; }
.q-ribbon-bar { display: flex; gap: 3px; height: 12px; margin: 14px 0 11px; border-radius: 6px; overflow: hidden;
  transform-origin: left; animation: q-grow 1s var(--q-ease) .15s both; }
.q-ribbon-bar > i { display: block; min-width: 3px; background: currentColor; opacity: .9; }
.q-ribbon-bar > i.is-pick { opacity: 1; }
.q-ribbon-legend { display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: .84rem; }
.q-ribbon-legend span { display: inline-flex; align-items: center; gap: 7px; color: var(--q-muted); }
.q-ribbon-legend span::before { content: ""; width: 8px; height: 8px; border-radius: 2px; background: currentColor; }
.q-ribbon-legend b { color: var(--q-text); font-weight: 650; }

/* signal card (expiry playbook) */
.q-signal { position: relative; overflow: hidden; display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 6px 16px; align-items: start;
  padding: 22px 24px 22px 26px; background: var(--q-panel); border: 1px solid var(--q-line); border-radius: 16px; box-shadow: var(--q-shadow); }
.q-signal::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: currentColor; }
.q-signal-dot { grid-row: span 2; width: 12px; height: 12px; margin-top: 12px; border-radius: 50%; background: currentColor;
  box-shadow: 0 0 0 5px color-mix(in srgb, currentColor 14%, transparent); }
.q-signal-title { font-family: var(--q-serif); font-size: 1.9rem; font-weight: 400; line-height: 1.15; letter-spacing: -0.01em; color: var(--q-text); }
.q-signal-body { color: var(--q-muted); font-size: .95rem; line-height: 1.6; }
.q-signal-body b { color: var(--q-text); font-weight: 650; }

/* status line */
.q-status { display: flex; align-items: center; gap: 10px; padding: 11px 14px; border-radius: 12px; font-size: .88rem;
  background: color-mix(in srgb, currentColor 7%, white); border: 1px solid color-mix(in srgb, currentColor 22%, transparent); }
.q-status span { color: var(--q-text); }
.q-status::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex: none; }

/* ── motion ───────────────────────────────────────────── */
@keyframes q-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes q-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes q-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes q-pulse { 0% { transform: scale(1); opacity: .55; } 70%, 100% { transform: scale(2.8); opacity: 0; } }
.q-enter > * { animation: q-rise .7s var(--q-ease) both; }
.q-enter > *:nth-child(2) { animation-delay: .07s; }
.q-enter > *:nth-child(3) { animation-delay: .14s; }
.q-enter > *:nth-child(4) { animation-delay: .21s; }
.q-enter > *:nth-child(n+5) { animation-delay: .28s; }
.q-stats.q-stagger > .q-stat { animation: q-rise .7s var(--q-ease) both; }
.q-stats.q-stagger > .q-stat:nth-child(2) { animation-delay: .05s; }
.q-stats.q-stagger > .q-stat:nth-child(3) { animation-delay: .10s; }
.q-stats.q-stagger > .q-stat:nth-child(4) { animation-delay: .15s; }
.q-stats.q-stagger > .q-stat:nth-child(5) { animation-delay: .20s; }
.q-stats.q-stagger > .q-stat:nth-child(n+6) { animation-delay: .25s; }

@media (max-width: 720px) {
  [data-testid="stMainBlockContainer"] { padding-top: 4.6rem; }
  .q-regime { grid-template-columns: minmax(0, 1fr); }
  .q-price { text-align: left; }
  .q-stats { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 10px; }
  .q-stat { padding: 13px 14px; }
  .q-stat-value { font-size: 1.3rem; }
  .q-section h2 { font-size: 1.6rem; }
}
.q-insight { display: flex; gap: 10px; align-items: flex-start; margin: 12px 0 0; padding: 11px 14px;
  background: var(--q-accent-soft); border-left: 3px solid var(--q-accent); border-radius: 0 10px 10px 0;
  color: var(--q-text); font-size: .9rem; line-height: 1.55; max-width: 100ch; }
.q-insight b.q-insight-k { color: var(--q-accent); font-weight: 700; white-space: nowrap; font-size: .76rem;
  letter-spacing: .06em; text-transform: uppercase; padding-top: 2px; }
.q-kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; font-size: .9rem; margin: 4px 0 0; }
.q-kv dt { color: var(--q-faint); } .q-kv dd { margin: 0; color: var(--q-text); font-weight: 600; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def apply() -> None:
    """Register the Plotly template, inject global CSS and the logo. Call once per run."""
    pio.templates["quantos"] = TEMPLATE
    pio.templates.default = "quantos"
    st.html(CSS)
    logo = ASSETS / "logo.svg"
    if logo.exists():
        st.logo(str(logo), size="large", icon_image=str(ASSETS / "mark.svg"))


# ── components ────────────────────────────────────────────────────────────────
def _render(markup: str) -> None:
    st.html(f'<div class="q">{markup}</div>')


def hero(title: str, lede: str | None = None, pills: Iterable[str] = (), eyebrow: str | None = None) -> None:
    """Page title, one plain sentence about what the page shows, optional status pills (HTML)."""
    pill_html = "".join(f'<span class="q-pill">{p}</span>' for p in pills)
    _render(
        '<header class="q-hero q-enter">'
        + (f'<div class="q-eyebrow">{esc(eyebrow)}</div>' if eyebrow else "")
        + f"<h1>{esc(title)}</h1>"
        + (f'<p class="q-lede">{esc(lede)}</p>' if lede else "")
        + (f'<div class="q-pills">{pill_html}</div>' if pill_html else "")
        + "</header>"
    )


def section(title: str, note: str | None = None) -> None:
    _render(f'<div class="q-section"><h2>{esc(title)}</h2>' + (f"<p>{esc(note)}</p>" if note else "") + "</div>")


def note(text: str) -> None:
    """Small explanatory text under a chart or table (plain text, escaped)."""
    _render(f'<p class="q-note">{esc(text)}</p>')


def insight(text: str, label: str = "What this means") -> None:
    """The plain-language reading of a chart or number (text is escaped; use <b> via insight_html)."""
    insight_html(esc(text), label)


def insight_html(body_html: str, label: str = "What this means") -> None:
    _render(f'<div class="q-insight"><b class="q-insight-k">{esc(label)}</b><span>{body_html}</span></div>')


def method(markdown: str, title: str = "Method and caveats") -> None:
    """Collapsed methodology note (Markdown, may include LaTeX)."""
    with st.expander(title, icon=":material/functions:"):
        st.markdown(markdown)


def stat(label: str, value: str, sub: str | None = None, tone: str | None = None, help: str | None = None) -> dict:
    return {"label": label, "value": value, "sub": sub, "tone": tone, "help": help}


def stats(items: Sequence[dict], enter: bool = True, cols: int | None = None) -> None:
    cells = []
    for it in items:
        title = f' title="{esc(it["help"])}"' if it.get("help") else ""
        tone = f' t-{it["tone"]}' if it.get("tone") else ""
        cells.append(
            '<div class="q-stat">'
            f'<div class="q-stat-label"{title}>{esc(it["label"])}</div>'
            f'<div class="q-stat-value{tone}">{esc(it["value"])}</div>'
            + (f'<div class="q-stat-sub">{esc(it["sub"])}</div>' if it.get("sub") else "")
            + "</div>"
        )
    grid = f' style="grid-template-columns:repeat({cols},minmax(0,1fr))"' if cols else ""
    _render(f'<div class="q-stats{" q-stagger" if enter else ""}"{grid}>{"".join(cells)}</div>')


def chip(regime: str, text: str | None = None) -> str:
    return f'<span class="q-chip t-{tone_of(regime)}">{esc(text or word_of(regime))}</span>'


def signed_html(value: float, fmt: str = "{:+.2f}%", invert: bool = False) -> str:
    """Coloured signed number (green up / red down; `invert` for VIX-like series)."""
    if value is None or value != value:  # NaN
        return '<span class="t-muted">—</span>'
    good = value > 0 if not invert else value < 0
    tone = "t-muted" if value == 0 else ("t-bull" if good else "t-bear")
    return f'<span class="{tone}">{esc(fmt.format(value))}</span>'


def rupees(value: float, signed: bool = True) -> str:
    """'+₹1,013' / '−₹95' (or '₹1,013' unsigned)."""
    if value is None or value != value:
        return "—"
    sign = ("+" if value > 0 else "−" if value < 0 else "") if signed else ("−" if value < 0 else "")
    return f"{sign}₹{abs(value):,.0f}"


def rupee_html(value: float) -> str:
    """Coloured signed rupee amount."""
    if value is None or value != value:
        return '<span class="t-muted">—</span>'
    tone = "t-muted" if round(value) == 0 else ("t-bull" if value > 0 else "t-bear")
    return f'<span class="{tone}">{rupees(value)}</span>'


def mini_bar(fraction: float, label: str, tone: str = "accent") -> str:
    f = 0.0 if fraction != fraction else max(0.0, min(1.0, float(fraction)))
    return (f'<span class="q-mini t-{tone}"><span class="q-mini-track"><i style="width:{f * 100:.1f}%"></i></span>'
            f'<span class="strong" style="color:var(--q-text)">{esc(label)}</span></span>')


def table(headers: Sequence[tuple[str, str]], rows: Sequence[Sequence[str]], current: int | None = None) -> None:
    """HTML table. headers = [(label, "num"|"")]; rows hold pre-rendered (escaped) cell HTML."""
    head = "".join(f'<th class="{cls}">{esc(h)}</th>' for h, cls in headers)
    body = []
    for i, row in enumerate(rows):
        tds = "".join(f'<td class="{headers[j][1]}">{cell}</td>' for j, cell in enumerate(row))
        cls = ' class="is-current"' if i == current else ""
        body.append(f"<tr{cls}>{tds}</tr>")
    _render(f'<div class="q-table-wrap"><table class="q-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def meter(value: float, level: str, marks: Sequence[float] = (50, 75), tone: str = "accent") -> None:
    v = 0.0 if value != value else max(0.0, min(100.0, float(value)))
    ticks = "".join(f'<b style="left:{m}%"></b>' for m in marks)
    scale = "".join(f'<span style="left:{m}%">{m:g}</span>' for m in (0, *marks, 100))
    shown = f"{v:.0f}" if value == value else "—"
    _render(
        '<div class="q-card q-meter">'
        f'<div class="q-meter-head"><div class="q-meter-value">{shown}<small>/100</small></div>'
        f'<div class="q-meter-level t-{tone}">{esc(level.title())}</div></div>'
        f'<div class="q-meter-track"><i style="width:{v:.1f}%"></i>{ticks}</div>'
        f'<div class="q-meter-scale">{scale}</div>'
        "</div>"
    )


def status(text: str, tone: str = "bull") -> None:
    _render(f'<div class="q-status t-{tone}"><span>{esc(text)}</span></div>')


def signal(title: str, body_html: str, tone: str) -> None:
    _render(f'<div class="q-enter"><div class="q-signal t-{tone}"><span class="q-signal-dot"></span>'
            f'<div class="q-signal-title">{esc(title)}</div><div class="q-signal-body">{body_html}</div></div></div>')


def regime_hero(*, instrument: str, regime: str, meta_html: str, price_label: str, price: str,
                change_html: str = "", live: bool = False, ribbon_head: str = "",
                probs: dict[str, float] | None = None, pick: str | None = None) -> None:
    """The market page's signature header: the regime word, price, and the next-session probability ribbon."""
    live_dot = '<span class="q-live" aria-hidden="true"></span>' if live else ""
    ribbon = ""
    if probs:
        segs, legend = [], []
        for state, p in probs.items():
            t = tone_of(state)
            segs.append(f'<i class="t-{t}{" is-pick" if state == pick else ""}" style="flex:{max(p, 0.004):.4f}"></i>')
            legend.append(f'<span class="t-{t}">{esc(word_of(state))} <b>{p:.0%}</b></span>')
        ribbon = (f'<div class="q-ribbon"><div class="q-ribbon-head">{ribbon_head}</div>'
                  f'<div class="q-ribbon-bar" role="img" aria-label="Next-session probabilities">{"".join(segs)}</div>'
                  f'<div class="q-ribbon-legend">{"".join(legend)}</div></div>')
    _render(
        '<div class="q-enter">'
        '<div class="q-regime">'
        f'<div><div class="q-eyebrow">{esc(instrument)}</div>'
        f'<div class="q-regime-word t-{tone_of(regime)}">{esc(word_of(regime))}</div>'
        f'<div class="q-regime-meta">{meta_html}</div></div>'
        f'<div class="q-price"><div class="q-price-label">{live_dot}{esc(price_label)}</div>'
        f'<div class="q-price-value">{esc(price)}</div>'
        + (f'<div class="q-price-change">{change_html}</div>' if change_html else "")
        + "</div></div>"
        + ribbon
        + "</div>"
    )
