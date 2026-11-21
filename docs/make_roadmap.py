"""Builds docs/QuantOS_Roadmap.pdf (QuantOS v2 product and quant roadmap).  Needs: pip install reportlab

    python docs/make_roadmap.py docs/QuantOS_Roadmap.pdf

Headline numbers are read from data/analytics/ when present, so rebuild after `python -m quantos_platform.jobs analytics`.
"""

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/QuantOS_Roadmap.pdf"
ANALYTICS = Path(__file__).resolve().parents[1] / "data" / "analytics"
DATE = "20 November 2026"

INK, MUTED, ACCENT, RULE = (colors.HexColor(c) for c in ("#191B22", "#5C6270", "#3346D3", "#E4E0D6"))
DONE, NEXT, LATER = (colors.HexColor(c) for c in ("#13875A", "#B7791F", "#8C919C"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Times-Roman", fontSize=21, textColor=INK, spaceBefore=6,
                    spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10, textColor=INK, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.8, textColor=INK, alignment=TA_LEFT)
SM = ParagraphStyle("SM", parent=B, fontSize=8.1, leading=10.6, textColor=MUTED)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.1, leading=10.4)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
QUOTE = ParagraphStyle("Q", parent=B, fontSize=10.5, leading=14.5, leftIndent=10, borderPadding=(6, 8, 6, 8),
                       backColor=colors.HexColor("#F1F2FB"), textColor=INK)


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items, s=B):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", ParagraphStyle("bl", parent=s, leftIndent=10, firstLineIndent=-8))
            for i in items]


def table(rows, widths, header=True):
    data = [[P(c, CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F3EE")))
    t.setStyle(TableStyle(st))
    return t


def tag(state):
    col = {"BUILT": DONE, "NEXT": NEXT}.get(state, LATER)
    return f'<font color="{col.hexval()}"><b>[{state}]</b></font>'


def on_page(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, f"QuantOS v2 Roadmap  ·  {DATE}")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
    c.restoreState()


def load(name):
    p = ANALYTICS / name
    return json.loads(p.read_text()) if p.exists() else {}


vol = load("volatility_nifty_50.json")
ev = load("events_summary_nifty_50.json")
vrp = vol.get("vrp", {})
W = A4[0] - 36 * mm
story = []

# ---------------------------------------------------------------- cover
story += [
    P("QuantOS v2", ParagraphStyle("T", parent=H1, fontSize=30, leading=36, spaceAfter=4)),
    P("Product and quantitative roadmap", ParagraphStyle("st", parent=B, fontSize=13, textColor=MUTED)),
    P(f"Prepared {DATE} · Phase 1 built locally, awaiting review", SM), Spacer(1, 12),
    P("<i>Be the house, not the player.</i> QuantOS does not try to predict where the market goes next. It measures "
      "the state of the market and its risk, and shows what history says happened under similar conditions, with "
      "confidence intervals and significance tests. The reader makes the decision.", QUOTE),
    Spacer(1, 8),
]
story += [P("Why this direction", H2)] + bullets([
    "Short-horizon direction is close to a coin flip once costs are paid; our own V11 research found only the "
    "overnight US session adds signal. Competing on prediction is playing the casino's game.",
    "The durable edges in markets are structural: the volatility risk premium paid to option sellers, the shape of "
    "risk (fat tails, volatility clustering), and base rates that cut through narrative. Those can be measured "
    "honestly and are what professional desks monitor.",
    "Inspiration: Quant Guild (Roman Paolucci) on stochastic and rough volatility, Markov chains and "
    "market-neutral thinking; Zerodha's In The Money for its data-first studies (The Long &amp; The Short) and "
    "Weekly Market Metrics.",
    "Audience: traders and investors who want analysis rather than tips, and recruiters evaluating quantitative "
    "and machine-learning depth.",
])
story += [P("What QuantOS is today", H2)] + bullets([
    "<b>Market Regimes</b>: three-state Gaussian HMM per index, transition matrix, V11 next-session model.",
    "<b>Volatility Surface</b>: live option chains, SVI and SABR calibration, skew and term structure.",
    "<b>Expiry Playbook</b>: paper-tracked near-expiry option-selling research strategy.",
    "<b>US Stocks</b>: price history and fundamentals for 150 large US companies.",
    "Platform: <font face='Courier'>quantos_platform</font> package, scheduled jobs write artifacts, Streamlit pages "
    "only read them; public deployment on Streamlit Community Cloud with a GitHub Pages landing page.",
])

# ---------------------------------------------------------------- modules
story += [PageBreak(), P("The five modules", H1),
          P("Each module ships a computation package, a job, stored artifacts, a page, tests, a plain-language "
            "'what this means' line on every chart and a collapsed method-and-caveats note.", SM), Spacer(1, 4)]

modules = [
    ("1. Volatility &amp; the House Edge", "BUILT",
     "How much the market moves, how much option prices say it will move, and who is paid for the difference.",
     ["Volatility risk premium: India VIX against the realized volatility that actually followed (21 sessions), "
      "by VIX level and by year; today's premium and its percentile.",
      "Volatility cone: today's realized volatility against the full history for 10 to 252-session windows.",
      "Four realized-volatility estimators (close-to-close, Parkinson, Garman-Klass, Yang-Zhang).",
      "GARCH(1,1) with Student-t innovations: shock weight, persistence, half-life, tail thickness, term structure.",
      "Rough volatility: Hurst exponent of log-volatility from moment scaling (Gatheral, Jaisson &amp; Rosenbaum).",
      "Seller's edge by days to expiry: one-session ATM straddle outcomes from stored expired chains vs textbook "
      "theta decay."],
     (f"Finding: India VIX exceeded subsequent realized volatility on {vrp['share_implied_above']:.0%} of days since "
      f"{vrp['since'][:4]} (average {vrp['mean_spread'] * 100:.1f} vol points); NIFTY log-volatility has "
      f"H ≈ {vol['hurst']['H']:.2f}, i.e. rough.") if vrp and vol else None),
    ("2. Market States (Markov analytics)", "NEXT",
     "Turn the HMM page into a description of regimes rather than a forecast.",
     ["Transition matrix with bootstrap confidence bands; expected regime durations and survival curves.",
      "Regime-conditional return, volatility and drawdown statistics, using filtered (not smoothed) "
      "probabilities to avoid look-ahead.",
      "Rule-based correction and bear-market classification; full drawdown history since 2007.",
      "Regime-switch events added to the Event Lab once look-ahead-free labels exist."], None),
    ("3. Event Study Lab", "BUILT",
     "What actually happened after the events traders talk about.",
     ["14 events: 52-week highs and lows, weekly and daily streaks, ±2% days, ±1% gaps, VIX spikes and VIX "
      "entering its top decile, 10% corrections, 20% bear markets, Union Budget days.",
      "Forward returns at 1, 5, 21 and 63 sessions from the event-day close; de-clustering of repeated triggers.",
      "Bootstrap 95% confidence intervals, bootstrap p-value against ordinary days, average path with a band.",
      "Explicit multiple-testing note: with 14 events, about 0.7 pass at 5% by luck."],
     (f"Finding: over one month, {sum(1 for e in ev['events'].values() if (e['stats']['h21'].get('p_value') or 1) < 0.05)}"
      " of 14 NIFTY events differ from an ordinary day at the 5% level. Most famous signals do not.")
     if ev else None),
    ("4. Positioning &amp; Market Structure", "LATER",
     "Where option open interest concentrates and what dealers may need to hedge.",
     ["Dealer gamma exposure by strike, gamma-flip level, call and put walls.",
      "Put-call ratio, max pain, open-interest build-up classification (long build-up, short covering, ...).",
      "Every metric shown against its own history so levels have context."], None),
    ("5. Systemic &amp; Tail Risk", "NEXT",
     "How fragile the market is and how bad bad days get.",
     ["Rolling sector correlation heatmap, PCA absorption ratio, Mahalanobis turbulence index, sector dispersion "
      "and rotation, hierarchical clustering.",
      "GARCH-t filtered historical VaR and CVaR, extreme-value tail index, Monte Carlo stress, replays of 2008 and "
      "2020 on any index or sector mix."], None),
]
for title, state, lede, items, finding in modules:
    block = [P(f"{title} &nbsp; {tag(state)}", H2), P(lede, B)] + bullets(items, SM)
    if finding:
        block.append(P(f"<b>{finding}</b>", ParagraphStyle("f", parent=SM, textColor=INK, spaceBefore=3)))
    story.append(KeepTogether(block))

# ---------------------------------------------------------------- tracks
story += [PageBreak(), P("Two tracks alongside the modules", H1)]
story += [P("Strategy Lab (one strategy a week)", H2),
          P("A transparent research notebook, not a signal service. Every entry follows the same template:", B)]
story += bullets([
    "Hypothesis and the market mechanism behind it.",
    "Exact rules, costs and execution assumptions.",
    "Walk-forward backtest; deflated Sharpe ratio and probability of backtest overfitting.",
    "Kelly-based position sizing and risk-of-ruin, to show why sizing matters more than signals.",
    "Live paper log from the publish date, never edited retroactively.",
], SM)
story += [P("First entries: the Expiry Playbook (already paper-tracked), a VRP-filtered short strangle, a "
            "market-neutral BANK/FIN pair, sector momentum rotation, volatility-targeted NIFTY.", SM)]
story += [P("Weekly Market Metrics email", H2)] + bullets([
    "Monday before the open: regime state, volatility premium, move priced to expiry, positioning levels, risk "
    "dashboard, the event study of the week, Strategy Lab update and a short news summary.",
    "Built from the same job artifacts as the dashboard, so the numbers always match.",
    "Public sign-up on the landing page with double opt-in and one-click unsubscribe; sent via a free tier "
    "provider (Resend or Brevo) from a scheduled GitHub Actions job.",
], SM)

# ---------------------------------------------------------------- design + platform
story += [P("Design", H2)] + bullets([
    "New default <b>Today</b> page: the whole market state on one screen with links into each module.",
    "Keep the light Research-desk theme; shared components for metric cards, 'what this means' notes and "
    "method expanders, so every page reads the same way.",
    "Every number carries context: a percentile, a historical comparison or a confidence interval.",
], SM)
story += [P("Platform", H2)] + bullets([
    "Pattern per module: <font face='Courier'>quantos_platform/&lt;area&gt;/</font> computation, a job in "
    "<font face='Courier'>jobs.py</font>, artifacts in <font face='Courier'>data/</font>, a read-only page, tests.",
    "Phase 1 adds the <font face='Courier'>analytics</font> job (no API calls, ~7 s).",
    "Phase 4 moves data refresh to a scheduled GitHub Actions workflow (Upstox token as a repository secret) so "
    "the public site and the email stay current without a laptop.",
    "Changes are reviewed locally before anything is pushed or deployed.",
], SM)

# ---------------------------------------------------------------- phases
story += [P("Phases", H2), table([
    ["Phase", "Scope", "Status"],
    ["1", "Volatility &amp; House Edge · Event Study Lab · Today page · roadmap PDF", tag("BUILT") + " local, in review"],
    ["2", "Market States (Markov analytics) · Systemic &amp; Tail Risk", tag("NEXT")],
    ["3", "Positioning &amp; Market Structure · Strategy Lab", tag("LATER")],
    ["4", "Weekly Market Metrics email (public sign-up) · GitHub Actions data refresh", tag("LATER")],
], [16 * mm, W - 56 * mm, 40 * mm])]
story += [Spacer(1, 10), P("QuantOS is a research and education project. Nothing in it is investment advice; "
                           "past behaviour of markets does not guarantee future results.", SM)]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=16 * mm, title="QuantOS v2 Roadmap", author="QuantOS",
                        subject="Product and quantitative roadmap")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
