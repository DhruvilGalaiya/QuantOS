"""Builds docs/Deployment_Plan.pdf (QuantOS deployment plan + change log).  Needs: pip install reportlab

    python docs/make_deployment_plan.py docs/Deployment_Plan.pdf
"""

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = sys.argv[1]
INK, MUTED, ACCENT, RULE = colors.HexColor("#101828"), colors.HexColor("#52514e"), colors.HexColor("#2a78d6"), \
    colors.HexColor("#d0d5dd")
DONE, HOLD = colors.HexColor("#0ca30c"), colors.HexColor("#c98500")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, textColor=INK, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, textColor=INK, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.6, textColor=INK, alignment=TA_LEFT)
SM = ParagraphStyle("SM", parent=B, fontSize=8.2, leading=10.6, textColor=MUTED)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.2, leading=10.4)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=7.6, leading=9.6, backColor=colors.HexColor("#f4f5f7"),
                      borderPadding=5, textColor=INK)


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items, s=B):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", ParagraphStyle("bl", parent=s, leftIndent=10, firstLineIndent=-8))
            for i in items]


def table(rows, widths, header=True):
    data = [[P(c, CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4fc")))
    t.setStyle(TableStyle(st))
    return t


def status_row(label, state):
    col = DONE if state == "DONE" else HOLD
    return f'<b>{label}</b> &nbsp; <font color="{col.hexval()}"><b>[{state}]</b></font>'


def on_page(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "QuantOS Deployment Plan  ·  v1.0  ·  18 Nov 2026")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
    c.restoreState()


W = A4[0] - 36 * mm
story = []

# ---------------------------------------------------------------- cover / summary
story += [P("QuantOS Deployment Plan", ParagraphStyle("T", parent=H1, fontSize=24, spaceAfter=2)),
          P("System architecture change log and deployment planner", ParagraphStyle("st", parent=B, fontSize=11,
                                                                                    textColor=MUTED)),
          P("Version 1.0 · prepared 18 November 2026", SM), Spacer(1, 10)]
story.append(table([
    ["Step", "Scope", "Status"],
    ["1", "Separate data jobs from dashboard pages (job runner, status DB, read-only public mode)",
     status_row("Step 1", "DONE")],
    ["2", "Restructure code into an installable package (quantos_platform), settings module, one front door",
     status_row("Step 2", "DONE")],
    ["3", "Reliability: data validation, unit + smoke tests, lint, CI, repo hygiene", status_row("Step 3", "DONE")],
    ["4", "Package and deploy (locked deps, Docker, VM, HTTPS, domain, secrets)", status_row("Step 4", "ON HOLD")],
    ["5", "Operations (uptime monitor, alerts, backups, auto-deploy)", status_row("Step 5", "ON HOLD")],
], [12 * mm, W - 52 * mm, 40 * mm]))
story += [Spacer(1, 8), P(
    "Goal: an always-on public showcase of QuantOS for a handful of visitors. The core principle introduced in "
    "Steps 1-3 is <b>compute separately, serve read-only</b>: scheduled jobs fetch data and run models; the website "
    "only reads finished artifacts, so it stays fast and keeps working (with an \"as of\" time) if Upstox or a job "
    "fails. Steps 4 and 5 are planned below and wait for your go-ahead.")]

story += [P("Architecture after Steps 1-3", H2), Preformatted(
    """  Upstox / Yahoo / Nasdaq APIs
            |
            v
  quantos_platform.jobs   (india | vol | playbook | us_equities)      <- cron / timers in Step 4
     lock per job · status DB · logs · retries · validation before every write
            |  writes
            v
  data/  (Parquet, CSV, SQLite)  +  data/system/status.sqlite  (job runs, freshness)
            |  reads only
            v
  Streamlit app  backend/app/regime_dashboard_v10.py
     Market Regimes | Volatility Surface | Expiry Playbook | US Stocks
     QUANTOS_MODE=local  -> refresh stale data on open, refresh buttons
     QUANTOS_MODE=public -> strictly read-only (deployment)""", CODE)]

# ---------------------------------------------------------------- STEP 1
story += [PageBreak(), P(status_row("Step 1 · Separate jobs from pages", "DONE"), H1)]
story += [P("What changed", H2)] + bullets([
    "<b>New job runner</b> <font face='Courier'>quantos_platform/jobs.py</font> - the only code that fetches data "
    "or runs models. Jobs: <b>india</b> (daily candles -> V10 HMM -> V11 forecast), <b>vol</b> (option-chain "
    "snapshot -> SVI/SABR surfaces), <b>playbook</b> (15-min bars + 15:15 signal -> paper log), "
    "<b>us_equities</b> (US closes), plus <b>all</b> and <b>status</b>.",
    "Every job takes a <b>lock</b> (no overlapping runs; stale locks from crashes expire after 30 min), records "
    "start/finish/result in a <b>status database</b>, logs to <font face='Courier'>data/system/logs/quantos.log</font>, "
    "and <b>never raises to the caller</b> - the site keeps serving the last good data.",
    "<b>Status store</b> <font face='Courier'>quantos_platform/core/status.py</font> (SQLite): job_runs table and a "
    "<b>freshness</b> table (india_daily, v11, options_snapshot, playbook, us_equities).",
    "<b>Pages no longer compute.</b> Market Regimes, Volatility Surface, Expiry Playbook and US Stocks read artifacts. "
    "Tonight's playbook signal is written by the job to <font face='Courier'>tonight_&lt;symbol&gt;.json</font>.",
    "<b>Run modes</b> via <font face='Courier'>QUANTOS_MODE</font>: <i>local</i> (default) refreshes stale data when a "
    "page opens (keeps your one-command workflow) and shows refresh buttons; <i>public</i> hides all refresh buttons, "
    "makes no API calls from page views (no live LTP, stored US prices only) and relies on the scheduler.",
    "<b>Shared UI helpers</b> <font face='Courier'>backend/app/ui.py</font>: data-as-of captions, failed-job and "
    "behind-calendar warnings, refresh policy.",
    "<b>Bug fixed</b>: Upstox's historical endpoint publishes today's daily candle only later (often next morning), "
    "so after the close the dashboard stayed a day behind. The daily job now takes today's candle from the intraday "
    "endpoint after 15:45; the next run's overlap download replaces it with the historical value.",
])
story += [P("Job schedule designed for deployment (IST, weekdays)", H2), table([
    ["Time", "Job", "Purpose"],
    ["07:00", "us_equities", "US closes after the New York session"],
    ["08:45", "india", "V11 forecast becomes FINAL once the US session has closed"],
    ["09:15-15:30, every 15 min", "vol", "Fresh option-chain snapshot and surfaces during market hours"],
    ["15:12", "playbook", "Expiry playbook signal at the 15:15 snapshot"],
    ["15:50", "india, vol, playbook", "End-of-day: daily candles, V10/V11, surfaces, paper-log scoring"],
], [38 * mm, 32 * mm, W - 70 * mm])]
story += [P("Verification", H2)] + bullets([
    "<font face='Courier'>python -m quantos_platform.jobs all</font>: india OK (6.3 s), vol OK, playbook OK (18 s), "
    "us_equities OK (98 s, 114 tickers); status shows daily data through 18 Nov, V11 forecast for 19 Nov "
    "(PROVISIONAL until the US close).",
    "All four pages render without exceptions in both local and public mode; public mode shows no refresh buttons.",
])

# ---------------------------------------------------------------- STEP 2
story += [PageBreak(), P(status_row("Step 2 · Restructure into a package", "DONE"), H1)]
story += [P("What changed", H2)] + bullets([
    "<b>Installable package</b> <font face='Courier'>quantos_platform</font> with "
    "<font face='Courier'>pyproject.toml</font> (dependencies, dev extras pytest/ruff, a "
    "<font face='Courier'>quantos-jobs</font> command). Installed editable in .venv: "
    "<font face='Courier'>pip install -e \".[dev]\"</font>. Imports work from any folder.",
    "<b>Name note</b>: the package is <font face='Courier'>quantos_platform</font>, not <font face='Courier'>quantos</font>: "
    "the repo already contains a folder <font face='Courier'>QuantOS/</font> that is a separate nested git repository, "
    "and macOS treats quantos/QuantOS as the same name. That nested repo was not modified.",
    "<b>One settings module</b> <font face='Courier'>core/settings.py</font>: all paths derive from "
    "<font face='Courier'>DATA_DIR</font> (overridable with <font face='Courier'>QUANTOS_DATA_DIR</font> for a server "
    "volume), run mode, log level, token from the environment. No more cwd-relative paths or path patching.",
    "<b>sys.path hacks removed</b> from production code. The only remaining bridges: "
    "<font face='Courier'>quantos_platform/legacy.py</font> (runs the research V10 chain in src/ unchanged) and a "
    "single fallback in the dashboard router.",
    "<b>Upstox client hardened</b> (<font face='Courier'>core/upstox.py</font>): session reuse, retries with backoff "
    "for network errors / 429 / 5xx, and a clear <i>UpstoxAuthError</i> when the token is missing or expired.",
    "<b>Shared market calendar</b> (<font face='Courier'>core/calendar.py</font>): holidays, sessions, market open, "
    "last close, expected latest session, sessions-to-expiry (previously duplicated in three modules).",
    "<b>Volatility engine split</b>: <font face='Courier'>vol/pricing.py</font> (Black-76),"
    " <font face='Courier'>vol/smile_models.py</font> (SVI, Durrleman check, vectorised Hagan SABR - removes the "
    "dependency on the research sabr_surface.py), <font face='Courier'>vol/surface.py</font>, "
    "<font face='Courier'>vol/snapshot.py</font>.",
    "<b>Backwards compatible</b>: every moved src/ file is now a small shim that forwards imports and commands, so "
    "old commands (e.g. <font face='Courier'>python src/quantos_hmm_v11.py</font>, "
    "<font face='Courier'>python src/live_pipeline.py</font>) still work.",
])
story += [P("Module map (old -> new)", H2), table([
    ["Old location", "New module"],
    ["src/upstox_client.py, src/config.py", "core/upstox.py, core/instruments.py"],
    ["src/live_pipeline.py", "data/daily.py (data) + models/v10.py (V10 bridge) + jobs.py (orchestration)"],
    ["src/intraday_data.py, derivatives_data.py, backfill_daily_history.py", "data/intraday.py, derivatives.py, backfill.py"],
    ["src/quantos_portfolio/us_data.py (incremental), us_universe.py, us_fundamentals.py",
     "data/us_equities.py, data/us_universe.py, data/us_fundamentals.py"],
    ["src/quantos_v11_features.py, quantos_v11_regime.py, quantos_hmm_v11.py",
     "models/features_v11.py, models/regime_hmm.py, models/v11.py"],
    ["src/quantos_v12_snapshot.py, quantos_v12_intraday_hmm.py", "models/snapshot_1515.py, models/intraday_hmm.py"],
    ["src/quantos_overnight_options.py, quantos_expiry_playbook.py",
     "strategies/overnight_options.py, strategies/expiry_playbook.py"],
    ["src/quantos_volatility/surface_engine.py, vol_pipeline.py", "vol/pricing.py, vol/smile_models.py, vol/surface.py, vol/snapshot.py"],
    ["(new)", "core/settings.py, calendar.py, status.py, validation.py, logging.py; jobs.py; legacy.py"],
], [W * 0.47, W * 0.53])]
story += [P("Front door", H2), P(
    "The single Streamlit app (<font face='Courier'>backend/app/regime_dashboard_v10.py</font>) is the deployable "
    "front door. The FastAPI backend, React frontend, the older dashboards (backend/app/regime_dashboard.py, "
    "src/quantos_volatility/options_dashboard.py) and the mlops/ serving app were <b>left in place, not archived</b>: "
    "they may still matter to you, so archiving them is listed as a decision under Step 4.")]

# ---------------------------------------------------------------- STEP 3
story += [PageBreak(), P(status_row("Step 3 · Reliability", "DONE"), H1)]
story += [P("Data validation before every write", H2)] + bullets([
    "<font face='Courier'>core/validation.py</font>: OHLC checks (columns, unparseable/duplicate/out-of-order "
    "timestamps, non-positive prices, high/low consistency, implausible jumps: 20% for indices, 100% for India VIX), "
    "option-snapshot size, surface sanity (ATM IV 1-200%, SVI error &lt; 5 vol pts), prediction-history schema.",
    "Wired in: daily candles (new rows checked before the atomic write), option snapshots, volatility surfaces "
    "(a failing surface keeps the last good file), V11 history (atomic write after a schema check).",
])
story += [P("Tests (55 collected: 54 pass + 1 intentional skip)", H2), table([
    ["File", "What it proves"],
    ["test_pricing.py", "Black-76 put-call parity; IV inversion round-trips across strikes/vols; prices outside "
                        "no-arbitrage bounds are rejected"],
    ["test_smile_models.py", "SVI fit recovers a known smile (&lt; 0.05 vol pts); Durrleman check passes a clean "
                             "slice and flags the Gatheral-Jacquier arbitrage example; new SABR formula equals the "
                             "original research implementation"],
    ["test_no_lookahead.py", "HMM regime filter is causal (past probabilities unchanged when future data arrives); "
                             "the 15:15 playbook model ignores everything on/after the signal day"],
    ["test_validation.py", "Good bars pass; duplicates, wrong order, broken high/low and jumps are caught"],
    ["test_calendar.py", "Sessions-to-expiry and holiday handling (network mocked)"],
    ["test_jobs.py", "Failures are recorded not raised; freshness is stored; overlapping runs are skipped; "
                     "public mode never runs jobs"],
    ["test_pages.py (slow)", "All four pages render read-only in public mode with no refresh buttons (skipped "
                             "automatically when the data directory is absent, e.g. in CI)"],
], [36 * mm, W - 36 * mm])]
story += [P("Lint and CI", H2)] + bullets([
    "<b>ruff</b> configured in pyproject.toml; new code (quantos_platform, tests, backend/app/ui.py) is lint-clean. "
    "Research code in src/ and the large page files are excluded for now.",
    "<b>New workflow</b> <font face='Courier'>.github/workflows/platform-ci.yml</font>: install package, ruff, pytest "
    "on every push/PR touching the platform.",
    "<b>Repaired</b> <font face='Courier'>.github/workflows/mlops-ci.yml</font>: the file contained a pasted shell "
    "script (cd ... / cat &gt; ... &lt;&lt;'YAML') rather than YAML, so GitHub could not parse it. Restored the YAML "
    "and made the Docker build step conditional on a Dockerfile existing (there is none at the repo root yet).",
])
story += [P("Repo hygiene", H2)] + bullets([
    "Stopped tracking 83 cache/OS files (.pyc, __pycache__, .DS_Store) - removed from the git index only; files "
    "remain on disk. Staged, not committed.",
    ".gitignore extended: .DS_Store, pytest/ruff caches, egg-info, build outputs, runtime state "
    "(<font face='Courier'>data/system/</font>), temp writes.",
    "Not changed (decision in Step 4): ~61 MB of market data is tracked in git.",
])

# ---------------------------------------------------------------- STEP 4 / 5
story += [PageBreak(), P(status_row("Step 4 · Package and deploy", "ON HOLD"), H1),
          P("Waiting for your go-ahead. Planned work, in order:", B)]
story.append(table([
    ["#", "Task", "Detail"],
    ["4.1", "Lock dependencies", "Generate a lock file (uv or pip-tools) from pyproject.toml so the server installs "
                                 "exactly the tested versions (Python 3.13)."],
    ["4.2", "Dockerfile + docker-compose", "Images: <b>app</b> (Streamlit, QUANTOS_MODE=public), <b>scheduler</b> "
                                           "(jobs on the schedule in Step 1), <b>caddy</b> (HTTPS reverse proxy). "
                                           "Named volume for data/ (QUANTOS_DATA_DIR)."],
    ["4.3", "Scheduler", "cron in the scheduler container (or systemd timers on the VM) running "
                         "<font face='Courier'>python -m quantos_platform.jobs ...</font>; IST timezone."],
    ["4.4", "Host", "Small VM (Hetzner / DigitalOcean / Lightsail, ~$5-10/month, 2 vCPU / 2-4 GB). Avoid free tiers "
                    "that sleep or wipe disks (Streamlit Community Cloud, free Hugging Face Spaces)."],
    ["4.5", "Domain + HTTPS", "Point a domain at the VM; Caddy obtains certificates automatically."],
    ["4.6", "Secrets", "UPSTOX_ANALYTICS_TOKEN in the server environment / compose secrets - never in the image or "
                       "repo. Record the analytics-token expiry date and set a reminder; the site already shows a clear "
                       "error when the token is rejected."],
    ["4.7", "Seed the data volume", "Copy the current data/ to the server once (or decide to rebuild from APIs). "
                                    "Decide whether market data stays in git or moves out (recommended: out of git, "
                                    "plus a small sample dataset so the app can run as a demo without a token)."],
    ["4.8", "Decisions needed", "Archive or keep: FastAPI backend, React frontend, older dashboards, mlops serving "
                                "app. Add an About / Methodology landing page and disclaimers for the showcase."],
], [10 * mm, 36 * mm, W - 46 * mm]))
story += [Spacer(1, 6), P("Exit criteria: site reachable over HTTPS on the domain; jobs run on schedule for 5 trading "
                           "days without manual help; public mode verified.", SM)]

story += [Spacer(1, 10), P(status_row("Step 5 · Operations", "ON HOLD"), H1),
          P("Waiting for your go-ahead. Planned work:", B)]
story.append(table([
    ["#", "Task", "Detail"],
    ["5.1", "Uptime monitoring", "Free uptime monitor (UptimeRobot / Better Stack) pinging "
                                 "<font face='Courier'>/_stcore/health</font> every 5 min."],
    ["5.2", "Job / staleness alerts", "Small job reading data/system/status.sqlite: alert by email or Telegram when a "
                                      "job fails twice or a dataset is older than expected (calendar-aware)."],
    ["5.3", "Backups", "Nightly backup of data/ to object storage (Backblaze B2 / Cloudflare R2 / S3). The paper-trade "
                       "log and prediction histories cannot be recreated."],
    ["5.4", "Auto-deploy", "CI builds and pushes the image on main; the server pulls and restarts (watchtower or a "
                           "deploy step over SSH)."],
    ["5.5", "Track-record integrity", "Append-only, timestamped forecast and signal logs; optionally a daily "
                                      "commit of the logs to a public repo as proof they were made before outcomes."],
], [10 * mm, 36 * mm, W - 46 * mm]))

# ---------------------------------------------------------------- reference
story += [PageBreak(), P("Reference", H1), P("Everyday commands", H2), Preformatted(
    """pip install -e ".[dev]"                                    # once per environment
python -m streamlit run backend/app/regime_dashboard_v10.py   # the dashboard (all pages)
python -m quantos_platform.jobs india|vol|playbook|us_equities|all [--force]
python -m quantos_platform.jobs status                         # last runs + data freshness
QUANTOS_MODE=public python -m streamlit run backend/app/regime_dashboard_v10.py   # preview public site
python -m pytest -q                                            # tests
ruff check quantos_platform tests backend/app/ui.py            # lint""", CODE)]
story += [P("Environment variables", H2), table([
    ["Variable", "Meaning"],
    ["UPSTOX_ANALYTICS_TOKEN", "Upstox read-only market-data token (required by jobs)"],
    ["QUANTOS_MODE", "local (default) or public"],
    ["QUANTOS_DATA_DIR", "Data directory (default: repo/data)"],
    ["QUANTOS_LOG_LEVEL", "Logging level (default INFO)"],
], [50 * mm, W - 50 * mm])]
story += [P("Known limitations / follow-ups", H2)] + bullets([
    "US Stocks fundamentals (ratios, financial statements) are still fetched live from Yahoo on page view (cached "
    "6 h). Fine for a few visitors; could be moved into the us_equities job later.",
    "The large page files (market_regimes.py, us_stocks.py) are not yet under the linter.",
    "The V10 model still runs through the original research scripts via the legacy bridge (unchanged by design).",
    "Changes are not committed. Code backup taken before the restructure (scratchpad, temporary).",
])

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=16 * mm, title="QuantOS Deployment Plan", author="QuantOS",
                        subject="Architecture change log and deployment planner")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
