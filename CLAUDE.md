QuantOS is built around a Python-based quantitative analytics engine. We integrated Upstox REST APIs to retrieve market and options data, then engineered time-series features such as returns, rolling volatility, drawdowns and sector dispersion. A three-state Gaussian Hidden Markov Model processes these features to identify Bull, Sideways and Bear regimes and estimate transition probabilities. For options analytics, we construct implied volatility surfaces across strikes and expiries and calibrate SABR models to analyse volatility skew and term structure. We also implemented GARCH(1,1)-Student-t volatility forecasting and 10,000-path Monte Carlo simulations for market-risk scenarios. Results are exposed through interactive dashboards and APIs.
Important technology stack
Technology	Where and how we used it
Python	Core language for data processing, feature engineering, model training, forecasting and quantitative analytics.
Pandas, NumPy	Time-series manipulation, numerical calculations, rolling indicators and preparation of model inputs.
HMM / Gaussian Models	Three-state Gaussian HMM for identifying latent market regimes and estimating transition probabilities.
GARCH(1,1) + Student-t	Modelling volatility clustering and producing volatility forecasts with heavy-tailed return assumptions.
SABR Model	Calibrating stochastic-volatility parameters to options data and analysing implied volatility across strikes and expiries.
Monte Carlo Simulation	Generating 10,000 simulated market paths for probabilistic scenario and risk analysis.
Upstox API	Retrieving live market prices and options-chain data for NIFTY 50, BANK NIFTY and SENSEX.
FastAPI	Backend API layer for exposing analytics and connecting the quantitative engine to the application.
React	Frontend application for presenting platform functionality and analytical outputs.
Streamlit	Interactive dashboards for specialised HMM and volatility-surface analysis.
SQLite	Local storage for live options snapshots and related market data.

python -m streamlit run backend/app/regime_dashboard_v10.py

## QuantOS v3: React terminal + Node gateway (replacing the Streamlit dashboard, milestones in ~/.claude/plans/mighty-sparking-graham.md)
Layout: web/ (Vite, React 19, TS, Tailwind v4, shadcn/ui, Lightweight Charts) -> gateway/ (Fastify, TS: helmet, CORS, rate limit, ETag cache)
-> quantos_platform/api/ (FastAPI, internal only) -> quantos_platform/services/ (page logic, JSON-ready dicts). Types: packages/types (generated).
Setup once: npm install
npm run dev:api      # Python quant service on 127.0.0.1:8001 (docs at /internal/docs)
npm run dev:gateway  # Node gateway on 127.0.0.1:8787 (/api/v1/*)
npm run dev:web      # React app on http://localhost:5173 (proxies /api to the gateway)
npm run gen:types    # after changing quantos_platform/api/schemas.py
npm run build && npm run preview -w @quantos/web   # public static build (exports JSON first), preview on :4173
npm run typecheck && npm run lint && npm test      # web + gateway checks (CI: .github/workflows/web-ci.yml)
Public site: https://dhruvilgalaiya.github.io/QuantOS/ deployed by .github/workflows/deploy-site.yml on every push to main that
touches data/, quantos_platform/, web/ (export JSON from committed artifacts -> VITE_BASE=/QuantOS/ build -> web/scripts/prerender.mjs
writes per-route HTML, 404.html, sitemap.xml, robots.txt). Pages source must be "GitHub Actions".
Daily data: .github/workflows/daily-data.yml runs the jobs on GitHub (14:45 IST vol, 18:00 IST india+analytics, 07:00 IST US;
manual "Run workflow" too), saves data/ to the "data-latest" release (not git), then redeploys. Needs repo secret UPSTOX_ANALYTICS_TOKEN.
scripts/data_bundle.sh pull   # bring the cloud's newest data/ down to this Mac (do this before running jobs locally)
Owner live prices: npm run hash-password -w gateway -> put the 2 lines in gateway/.env -> restart gateway -> ⌘K "Sign in for live prices"
(Server-Sent Events /api/v1/ticks/stream, Python /internal/v1/ticks; off in public mode). Data status page: /status (/internal/v1/system).
npm run e2e -w @quantos/web   # Playwright smoke tests on the built site (installed Chrome; BASE=/QuantOS/ for a Pages build)
python -m quantos_platform.data.us_snapshot [--force]   # weekly US fundamentals/statements snapshot (also inside the us_equities job)
React routes: / landing · /today · /status · /regimes/:symbol · /states/:symbol · /events/:symbol · /volatility/:symbol · /surface/:symbol · /risk · /us · /us/:ticker (all pages built; Streamlit is now the research lab only)

## Production platform (quantos_platform package)
Setup once: ./.venv/bin/pip install -e ".[dev]"
Layout: quantos_platform/{core,data,models,vol,strategies}, jobs.py; UI pages in backend/app/views/ (+ backend/app/ui.py); research code stays in src/ (old src/ entry points for moved modules are thin compatibility shims).
Rule: only jobs fetch data / run models; pages read artifacts. QUANTOS_MODE=public makes pages strictly read-only (no refresh on open, no refresh buttons).

python -m streamlit run backend/app/regime_dashboard_v10.py   # THE dashboard (deploy this one): Market Regimes | Volatility Surface | Expiry Playbook | US Stocks
python -m quantos_platform.jobs india|vol|playbook|us_equities|all [--force]   # data + model jobs (lock, status DB, logs in data/system/)
python -m quantos_platform.jobs status        # last runs + data freshness
python -m quantos_platform.models.v11 [--rebuild]        # V11 full walk-forward rebuild (~2 min)
python -m quantos_platform.strategies.expiry_playbook --backtest   # rebuild expiry playbook backtest book
python -m quantos_platform.data.backfill      # one-off: extend daily index history back to 2007
python -m quantos_platform.data.us_universe   # rebuild US Stocks list (Nasdaq-100 top 50 + S&P 500 top 100)
python -m quantos_platform.data.expiry_week     # expiry-week option chains for the Expiry Strategy Lab (resumable; first run ~2.5 h)
python -m quantos_platform.analytics.news_calendar   # coming week's scheduled news (also runs inside the analytics job)
python -m pytest -q && ruff check quantos_platform tests backend/app/ui.py   # tests + lint (CI: .github/workflows/platform-ci.yml)
Deployment plan / change log: docs/Deployment_Plan.pdf
python -m streamlit run src/quantos_volatility/volatility_surface_dashboard.pygit add . && git commit -m "Update project changes" && git push
python -m src.quantos_portfolio.us_datastreamlit run backend/app/regime_dashboard.py --server.port 8501

## IMPORTANT: Sound Notification

After finishing responding to my request or running a command, run this command to notify me by sound:

```bash
afplay /System/Library/Sounds/Funk.aiff
```