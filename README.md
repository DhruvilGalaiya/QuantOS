# QuantOS

Volatility and risk analytics for Indian index markets: NIFTY 50, NIFTY BANK and SENSEX, refreshed after every
session.

**Live site: https://dhruvilgalaiya.github.io/QuantOS/**

QuantOS measures the market instead of predicting it. It shows the option-implied range to expiry, which regime the
market is in and how long regimes usually last, what happened after similar signals in the past (with confidence
intervals), and how much a portfolio could lose on a bad day. Every number comes with a plain-language reading.

| Page | What it answers | Method |
|---|---|---|
| Today | Where the market stands in one screen | Option-implied ±1σ cone, regime, volatility, unusual moves |
| Market regimes | Bull, sideways or bear, and for how long | Three-state Gaussian HMM; walk-forward scored next-session model |
| Market states | Calm, normal or stressed, and what usually follows | Look-ahead-free HMM (forward filter, yearly refit) ordered by volatility; survival curves, bootstrap transition bands |
| Event lab | What followed gaps, streaks and volatility spikes | Event study with bootstrap confidence intervals |
| Volatility & edge | Is option premium rich or cheap | Realized vs implied, volatility cone, GARCH(1,1)-Student-t, straddle decay per lot |
| Volatility surface | Implied volatility across strikes and expiries | SVI and SABR fits, arbitrage checks, interactive 3D surface |
| Risk monitor | How much could I lose, and do sectors move together | GARCH-t filtered historical simulation and Monte Carlo VaR/CVaR with a walk-forward backtest; absorption ratio and turbulence; crash replay |
| US stocks | Large US companies, explained | Ratios, statements and prices in plain words |
| Data status | Is the data fresh | Last run of every job, dataset freshness |

## Architecture

```
Browser ── React 19 + TypeScript (web/)          static JSON on GitHub Pages for the public site
              │ /api/v1/*  ·  live prices over Server-Sent Events (owner only)
              ▼
         Node gateway (gateway/, Fastify)        security boundary: helmet CSP, CORS allow-list, rate limits,
              │                                  zod validation, ETag cache, scrypt + signed-cookie owner sign-in
              ▼
         Python quant service (quantos_platform/api, FastAPI, internal only)
              │
         quantos_platform/services → models, vol, analytics → data/ (parquet + JSON artifacts)
```

- **Only jobs fetch data or run models; pages read artifacts.** The jobs run on GitHub Actions every weekday
  (option chain in market hours, index candles and models after the close, US prices next morning). The newest
  `data/` is kept as one file on the `data-latest` release, not in git history, and the site rebuilds after each run.
- **One contract end to end.** Pydantic response models → OpenAPI → generated TypeScript types (`packages/types`),
  used by the gateway and the web app. The static export validates every payload with the same models.
- **The public site needs no servers.** `scripts/export_static.py` writes every API response to JSON, the web app is
  built in static mode, and every route is prerendered with its own title, description and JSON-LD.

## Stack

Python (pandas, NumPy, SciPy, hmmlearn, arch), FastAPI · Node 22, Fastify, zod · React 19, Vite, Tailwind v4,
shadcn/ui, TanStack Query, Lightweight Charts, custom SVG/canvas charts, React Bits · Upstox API (prices, option
chains), Yahoo Finance (US) · GitHub Actions, GitHub Pages · pytest, Vitest, Playwright.

## Run it locally

```bash
./.venv/bin/pip install -e ".[dev]"    # Python engine (Python 3.11+)
npm install                            # Node workspaces: web, gateway, packages/types

npm run dev:api        # quant service on 127.0.0.1:8001 (docs at /internal/docs)
npm run dev:gateway    # gateway on 127.0.0.1:8787
npm run dev:web        # app on http://localhost:5173

python -m quantos_platform.jobs all    # refresh data and models (needs UPSTOX_ANALYTICS_TOKEN)
scripts/data_bundle.sh pull            # or download the cloud's newest data
```

Live prices for the owner: run `npm run hash-password -w gateway`, put the two printed lines in `gateway/.env`
(never committed), restart the gateway, then press ⌘K and choose **Sign in for live prices**.

## Tests

```bash
python -m pytest -q && ruff check quantos_platform tests       # services, API, models
npm run typecheck && npm run lint && npm test                   # gateway routes, auth, web units
npm run build && npm run e2e -w @quantos/web                    # every page in Chrome, desktop and phone
```

CI runs all of these; the site is published only when the browser smoke tests pass.

## Notes

This is a research and education project. Nothing here is investment advice.
