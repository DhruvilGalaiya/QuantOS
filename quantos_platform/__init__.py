"""
QuantOS: quantitative market analytics platform (production package).

    quantos_platform.core        settings, Upstox client, market calendar, job status, validation
    quantos_platform.data        market data ingestion (daily, intraday, derivatives, US equities)
    quantos_platform.models      V10 bridge, V11 next-session model, 15:15 snapshot features
    quantos_platform.vol         Black-76 pricing, SVI / SABR smiles, volatility surface engine
    quantos_platform.strategies  overnight option backtests, expiry-week playbook
    quantos_platform.jobs        scheduled jobs (the only code that fetches data or runs models)

Research scripts that are not part of the live system stay in src/.
"""

__version__ = "1.0.0"
