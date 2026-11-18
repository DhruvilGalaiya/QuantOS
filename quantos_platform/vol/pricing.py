"""Black-76 pricing and implied volatility for European options on a forward."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76(F, K, T, df, sigma, is_call):
    sigma = np.maximum(sigma, 1e-12)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def implied_vol(price, F, K, T, df, is_call) -> float:
    """Black-76 implied vol, NaN if the price breaks no-arbitrage bounds."""
    if not (np.isfinite(price) and price > 0):
        return np.nan
    intrinsic = df * max((F - K) if is_call else (K - F), 0.0)
    upper = df * (F if is_call else K)
    if price <= intrinsic + 1e-10 or price >= upper:
        return np.nan
    f = lambda s: float(black76(F, K, T, df, s, is_call)) - price  # noqa: E731
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-7, maxiter=200)
    except ValueError:
        return np.nan
