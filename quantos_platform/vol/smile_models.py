"""Smile models: raw SVI (Gatheral) and SABR (Hagan), with no-arbitrage helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def sabr_hagan_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied volatility (vectorised over K)."""
    F, T = float(F), float(T)
    K = np.asarray(K, dtype=float)
    if F <= 0 or T <= 0 or alpha <= 0 or nu <= 0:
        return np.full_like(K, np.nan)
    out = np.full_like(K, np.nan)
    ok = K > 0
    k = K[ok]
    omb = 1.0 - beta
    fk = F * k
    log_fk = np.log(F / k)
    fk_pow = fk ** (omb / 2.0)
    correction = 1.0 + (omb ** 2 / 24.0 * alpha ** 2 / fk ** omb
                        + rho * beta * nu * alpha / (4.0 * fk_pow)
                        + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0) * T
    denom = fk_pow * (1.0 + omb ** 2 / 24.0 * log_fk ** 2 + omb ** 4 / 1920.0 * log_fk ** 4)
    z = nu / alpha * fk_pow * log_fk
    sq = np.sqrt(np.maximum(1e-12, 1.0 - 2.0 * rho * z + z * z))
    xz = np.log((sq + z - rho) / (1.0 - rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_over_x = np.where(np.abs(xz) < 1e-12, 1.0, z / xz)
    vol = alpha / denom * z_over_x * correction
    # exactly at the money use the closed-form ATM expansion
    atm = np.abs(F - k) < 1e-10
    if atm.any():
        vol[atm] = alpha / F ** omb * (1.0 + (omb ** 2 / 24.0 * alpha ** 2 / F ** (2 - 2 * beta)
                                              + rho * beta * nu * alpha / (4.0 * F ** omb)
                                              + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0) * T)
    out[ok] = vol
    return out


def svi_w(k, a, b, rho, m, sig):
    """Raw SVI total implied variance."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def svi_derivs(k, a, b, rho, m, sig):
    root = np.sqrt((k - m) ** 2 + sig ** 2)
    w = a + b * (rho * (k - m) + root)
    w1 = b * (rho + (k - m) / root)
    w2 = b * sig ** 2 / root ** 3
    return w, w1, w2


def durrleman_g(k, params):
    """Butterfly-arbitrage function: the implied density is >= 0 iff g(k) >= 0."""
    w, w1, w2 = svi_derivs(k, *params)
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def fit_svi(k, iv, T, weights):
    w_mkt = iv ** 2 * T
    atm_w = float(np.interp(0.0, k, w_mkt)) if k.min() < 0 < k.max() else float(np.median(w_mkt))
    span = max(k.max() - k.min(), 0.02)

    def resid(p):
        w = svi_w(k, *p)
        model_iv = np.sqrt(np.maximum(w, 1e-12) / T)
        r = (model_iv - iv) * weights
        # keep total variance positive everywhere: a + b*sig*sqrt(1-rho^2) >= 0
        a, b, rho, m, sig = p
        floor = a + b * sig * np.sqrt(1 - rho ** 2)          # min total variance >= 0
        lee = b * (1 + abs(rho)) - 2.0                       # Roger Lee: wing slope of w <= 2
        pen = 10.0 * max(0.0, -floor) / max(atm_w, 1e-8) + 10.0 * max(0.0, lee)
        return np.append(r, pen)

    lo = [-atm_w * 2, 1e-6, -0.99, k.min(), 1e-4]
    hi = [atm_w * 2 + 1e-6, 2.0, 0.99, k.max(), 2 * span + 0.5]
    best = None
    for rho0 in (-0.6, -0.2, 0.2):
        for sig0 in (0.25 * span, span):
            b0 = max(atm_w / max(sig0, 1e-3), 1e-4)
            x0 = np.clip([atm_w * 0.5, b0, rho0, 0.0, sig0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            try:
                r = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.01, max_nfev=4000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return None if best is None else best.x


def fit_sabr(F, strikes, iv, T, weights):
    def model(p):
        return sabr_hagan_iv(F, strikes, T, p[0], 1.0, p[1], p[2])

    atm = float(iv[np.argmin(np.abs(strikes - F))])

    def resid(p):
        m = model(p)
        m[~np.isfinite(m)] = 5.0
        return (m - iv) * weights

    best = None
    for rho0, nu0 in ((-0.4, 1.0), (-0.1, 3.0), (-0.7, 0.5)):
        try:
            r = least_squares(resid, [atm, rho0, nu0], bounds=([1e-4, -0.999, 1e-4], [5.0, 0.999, 25.0]),
                              loss="soft_l1", f_scale=0.01, max_nfev=3000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return None if best is None else best.x


def sabr_curve(F, k, T, params):
    a, rho, nu = params
    return sabr_hagan_iv(F, F * np.exp(np.atleast_1d(k)), T, a, 1.0, rho, nu)
