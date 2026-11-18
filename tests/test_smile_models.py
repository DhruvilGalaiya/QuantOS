"""SVI fitting / arbitrage checks and the SABR formula (vs the original research implementation)."""

import numpy as np
import pytest

from quantos_platform.vol.smile_models import durrleman_g, fit_svi, sabr_hagan_iv, svi_w


def test_svi_fit_recovers_a_known_smile():
    T = 20 / 365
    true = (0.0004, 0.02, -0.5, 0.01, 0.05)          # a, b, rho, m, sigma
    k = np.linspace(-0.08, 0.06, 40)
    iv = np.sqrt(svi_w(k, *true) / T)
    fitted = fit_svi(k, iv, T, np.ones_like(k))
    iv_fit = np.sqrt(np.maximum(svi_w(k, *fitted), 1e-12) / T)
    assert np.max(np.abs(iv_fit - iv)) < 5e-4          # < 0.05 vol points


def test_durrleman_flags_butterfly_arbitrage():
    k = np.linspace(-1.5, 1.5, 601)
    clean = (0.02, 0.1, -0.4, 0.0, 0.2)
    # Gatheral & Jacquier (2014) example of an SVI slice WITH butterfly arbitrage
    arb = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    assert durrleman_g(k, clean).min() >= 0
    assert durrleman_g(k, arb).min() < 0


def test_sabr_matches_original_implementation():
    from quantos_platform.legacy import legacy_module

    legacy = legacy_module("quantos_volatility.sabr_surface").sabr_implied_volatility
    F, T = 25000.0, 25 / 365
    for alpha, rho, nu in [(0.15, -0.4, 1.0), (0.2, -0.1, 3.0), (0.12, 0.3, 0.5)]:
        K = np.array([22000, 23500, 24900, 25000, 25100, 26500, 28000], dtype=float)
        ours = sabr_hagan_iv(F, K, T, alpha, 1.0, rho, nu)
        theirs = np.array([legacy(F, k, T, alpha, 1.0, rho, nu) for k in K])
        assert ours == pytest.approx(theirs, rel=1e-9)
