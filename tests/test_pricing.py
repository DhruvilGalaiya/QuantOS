"""Black-76: put-call parity, IV inversion round-trips, and no-arbitrage bounds."""

import numpy as np
import pytest

from quantos_platform.vol.pricing import black76, implied_vol


@pytest.mark.parametrize("K", [22000, 24000, 25000, 26000, 28000])
@pytest.mark.parametrize("T", [1 / 365, 7 / 365, 30 / 365, 1.0])
def test_put_call_parity(K, T):
    F, df, s = 25000.0, np.exp(-0.065 * T), 0.15
    c = float(black76(F, K, T, df, s, True))
    p = float(black76(F, K, T, df, s, False))
    assert c - p == pytest.approx(df * (F - K), abs=1e-6)


@pytest.mark.parametrize("sigma", [0.05, 0.13, 0.30, 0.80])
@pytest.mark.parametrize("K,is_call", [(24000, False), (25000, True), (25000, False), (26500, True)])
def test_iv_round_trip(sigma, K, is_call):
    F, T, df = 25000.0, 10 / 365, 0.998
    price = float(black76(F, K, T, df, sigma, is_call))
    if price < 1e-6:
        pytest.skip("price below numerical resolution")
    assert implied_vol(price, F, K, T, df, is_call) == pytest.approx(sigma, abs=1e-5)


def test_iv_rejects_prices_outside_bounds():
    F, K, T, df = 25000.0, 24000.0, 10 / 365, 0.998
    intrinsic = df * (F - K)
    assert np.isnan(implied_vol(intrinsic * 0.99, F, K, T, df, True))   # below intrinsic
    assert np.isnan(implied_vol(df * F * 1.01, F, K, T, df, True))      # above the forward
    assert np.isnan(implied_vol(0.0, F, K, T, df, True))
