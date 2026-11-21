"""Unit tests for the descriptive analytics (synthetic data, no local data needed)."""

import numpy as np
import pandas as pd
import pytest

from quantos_platform.analytics import events, volatility


def gbm_ohlc(n=1500, sigma=0.20, seed=1, steps=78):
    """Daily OHLC from an intraday GBM path (78 five-minute steps per day, no overnight gap)."""
    rng = np.random.default_rng(seed)
    dt = 1 / (252 * steps)
    log_p = np.cumsum(rng.normal(-0.5 * sigma**2 * dt, sigma * np.sqrt(dt), n * steps)) + np.log(100)
    days = log_p.reshape(n, steps)
    prev_close = np.r_[np.log(100), days[:-1, -1]]
    o = prev_close
    h = np.maximum(days.max(axis=1), o)
    low = np.minimum(days.min(axis=1), o)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"open": np.exp(o), "high": np.exp(h), "low": np.exp(low), "close": np.exp(days[:, -1])},
                        index=idx)


@pytest.mark.parametrize("method", ["close", "parkinson", "garman_klass", "yang_zhang"])
def test_realized_vol_recovers_sigma(method):
    df = gbm_ohlc()
    rv = volatility.realized_vol(df, window=63, method=method).dropna()
    # discrete sampling biases range estimators slightly low; allow 15%
    assert rv.mean() == pytest.approx(0.20, rel=0.15)


def test_vol_cone_percentile_bounds():
    df = gbm_ohlc(n=800)
    cone = volatility.vol_cone(df["close"], horizons=(10, 21, 63))
    assert [c["horizon"] for c in cone] == [10, 21, 63]
    for c in cone:
        assert c["min"] <= c["p10"] <= c["p50"] <= c["p90"] <= c["max"]
        assert 0 <= c["percentile"] <= 1


def test_forward_realized_uses_future_only():
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 100))))
    fwd = volatility.forward_realized(close, h=5)
    r = np.log(close).diff()
    assert fwd.iloc[10] == pytest.approx(r.iloc[11:16].std() * np.sqrt(252))
    assert fwd.iloc[-5:].isna().all()


def test_hurst_near_half_for_brownian_log_vol():
    rng = np.random.default_rng(3)
    n = 4000
    log_sigma = np.log(0.15) + np.cumsum(rng.normal(0, 0.03, n))
    idx = pd.bdate_range("2010-01-01", periods=n)
    sigma_d = np.exp(log_sigma) / np.sqrt(252)
    rng_hl = sigma_d * 2   # noise-free range proxy: log range moves exactly with log sigma
    df = pd.DataFrame({"open": 100.0, "close": 100.0, "high": 100 * np.exp(rng_hl / 2),
                       "low": 100 * np.exp(-rng_hl / 2)}, index=idx)
    h = volatility.hurst_log_vol(df, smooth=1, lags=range(1, 40))["H"]
    assert 0.35 < h < 0.65


def test_decluster_respects_gap():
    idx = pd.bdate_range("2020-01-01", periods=30)
    flags = pd.Series(False, index=idx)
    flags.iloc[[0, 2, 5, 12, 13, 25]] = True
    kept = events.decluster(flags, min_gap=10)
    assert list(kept) == [idx[0], idx[12], idx[25]]


def test_forward_returns_start_after_event_close():
    close = pd.Series([100, 110, 121, 133.1], index=pd.bdate_range("2021-01-01", periods=4))
    fwd = events.forward_returns(close, horizons=(1, 2))
    assert fwd["fwd_1"].iloc[0] == pytest.approx(0.10)
    assert fwd["fwd_2"].iloc[0] == pytest.approx(0.21)
    assert np.isnan(fwd["fwd_1"].iloc[-1])


def test_bootstrap_ci_brackets_mean_and_pvalue_detects_shift():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.01, 5000)
    x = rng.normal(0.01, 0.01, 60)
    lo, hi = events.bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert events.bootstrap_pvalue(x, base) < 0.01
    assert events.bootstrap_pvalue(rng.choice(base, 60), base) > 0.01


def test_event_rules_use_no_future_data():
    df = gbm_ohlc(n=700)
    df["vix"] = 0.15 + 0.0 * df["close"]
    full = events.detect(df)
    cut = df.iloc[:500]
    partial = events.detect(cut)
    for eid, dates in partial.items():
        if eid in ("down_weeks_4", "up_weeks_4"):   # the last partial week is held back on purpose
            dates = dates[dates < cut.index[-5]]
        assert set(dates) <= set(full[eid]), eid


def test_straddle_decay_theory_is_monotone_in_time():
    # textbook decay share grows as expiry approaches
    vals = [1 - np.sqrt((t - 1 + 0.04) / (t + 0.04)) for t in (5, 4, 3, 2, 1)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
