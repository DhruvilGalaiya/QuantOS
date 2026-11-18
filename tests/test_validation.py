"""Data checks reject bad input before it is written."""

import pandas as pd
import pytest

from quantos_platform.core.validation import DataValidationError, check_ohlc, check_surface, require


def bars(n=5):
    ts = pd.date_range("2026-11-02", periods=n, freq="B", tz="Asia/Kolkata")
    close = [25000 + 50 * i for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "open": close, "high": [c + 80 for c in close],
                         "low": [c - 80 for c in close], "close": close})


def test_good_bars_pass():
    assert check_ohlc(bars()) == []


def test_duplicates_order_and_gaps_are_caught():
    b = bars()
    assert any("duplicate" in p for p in check_ohlc(pd.concat([b, b.iloc[[-1]]])))
    assert any("order" in p for p in check_ohlc(b.iloc[::-1]))
    bad = b.copy()
    bad.loc[2, "high"] = bad.loc[2, "close"] - 500
    assert any("high/low" in p for p in check_ohlc(bad))
    jump = b.copy()
    jump.loc[3, ["open", "high", "low", "close"]] = 40000
    assert any("move" in p for p in check_ohlc(jump))


def test_require_raises():
    with pytest.raises(DataValidationError):
        require(["x"], "thing")


def test_surface_check():
    assert check_surface({"expiries": []}) == ["no fitted expiries"]
    assert check_surface({"expiries": [{"expiry": "x", "atm_iv": 0.14, "svi_rmse": 0.002}]}) == []
    assert check_surface({"expiries": [{"expiry": "x", "atm_iv": 5.0, "svi_rmse": 0.002}]})
