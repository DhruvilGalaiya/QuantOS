"""Session counting for expiry logic (holidays mocked: no network)."""

import pandas as pd

from quantos_platform.core import calendar


def test_sessions_to_expiry_with_holiday(monkeypatch):
    monkeypatch.setattr(calendar, "trading_holidays", lambda: {pd.Timestamp("2026-11-24")})
    known = pd.bdate_range("2026-11-02", "2026-11-18")
    # Wed 18 Nov -> expiry Mon 23 Nov: Thu, Fri, Mon = 3 sessions
    assert calendar.sessions_to_expiry(pd.Timestamp("2026-11-18"), pd.Timestamp("2026-11-23"), known) == 3
    # expiry the next session
    assert calendar.sessions_to_expiry(pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-13"), known) == 1
    # a holiday is not a session
    assert calendar.future_sessions(pd.Timestamp("2026-11-23"), pd.Timestamp("2026-11-25")) == \
        [pd.Timestamp("2026-11-25")]
