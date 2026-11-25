"""Market States and Risk Monitor building blocks (synthetic data, no local data needed)."""

import numpy as np
import pandas as pd
import pytest

from quantos_platform.analytics import risk, states


def _probs(rows):
    idx = pd.bdate_range("2020-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["p_calm", "p_normal", "p_stressed"])


def test_label_hysteresis_ignores_weak_flips():
    p = _probs([[0.9, 0.05, 0.05], [0.4, 0.55, 0.05], [0.3, 0.65, 0.05], [0.5, 0.45, 0.05]])
    assert list(states.label(p, switch=0.6)) == ["CALM", "CALM", "NORMAL", "NORMAL"]


def test_label_is_causal():
    rng = np.random.default_rng(0)
    raw = rng.dirichlet([1, 1, 1], size=200)
    full = states.label(_probs(raw))
    part = states.label(_probs(raw[:120]))
    assert (full.iloc[:120].to_numpy() == part.to_numpy()).all()


def test_survival_and_transitions():
    s = pd.Series(["CALM"] * 5 + ["NORMAL"] * 3 + ["CALM"] * 2 + ["STRESSED"] * 4,
                  index=pd.bdate_range("2020-01-01", periods=14))
    ep = states.episodes(s)
    assert list(ep["days"]) == [5, 3, 2, 4] and bool(ep["ongoing"].iloc[-1])
    surv = states.survival(ep, "CALM", kmax=6)
    assert surv[0] == 1.0 and surv[2] == pytest.approx(0.5) and surv[5] == 0.0
    m = states.transition_matrix(s)
    assert np.allclose(np.nansum(m, axis=1), 1) and m[0, 0] == pytest.approx(5 / 7)


def test_drawdown_episodes_and_recovery():
    close = pd.Series([100, 110, 99, 88, 95, 112, 120, 108, 115], index=pd.bdate_range("2020-01-01", periods=9),
                      dtype=float)
    ep = states.drawdown_episodes(close, min_depth=0.05)
    first, last = ep.iloc[0], ep.iloc[-1]
    assert first["depth"] == pytest.approx(88 / 110 - 1) and first["days_to_recover"] == 2
    assert bool(last["ongoing"]) and last["depth"] == pytest.approx(108 / 120 - 1)


def test_kupiec_and_hill():
    assert risk.kupiec(1000, 50, 0.05) > 0.5            # exactly on target
    assert risk.kupiec(1000, 100, 0.05) < 0.001         # twice too many breaches
    rng = np.random.default_rng(1)
    pareto = rng.pareto(3.0, 200_000) + 1                 # tail index 3
    assert risk.hill_tail_index(pareto, 0.01) == pytest.approx(3.0, rel=0.15)
