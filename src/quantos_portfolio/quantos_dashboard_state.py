"""
QuantOS Dashboard State Builder

Combines:
1. Unified market state
2. Portfolio allocation/backtest results
3. HMM conditional allocation results

Purpose:
Create one clean JSON/CSV-ready state object for the QuantOS dashboard.

This module DOES NOT:
- retrain the HMM
- modify portfolio weights
- run a new backtest
- make investment recommendations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNIFIED_STATE = (
    PROJECT_ROOT
    / "data"
    / "portfolio"
    / "unified_market_state"
    / "current_market_state.csv"
)

HMM_ALLOCATION_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "portfolio"
    / "hmm_allocation"
    / "hmm_allocation_summary.csv"
)

HMM_ALLOCATION_REBALANCES = (
    PROJECT_ROOT
    / "data"
    / "portfolio"
    / "hmm_allocation"
    / "hmm_allocation_rebalances.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "portfolio"
    / "dashboard_state"
)

OUTPUT_JSON = OUTPUT_DIR / "quantos_dashboard_state.json"
OUTPUT_CSV = OUTPUT_DIR / "quantos_dashboard_state.csv"

MONTE_CARLO_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "portfolio"
    / "monte_carlo_outputs"
    / "nifty_50_monte_carlo_output.json"
)

# ============================================================
# HELPERS
# ============================================================

def clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def load_csv(path: Path, name: str) -> pd.DataFrame:
    """Load a CSV with an explicit error message."""

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}\n"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{name} exists but contains no rows:\n{path}")

    return df


def first_row_as_dict(df: pd.DataFrame) -> dict:
    """Return the first dataframe row as a clean dictionary."""

    row = df.iloc[0].to_dict()

    return {
        str(k): clean_value(v)
        for k, v in row.items()
    }


def normalize_percentage(value):
    """Convert decimal percentage values into percentage points."""

    if value is None:
        return None

    return float(value) * 100.0


# ============================================================
# MARKET STATE
# ============================================================

def build_market_state() -> dict:
    """Load the unified QuantOS market state using the exact CSV schema."""

    df = load_csv(
        UNIFIED_STATE,
        "Unified market state",
    )

    # The unified CSV has exactly one current-state row.
    row = df.iloc[0].to_dict()

    def get(name):
        """Safely retrieve a value from the unified state."""

        value = row.get(name)

        if pd.isna(value):
            return None

        return clean_value(value)

    market = {
        "asset": get("asset"),

        "hmm_date": get("hmm_date"),

        "regime": get("hmm_regime"),

        "regime_probability": {
            "bull": get("hmm_p_bull"),
            "side": get("hmm_p_side"),
            "bear": get("hmm_p_bear"),
        },

        "hmm_current_volatility": {
            "vol_5d": get("hmm_current_vol_5d"),
            "vol_20d": get("hmm_current_vol_20d"),
            "vol_60d": get("hmm_current_vol_60d"),
        },

        "realized_volatility": {
            "vol_21d": get("realized_vol_21d"),
            "long_run_vol_21d": get("long_run_vol_21d"),
            "volatility_ratio": get("realized_vol_ratio"),
            "volatility_level": get("realized_vol_level"),
        },

        "forecast_volatility": {
            "garch_1d": get("garch_1d"),
            "garch_5d": get("garch_5d"),
            "garch_21d": get("garch_21d"),
            "har_1d": get("har_1d"),
        },

        "options": {
            "timestamp": get("options_generated_at"),
            "spot": get("options_spot"),
            "atm_strike": get("atm_strike"),
            "nearest_expiry": get("nearest_expiry"),
            "next_expiry": get("next_expiry"),
            "atm_iv_nearest": get("atm_iv_nearest"),
            "atm_iv_next": get("atm_iv_next"),
            "iv_realized_ratio": get("iv_rv_ratio"),
            "iv_minus_realized": get("iv_minus_realized"),
            "moneyness_skew": get("moneyness_skew"),
            "term_structure_difference": get(
                "term_structure_difference"
            ),
        },

        "sabr": {
            "sabr_iv": get("sabr_iv"),
            "alpha": get("sabr_alpha"),
            "beta": get("sabr_beta"),
            "rho": get("sabr_rho"),
            "nu": get("sabr_nu"),
        },

        "risk_state": {
            "regime_conditioned_21d_vol": get(
                "hmm_regime_conditioned_21d_vol"
            ),

            "realized_vol_state": get(
                "realized_vol_state"
            ),

            "iv_vs_realized": get(
                "implied_vs_realized_state"
            ),

            "term_structure_state": get(
                "term_structure_state"
            ),
        },

        "market_data": {
            "volatility_model_date": get(
                "volatility_model_date"
            ),

            "latest_close": get(
                "latest_close"
            ),
        },
    }

    return market

# ============================================================
# PORTFOLIO RESULTS
# ============================================================

def build_portfolio_state() -> dict:
    """Load the portfolio strategy comparison."""

    df = load_csv(
        HMM_ALLOCATION_SUMMARY,
        "HMM allocation summary",
    )

    # Normalize column names.
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Locate strategy column.
    strategy_column = None

    for candidate in [
        "Strategy",
        "strategy",
        "strategy_name",
    ]:
        if candidate in df.columns:
            strategy_column = candidate
            break

    if strategy_column is None:
        raise ValueError(
            "Could not find strategy column in "
            f"{HMM_ALLOCATION_SUMMARY}"
        )

    strategies = {}

    for _, row in df.iterrows():

        strategy_name = str(row[strategy_column])

        metrics = {}

        for column in df.columns:

            if column == strategy_column:
                continue

            value = clean_value(row[column])

            if value is not None:
                metrics[column.lower().replace(" ", "_")] = value

        strategies[strategy_name] = metrics

    return {
        "strategies": strategies,
        "source": str(
            HMM_ALLOCATION_SUMMARY.relative_to(PROJECT_ROOT)
        ),
    }


def load_monte_carlo_state() -> dict:
    """Load the latest NIFTY 50 Monte Carlo output."""

    if not MONTE_CARLO_OUTPUT.exists():
        print(
            f"WARNING: Monte Carlo output not found: "
            f"{MONTE_CARLO_OUTPUT}"
        )
        return {}

    with open(
        MONTE_CARLO_OUTPUT,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data

# ============================================================
# HMM REBALANCE STATE
# ============================================================

def build_rebalance_state() -> dict:
    """Summarize the HMM conditional allocation rebalances."""

    if not HMM_ALLOCATION_REBALANCES.exists():
        return {
            "available": False,
            "rows": 0,
        }

    df = pd.read_csv(HMM_ALLOCATION_REBALANCES)

    if df.empty:
        return {
            "available": False,
            "rows": 0,
        }

    result = {
        "available": True,
        "rows": len(df),
    }

    # Try to identify useful columns without assuming
    # an exact schema.

    date_columns = [
        c for c in df.columns
        if "date" in c.lower()
    ]

    regime_columns = [
        c for c in df.columns
        if "regime" in c.lower()
    ]

    if date_columns:
        result["first_rebalance"] = clean_value(
            df.iloc[0][date_columns[0]]
        )

        result["last_rebalance"] = clean_value(
            df.iloc[-1][date_columns[0]]
        )

    if regime_columns:
        regime_col = regime_columns[0]

        counts = (
            df[regime_col]
            .value_counts()
            .to_dict()
        )

        result["regime_rebalance_counts"] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    return result


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

def build_summary(market: dict, portfolio: dict) -> dict:
    """
    Build concise values used by the dashboard header/cards.

    These are descriptive summaries only.
    """

    strategies = portfolio.get("strategies", {})

    summary = {
        "market_regime": market.get("regime"),

        "market_regime_probability": (
            market
            .get("regime_probability", {})
            .get(
                str(market.get("regime", "")).lower()
            )
        ),

        "realized_volatility_state": (
            market
            .get("risk_state", {})
            .get("realized_vol_state")
        ),

        "iv_vs_realized": (
            market
            .get("risk_state", {})
            .get("iv_vs_realized")
        ),

        "term_structure_state": (
            market
            .get("risk_state", {})
            .get("term_structure_state")
        ),

        "strategies_available": list(
            strategies.keys()
        ),
    }

    return summary


# ============================================================
# BUILD COMPLETE STATE
# ============================================================

def build_dashboard_state() -> dict:

    print("=" * 70)
    print("QUANTOS DASHBOARD STATE BUILDER")
    print("=" * 70)

    print("\n[1/5] Loading unified market state...")
    market = build_market_state()

    print("[2/5] Loading portfolio results...")
    portfolio = build_portfolio_state()

    print("[3/5] Loading allocation rebalance state...")
    rebalances = build_rebalance_state()

    print("[4/5] Loading Monte Carlo state...")
    monte_carlo = load_monte_carlo_state()

    print("[5/5] Building dashboard state...")

    summary = build_summary(
    market,
    portfolio,
)

    state = {
        "project": "QuantOS",

        "generated_from": {
            "unified_market_state": str(
                UNIFIED_STATE.relative_to(PROJECT_ROOT)
            ),
            "portfolio_summary": str(
                HMM_ALLOCATION_SUMMARY.relative_to(PROJECT_ROOT)
            ),
        },

        "market": market,

        "portfolio": portfolio,

        "allocation": {
            "hmm_conditional": rebalances,
        },

        "monte_carlo": monte_carlo,

        "summary": summary,
    }

    return state


# ============================================================
# SAVE
# ============================================================

def save_state(state: dict):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # JSON
    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # Compact CSV for simple frontend loading/debugging.
        flat = {
        "asset": state["market"].get("asset"),

        "hmm_date": state["market"].get("hmm_date"),

        "regime": state["market"].get("regime"),

        "p_bull": state["market"]
        ["regime_probability"]
        .get("bull"),

        "p_side": state["market"]
        ["regime_probability"]
        .get("side"),

        "p_bear": state["market"]
        ["regime_probability"]
        .get("bear"),

        "hmm_current_vol_5d": state["market"]
        ["hmm_current_volatility"]
        .get("vol_5d"),

        "hmm_current_vol_20d": state["market"]
        ["hmm_current_volatility"]
        .get("vol_20d"),

        "hmm_current_vol_60d": state["market"]
        ["hmm_current_volatility"]
        .get("vol_60d"),

        "realized_vol_21d": state["market"]
        ["realized_volatility"]
        .get("vol_21d"),

        "long_run_vol_21d": state["market"]
        ["realized_volatility"]
        .get("long_run_vol_21d"),

        "realized_vol_ratio": state["market"]
        ["realized_volatility"]
        .get("volatility_ratio"),

        "realized_vol_level": state["market"]
        ["realized_volatility"]
        .get("volatility_level"),

        "garch_1d": state["market"]
        ["forecast_volatility"]
        .get("garch_1d"),

        "garch_5d": state["market"]
        ["forecast_volatility"]
        .get("garch_5d"),

        "garch_21d": state["market"]
        ["forecast_volatility"]
        .get("garch_21d"),

        "har_1d": state["market"]
        ["forecast_volatility"]
        .get("har_1d"),

        "options_spot": state["market"]
        ["options"]
        .get("spot"),

        "atm_strike": state["market"]
        ["options"]
        .get("atm_strike"),

        "nearest_expiry": state["market"]
        ["options"]
        .get("nearest_expiry"),

        "next_expiry": state["market"]
        ["options"]
        .get("next_expiry"),

        "atm_iv_nearest": state["market"]
        ["options"]
        .get("atm_iv_nearest"),

        "atm_iv_next": state["market"]
        ["options"]
        .get("atm_iv_next"),

        "iv_rv_ratio": state["market"]
        ["options"]
        .get("iv_realized_ratio"),

        "iv_minus_realized": state["market"]
        ["options"]
        .get("iv_minus_realized"),

        "moneyness_skew": state["market"]
        ["options"]
        .get("moneyness_skew"),

        "term_structure_difference": state["market"]
        ["options"]
        .get("term_structure_difference"),

        "sabr_iv": state["market"]
        ["sabr"]
        .get("sabr_iv"),

        "sabr_alpha": state["market"]
        ["sabr"]
        .get("alpha"),

        "sabr_beta": state["market"]
        ["sabr"]
        .get("beta"),

        "sabr_rho": state["market"]
        ["sabr"]
        .get("rho"),

        "sabr_nu": state["market"]
        ["sabr"]
        .get("nu"),

        "regime_conditioned_21d_vol": state["market"]
        ["risk_state"]
        .get("regime_conditioned_21d_vol"),

        "realized_vol_state": state["market"]
        ["risk_state"]
        .get("realized_vol_state"),

        "implied_vs_realized_state": state["market"]
        ["risk_state"]
        .get("iv_vs_realized"),

        "term_structure_state": state["market"]
        ["risk_state"]
        .get("term_structure_state"),
    }

    pd.DataFrame([flat]).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nFILES SAVED")
    print("-" * 70)
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


# ============================================================
# MAIN
# ============================================================

def main():

    state = build_dashboard_state()

    save_state(state)

    print("\n" + "=" * 70)
    print("QUANTOS DASHBOARD STATE COMPLETE")
    print("=" * 70)

    market = state.get("market", {})

    probabilities = market.get(
        "regime_probability",
        {}
    )

    realized_vol = market.get(
        "realized_volatility",
        {}
    )

    options = market.get(
        "options",
        {}
    )

    risk_state = market.get(
        "risk_state",
        {}
    )

    regime = market.get("regime")

    p_bull = probabilities.get("bull")
    p_side = probabilities.get("side")
    p_bear = probabilities.get("bear")

    vol_21d = realized_vol.get(
        "vol_21d"
    )

    atm_iv = options.get(
        "atm_iv_nearest"
    )

    print(
        f"\nRegime      : {regime}"
    )

    print(
        f"P(BULL)     : "
        f"{float(p_bull):.2%}"
    )

    print(
        f"P(SIDE)     : "
        f"{float(p_side):.2%}"
    )

    print(
        f"P(BEAR)     : "
        f"{float(p_bear):.2%}"
    )

    print(
        f"Realized 21D Vol : "
        f"{float(vol_21d):.2%}"
    )

    print(
        f"ATM IV           : "
        f"{float(atm_iv):.2%}"
    )

    print(
        f"Regime-conditioned 21D Vol : "
        f"{float(risk_state.get('regime_conditioned_21d_vol')):.2%}"
    )

    print(
        f"IV vs Realized   : "
        f"{risk_state.get('iv_vs_realized')}"
    )

    print(
        f"Term Structure   : "
        f"{risk_state.get('term_structure_state')}"
    )

    print("\nOutput files verified:")
    print(f"JSON : {OUTPUT_JSON}")
    print(f"CSV  : {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("DASHBOARD STATE READY")
    print("=" * 70)
    

if __name__ == "__main__":
    main()