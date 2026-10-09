from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .multi_market_returns import build_raw_return_matrix
from .allocation_engine import risk_parity


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

HMM_SIGNAL_PATH = (
    ROOT / "data" / "regime" / "portfolio_hmm_signals.parquet"
)

VOL_SIGNAL_PATH = (
    ROOT / "data" / "regime" / "historical_volatility_signals.parquet"
)

OUTPUT_PATH = (
    ROOT / "data" / "regime" / "adaptive_allocation_signals.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_WINDOW = 504

TRANSACTION_COST = 0.001

# Base risky exposure by current market regime.
REGIME_RISK_BUDGET = {
    "BULL": 1.00,
    "SIDE": 0.80,
    "BEAR": 0.60,
}

# Neutral exposure used when regime confidence is low.
NEUTRAL_RISK_BUDGET = 0.80

# Long-run/reference annualized volatility target.
VOL_TARGET = 0.15

# Prevent the volatility scaler from becoming extreme.
MIN_VOL_MULTIPLIER = 0.50
MAX_VOL_MULTIPLIER = 1.00


# ============================================================
# TIMEZONE NORMALIZATION
# ============================================================

def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Convert timestamps to pandas datetime and remove timezone metadata
    without shifting the calendar date/time.

    This is important because:
      - Upstox-derived data can be timezone-aware (+05:30)
      - portfolio price data is timezone-naive
      - comparisons between aware and naive timestamps fail
    """

    series = pd.to_datetime(series, errors="coerce")

    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_localize(None)

    return series


def normalize_datetime_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Normalize a DatetimeIndex to timezone-naive timestamps while
    preserving the original calendar date/time.
    """

    index = pd.to_datetime(index)

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    return index


# ============================================================
# DATA LOADING
# ============================================================

def load_hmm_signals() -> pd.DataFrame:
    if not HMM_SIGNAL_PATH.exists():
        raise FileNotFoundError(
            f"HMM signal file not found:\n{HMM_SIGNAL_PATH}"
        )

    hmm = pd.read_parquet(HMM_SIGNAL_PATH)

    if "timestamp" not in hmm.columns:
        raise ValueError(
            "HMM signal file does not contain a 'timestamp' column."
        )

    hmm["timestamp"] = normalize_timestamp_series(hmm["timestamp"])

    hmm = hmm.dropna(subset=["timestamp"]).copy()

    hmm = (
        hmm.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return hmm


def load_volatility_signals() -> pd.DataFrame:
    if not VOL_SIGNAL_PATH.exists():
        raise FileNotFoundError(
            f"Historical volatility signal file not found:\n{VOL_SIGNAL_PATH}"
        )

    volatility = pd.read_parquet(VOL_SIGNAL_PATH)

    if "timestamp" not in volatility.columns:
        raise ValueError(
            "Volatility signal file does not contain a 'timestamp' column."
        )

    volatility["timestamp"] = normalize_timestamp_series(
        volatility["timestamp"]
    )

    volatility = volatility.dropna(subset=["timestamp"]).copy()

    volatility = (
        volatility.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return volatility


def load_returns() -> pd.DataFrame:
    returns = build_raw_return_matrix()

    returns.index = normalize_datetime_index(returns.index)

    returns = returns.sort_index()

    returns = returns[~returns.index.duplicated(keep="last")]

    return returns


# ============================================================
# VALIDATION
# ============================================================

def validate_signal_columns(
    hmm: pd.DataFrame,
    volatility: pd.DataFrame,
) -> None:

    required_hmm = {
        "timestamp",
        "current_regime",
        "next_regime",
        "next_bull_probability",
        "next_side_probability",
        "next_bear_probability",
    }

    missing_hmm = required_hmm - set(hmm.columns)

    if missing_hmm:
        raise ValueError(
            "HMM signal file is missing required columns: "
            + ", ".join(sorted(missing_hmm))
        )

    required_volatility = {
        "timestamp",
        "nifty_50_garch_21d",
        "nifty_bank_garch_21d",
        "sensex_garch_21d",
    }

    missing_volatility = required_volatility - set(volatility.columns)

    if missing_volatility:
        raise ValueError(
            "Volatility signal file is missing required columns: "
            + ", ".join(sorted(missing_volatility))
        )


# ============================================================
# REGIME CONFIDENCE
# ============================================================

def get_next_regime_probability(row: pd.Series) -> float:
    """
    Extract the probability that the market transitions into the
    currently predicted next regime.
    """

    next_regime = str(row["next_regime"]).upper()

    if next_regime == "BULL":
        return float(row["next_bull_probability"])

    if next_regime == "SIDE":
        return float(row["next_side_probability"])

    if next_regime == "BEAR":
        return float(row["next_bear_probability"])

    raise ValueError(
        f"Unknown next_regime: {next_regime}"
    )


def confidence_adjusted_budget(
    current_regime: str,
    next_regime_probability: float,
) -> float:
    """
    Move between the regime-specific risk budget and neutral exposure
    according to confidence in the next-regime forecast.

    confidence = 0  -> neutral exposure
    confidence = 1  -> full regime-specific exposure
    """

    current_regime = str(current_regime).upper()

    if current_regime not in REGIME_RISK_BUDGET:
        raise ValueError(
            f"Unknown current regime: {current_regime}"
        )

    probability = float(
        np.clip(next_regime_probability, 0.0, 1.0)
    )

    regime_budget = REGIME_RISK_BUDGET[current_regime]

    return (
        probability * regime_budget
        + (1.0 - probability) * NEUTRAL_RISK_BUDGET
    )


# ============================================================
# VOLATILITY SCALING
# ============================================================

def calculate_volatility_multiplier(row: pd.Series) -> float:
    """
    Calculate a transparent volatility scaling multiplier using the
    three market-level 21-day GARCH forecasts.

    The multiplier is clipped so that the strategy cannot become
    excessively aggressive simply because forecast volatility is low.
    """

    forecast_columns = [
        "nifty_50_garch_21d",
        "nifty_bank_garch_21d",
        "sensex_garch_21d",
    ]

    forecasts = []

    for column in forecast_columns:
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value) and np.isfinite(value) and value > 0:
            forecasts.append(float(value))

    if not forecasts:
        return 1.0

    average_forecast = float(np.mean(forecasts))

    multiplier = VOL_TARGET / average_forecast

    multiplier = float(
        np.clip(
            multiplier,
            MIN_VOL_MULTIPLIER,
            MAX_VOL_MULTIPLIER,
        )
    )

    return multiplier


# ============================================================
# RISK PARITY
# ============================================================

def calculate_risk_parity_weights(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Calculate long-only risk-parity weights for the available assets.
    """

    covariance = covariance.copy()

    covariance = covariance.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    covariance = covariance.dropna(
        axis=0,
        how="all",
    ).dropna(
        axis=1,
        how="all",
    )

    # Keep only assets available in both dimensions.
    common_assets = [
        asset
        for asset in covariance.index
        if asset in covariance.columns
    ]

    covariance = covariance.loc[
        common_assets,
        common_assets,
    ]

    covariance = covariance.fillna(0.0)

    # Small diagonal regularization improves numerical stability.
    covariance_values = covariance.to_numpy(dtype=float)

    covariance_values = (
        covariance_values
        + np.eye(len(covariance_values)) * 1e-10
    )

    covariance = pd.DataFrame(
        covariance_values,
        index=common_assets,
        columns=common_assets,
    )

    weights = risk_parity(covariance)

    weights = pd.Series(
        weights,
        index=common_assets,
        dtype=float,
    )

    weights = weights.clip(lower=0.0)

    total = float(weights.sum())

    if total <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights /= total

    return weights


# ============================================================
# SINGLE ALLOCATION
# ============================================================

def calculate_allocation(
    target_date: pd.Timestamp,
    current_regime: str,
    next_regime_probability: float,
    volatility_row: pd.Series,
    returns: pd.DataFrame,
) -> tuple[pd.Series, float, float]:

    # --------------------------------------------------------
    # Historical data only
    # --------------------------------------------------------

    historical_returns = returns.loc[
        returns.index <= target_date
    ].copy()

    if len(historical_returns) < TRAIN_WINDOW:
        raise ValueError(
            f"Insufficient training data on {target_date.date()}. "
            f"Required {TRAIN_WINDOW}, "
            f"available {len(historical_returns)}."
        )

    historical_returns = historical_returns.tail(
        TRAIN_WINDOW
    )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    covariance = (
        historical_returns
        .cov()
        * 252.0
    )

    # Remove assets without sufficient covariance data.
    valid_assets = [
        column
        for column in covariance.columns
        if covariance[column].notna().all()
        and np.isfinite(covariance[column]).all()
    ]

    covariance = covariance.loc[
        valid_assets,
        valid_assets,
    ]

    if covariance.empty:
        raise ValueError(
            f"No valid covariance matrix on {target_date.date()}."
        )

    # --------------------------------------------------------
    # Base risky portfolio
    # --------------------------------------------------------

    risky_weights = calculate_risk_parity_weights(
        covariance
    )

    # --------------------------------------------------------
    # Regime exposure
    # --------------------------------------------------------

    budget = confidence_adjusted_budget(
        current_regime=current_regime,
        next_regime_probability=next_regime_probability,
    )

    # --------------------------------------------------------
    # Volatility scaling
    # --------------------------------------------------------

    volatility_multiplier = calculate_volatility_multiplier(
        volatility_row
    )

    risky_exposure = (
        budget
        * volatility_multiplier
    )

    risky_exposure = float(
        np.clip(
            risky_exposure,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Final allocation
    # --------------------------------------------------------

    weights = risky_weights * risky_exposure

    weights["CASH"] = 1.0 - float(weights.sum())

    weights = weights.clip(lower=0.0)

    weights = weights / weights.sum()

    return (
        weights,
        float(risky_exposure),
        float(volatility_multiplier),
    )


# ============================================================
# TURNOVER
# ============================================================

def calculate_turnover(
    previous_weights: pd.Series | None,
    current_weights: pd.Series,
) -> float:

    if previous_weights is None:
        # Starting from cash.
        return 1.0

    all_assets = sorted(
        set(previous_weights.index)
        | set(current_weights.index)
    )

    previous = previous_weights.reindex(
        all_assets,
        fill_value=0.0,
    )

    current = current_weights.reindex(
        all_assets,
        fill_value=0.0,
    )

    return float(
        0.5 * np.abs(
            current - previous
        ).sum()
    )


# ============================================================
# MAIN ENGINE
# ============================================================

def run_adaptive_allocation() -> pd.DataFrame:

    print("=" * 70)
    print("QuantOS ADAPTIVE ALLOCATION ENGINE")
    print("=" * 70)

    print("\nLoading portfolio returns...")
    returns = load_returns()

    print(
        f"Return matrix: "
        f"{returns.shape[0]} rows x "
        f"{returns.shape[1]} assets"
    )

    print(
        f"Return period: "
        f"{returns.index.min().date()} -> "
        f"{returns.index.max().date()}"
    )

    print("\nLoading HMM signals...")
    hmm = load_hmm_signals()

    print(
        f"HMM signals: {len(hmm)} rows"
    )

    print(
        f"HMM period: "
        f"{hmm['timestamp'].min().date()} -> "
        f"{hmm['timestamp'].max().date()}"
    )

    print("\nLoading historical volatility signals...")
    volatility = load_volatility_signals()

    print(
        f"Volatility signals: "
        f"{len(volatility)} rows"
    )

    print(
        f"Volatility period: "
        f"{volatility['timestamp'].min().date()} -> "
        f"{volatility['timestamp'].max().date()}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    validate_signal_columns(
        hmm,
        volatility,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Normalize timestamps AFTER both DataFrames exist.
    # --------------------------------------------------------

    hmm["timestamp"] = normalize_timestamp_series(
        hmm["timestamp"]
    )

    volatility["timestamp"] = normalize_timestamp_series(
        volatility["timestamp"]
    )

    returns.index = normalize_datetime_index(
        returns.index
    )

    # --------------------------------------------------------
    # Merge HMM + volatility signals
    # --------------------------------------------------------

    signals = pd.merge(
        hmm,
        volatility,
        on="timestamp",
        how="inner",
        suffixes=("", "_vol"),
    )

    signals = (
        signals
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if signals.empty:
        raise ValueError(
            "No overlapping dates between HMM and volatility signals."
        )

    print(
        f"\nAligned signal dates: "
        f"{len(signals)}"
    )

    print(
        f"Aligned period: "
        f"{signals['timestamp'].min().date()} -> "
        f"{signals['timestamp'].max().date()}"
    )

    # --------------------------------------------------------
    # Build allocations
    # --------------------------------------------------------

    records = []

    previous_weights = None

    for _, row in signals.iterrows():

        target_date = pd.Timestamp(
            row["timestamp"]
        )

        # Extra safety in case a timezone somehow survives.
        if target_date.tzinfo is not None:
            target_date = target_date.tz_localize(None)

        current_regime = str(
            row["current_regime"]
        ).upper()

        next_regime = str(
            row["next_regime"]
        ).upper()

        next_regime_probability = (
            get_next_regime_probability(row)
        )

        weights, risky_exposure, volatility_multiplier = (
            calculate_allocation(
                target_date=target_date,
                current_regime=current_regime,
                next_regime_probability=next_regime_probability,
                volatility_row=row,
                returns=returns,
            )
        )

        turnover = calculate_turnover(
            previous_weights,
            weights,
        )

        transaction_cost = (
            turnover
            * TRANSACTION_COST
        )

        # ----------------------------------------------------
        # Base record
        # ----------------------------------------------------

        record = {
            "timestamp": target_date,
            "current_regime": current_regime,
            "next_regime": next_regime,
            "next_regime_probability": next_regime_probability,
            "risky_exposure": risky_exposure,
            "volatility_multiplier": volatility_multiplier,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
        }

        # ----------------------------------------------------
        # Store every asset weight
        # ----------------------------------------------------

        for asset in weights.index:
            record[f"weight_{asset}"] = float(
                weights[asset]
            )

        # ----------------------------------------------------
        # Store regime probabilities
        # ----------------------------------------------------

        record["next_bull_probability"] = float(
            row["next_bull_probability"]
        )

        record["next_side_probability"] = float(
            row["next_side_probability"]
        )

        record["next_bear_probability"] = float(
            row["next_bear_probability"]
        )

        # ----------------------------------------------------
        # Store volatility forecasts
        # ----------------------------------------------------

        record["nifty_50_garch_21d"] = float(
            row["nifty_50_garch_21d"]
        )

        record["nifty_bank_garch_21d"] = float(
            row["nifty_bank_garch_21d"]
        )

        record["sensex_garch_21d"] = float(
            row["sensex_garch_21d"]
        )

        records.append(record)

        previous_weights = weights.copy()

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    output = pd.DataFrame(records)

    output = (
        output
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    weight_columns = [
        column
        for column in output.columns
        if column.startswith("weight_")
    ]

    output["weight_sum"] = output[
        weight_columns
    ].sum(axis=1)

    max_weight_error = float(
        np.abs(
            output["weight_sum"] - 1.0
        ).max()
    )

    if max_weight_error > 1e-8:
        raise ValueError(
            "Portfolio weights do not sum to 1. "
            f"Maximum error: {max_weight_error}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    latest = output.iloc[-1]

    print("\n" + "=" * 70)
    print("ADAPTIVE ALLOCATION COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput rows: {len(output)}"
    )

    print(
        f"Output period: "
        f"{output['timestamp'].min().date()} -> "
        f"{output['timestamp'].max().date()}"
    )

    print(
        f"\nLatest signal date: "
        f"{latest['timestamp'].date()}"
    )

    print(
        f"Current regime: "
        f"{latest['current_regime']}"
    )

    print(
        f"Next regime: "
        f"{latest['next_regime']}"
    )

    print(
        f"Next-regime probability: "
        f"{latest['next_regime_probability']:.4f}"
    )

    print(
        f"Risky exposure: "
        f"{latest['risky_exposure']:.4f}"
    )

    print(
        f"Volatility multiplier: "
        f"{latest['volatility_multiplier']:.4f}"
    )

    print("\nLatest portfolio weights:")

    latest_weights = (
        latest[weight_columns]
        .sort_values(
            ascending=False
        )
    )

    for column, value in latest_weights.items():
        asset = column.replace(
            "weight_",
            "",
        )

        print(
            f"  {asset:<20} "
            f"{float(value):.4f}"
        )

    print(
        f"\nLatest turnover: "
        f"{latest['turnover']:.4f}"
    )

    print(
        f"Latest transaction cost: "
        f"{latest['transaction_cost']:.6f}"
    )

    print(
        f"\nMaximum weight-sum error: "
        f"{max_weight_error:.2e}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    print("=" * 70)

    return output


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_adaptive_allocation()