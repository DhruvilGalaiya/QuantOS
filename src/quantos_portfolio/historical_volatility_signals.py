from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model


warnings.filterwarnings("ignore")


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

HMM_SIGNAL_PATH = (
    ROOT
    / "data"
    / "regime"
    / "portfolio_hmm_signals.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "regime"
    / "historical_volatility_signals.parquet"
)

VOLATILITY_FILES = {
    "NIFTY_50": (
        ROOT
        / "data"
        / "regime"
        / "volatility"
        / "nifty_50.parquet"
    ),
    "NIFTY_BANK": (
        ROOT
        / "data"
        / "regime"
        / "volatility"
        / "nifty_bank.parquet"
    ),
    "SENSEX": (
        ROOT
        / "data"
        / "regime"
        / "volatility"
        / "sensex.parquet"
    ),
}

TRAIN_WINDOW = 504

FORECAST_HORIZON = 21

# GARCH refit frequency is deliberately the same
# as the portfolio rebalance frequency.
REBALANCE_FREQUENCY = 21

MIN_OBSERVATIONS = 400


# ================================================================
# LOAD PRICE DATA
# ================================================================

def load_price_data(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{path.name}: missing timestamp column"
        )

    if "close" not in df.columns:
        raise ValueError(
            f"{path.name}: missing close column"
        )

    df = df[
        [
            "timestamp",
            "close",
        ]
    ].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = (
        df
        .dropna()
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ================================================================
# GARCH FORECAST
# ================================================================

def garch_forecast(
    returns: pd.Series,
) -> tuple[float, float]:
    """
    Fit GARCH(1,1) with Student-t innovations.

    Returns:
        forecast_1d_annualized_vol
        forecast_21d_annualized_vol
    """

    returns = (
        returns
        .dropna()
        .astype(float)
    )

    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            "Insufficient observations for GARCH."
        )

    # arch works more stably when returns are expressed
    # in percentage units.
    scaled_returns = (
        returns * 100.0
    )

    model = arch_model(
        scaled_returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    result = model.fit(
        disp="off"
    )

    forecast = result.forecast(
        horizon=FORECAST_HORIZON,
        reindex=False,
    )

    variance = (
        forecast
        .variance
        .iloc[-1]
        .to_numpy()
    )

    # Convert percentage-squared variance back
    # into decimal return variance.
    variance = (
        variance / 10000.0
    )

    # ------------------------------------------------------------
    # 1-day annualized volatility
    # ------------------------------------------------------------

    one_day_variance = (
        variance[0]
    )

    vol_1d = np.sqrt(
        one_day_variance
        * 252.0
    )

    # ------------------------------------------------------------
    # 21-day annualized volatility
    #
    # Sum the forecast daily variances over 21 days,
    # then annualize.
    # ------------------------------------------------------------

    horizon_variance = (
        variance.sum()
    )

    vol_21d = np.sqrt(
        horizon_variance
        *
        (252.0 / FORECAST_HORIZON)
    )

    return (
        float(vol_1d),
        float(vol_21d),
    )


# ================================================================
# BUILD SIGNAL FOR ONE DATE
# ================================================================

def forecast_for_date(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[float, float, float]:
    """
    Generate volatility forecast using ONLY data available
    on or before target_date.

    Returns:
        current_21d_realized_vol,
        garch_1d_forecast,
        garch_21d_forecast
    """

    historical = df[
        df["timestamp"] <= target_date
    ].copy()

    if len(historical) < TRAIN_WINDOW + 1:
        raise ValueError(
            f"Insufficient history before "
            f"{target_date.date()}"
        )

    historical = historical.tail(
        TRAIN_WINDOW + 1
    )

    returns = np.log(
        historical["close"]
        /
        historical["close"].shift(1)
    )

    returns = returns.dropna()

    # ------------------------------------------------------------
    # Current realized 21-day volatility
    # ------------------------------------------------------------

    current_21d = (
        returns
        .tail(FORECAST_HORIZON)
        .std()
        *
        np.sqrt(252.0)
    )

    # ------------------------------------------------------------
    # Causal GARCH forecast
    # ------------------------------------------------------------

    (
        forecast_1d,
        forecast_21d,
    ) = garch_forecast(
        returns
    )

    return (
        float(current_21d),
        float(forecast_1d),
        float(forecast_21d),
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print("QUANTOS CAUSAL HISTORICAL VOLATILITY SIGNAL ENGINE")
    print("=" * 80)

    print()
    print(
        f"Training window: "
        f"{TRAIN_WINDOW} observations"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON} trading days"
    )

    print(
        f"Rebalance frequency: "
        f"{REBALANCE_FREQUENCY} observations"
    )

    print()
    print(
        f"HMM signals: "
        f"{HMM_SIGNAL_PATH}"
    )

    # ------------------------------------------------------------
    # Load HMM signal dates
    # ------------------------------------------------------------

    hmm = pd.read_parquet(
        HMM_SIGNAL_PATH
    )

    hmm["timestamp"] = pd.to_datetime(
        hmm["timestamp"]
    )

    hmm = (
        hmm
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print()
    print(
        f"HMM signal dates: "
        f"{len(hmm)}"
    )

    # ------------------------------------------------------------
    # Load market data
    # ------------------------------------------------------------

    market_data = {}

    for symbol, path in (
        VOLATILITY_FILES.items()
    ):

        print()
        print(
            f"Loading {symbol}..."
        )

        df = load_price_data(
            path
        )

        market_data[symbol] = df

        print(
            f"  Rows: {len(df)}"
        )

        print(
            f"  Period: "
            f"{df['timestamp'].min().date()} "
            f"-> "
            f"{df['timestamp'].max().date()}"
        )

    # ------------------------------------------------------------
    # Generate causal forecasts
    # ------------------------------------------------------------

    results = []

    total = len(hmm)

    for counter, target_date in enumerate(
        hmm["timestamp"],
        start=1,
    ):

        print(
            f"\rGenerating volatility signal "
            f"{counter:>3}/{total} "
            f"| {target_date.date()}",
            end="",
            flush=True,
        )

        row = {
            "timestamp": target_date,
        }

        successful = True

        for symbol, df in (
            market_data.items()
        ):

            try:

                (
                    realized_21d,
                    forecast_1d,
                    forecast_21d,
                ) = forecast_for_date(
                    df,
                    target_date,
                )

                prefix = symbol.lower()

                row[
                    f"{prefix}_realized_21d"
                ] = realized_21d

                row[
                    f"{prefix}_garch_1d"
                ] = forecast_1d

                row[
                    f"{prefix}_garch_21d"
                ] = forecast_21d

            except Exception as exc:

                print()

                print(
                    f"WARNING: "
                    f"{symbol} "
                    f"{target_date.date()} "
                    f"failed: {exc}"
                )

                successful = False

                break

        if successful:
            results.append(row)

    print()
    print()

    if not results:
        raise RuntimeError(
            "No volatility signals generated."
        )

    signals = pd.DataFrame(
        results
    )

    signals = (
        signals
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Merge HMM regime information
    # ------------------------------------------------------------

    hmm_columns = [
        "timestamp",
        "current_regime",
        "next_regime",
        "bull_probability",
        "side_probability",
        "bear_probability",
        "next_bull_probability",
        "next_side_probability",
        "next_bear_probability",
    ]

    available_hmm_columns = [
        column
        for column in hmm_columns
        if column in hmm.columns
    ]

    signals = signals.merge(
        hmm[
            available_hmm_columns
        ],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    expected_rows = len(hmm)

    if len(signals) != expected_rows:

        print()
        print(
            "WARNING: "
            f"Expected {expected_rows} rows, "
            f"generated {len(signals)} rows."
        )

    forecast_columns = [
        column
        for column in signals.columns
        if column.endswith(
            "_garch_21d"
        )
    ]

    if signals[
        forecast_columns
    ].isna().any().any():

        raise RuntimeError(
            "Missing historical volatility forecasts."
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    signals.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print("=" * 80)
    print("HISTORICAL VOLATILITY SIGNAL RESULTS")
    print("=" * 80)

    print()
    print(
        f"Rows: {len(signals)}"
    )

    print(
        f"Period: "
        f"{signals['timestamp'].min().date()} "
        f"-> "
        f"{signals['timestamp'].max().date()}"
    )

    print()
    print("LATEST VOLATILITY FORECASTS")

    latest = (
        signals
        .tail(1)
        .T
    )

    print(
        latest.to_string()
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("=" * 80)
    print(
        "CAUSAL HISTORICAL VOLATILITY "
        "ENGINE COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
