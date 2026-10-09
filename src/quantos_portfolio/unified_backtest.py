from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .multi_market_returns import build_raw_return_matrix
from .allocation_engine import (
    minimum_variance,
    risk_parity,
)
from .constrained_allocation import (
    constrained_minimum_variance,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ADAPTIVE_SIGNAL_PATH = (
    ROOT
    / "data"
    / "regime"
    / "adaptive_allocation_signals.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "regime"
    / "unified_backtest"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "unified_backtest_summary.csv"
)

PERFORMANCE_PATH = (
    OUTPUT_DIR
    / "unified_backtest_performance.parquet"
)

EQUITY_PATH = (
    OUTPUT_DIR
    / "unified_equity_curves.parquet"
)

DRAWDOWN_PATH = (
    OUTPUT_DIR
    / "unified_drawdowns.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_WINDOW = 504

TRANSACTION_COST = 0.001

CONSTRAINED_MAX_WEIGHT = 0.25

TRADING_DAYS = 252


# ============================================================
# TIME NORMALIZATION
# ============================================================

def normalize_datetime_index(index):

    index = pd.to_datetime(index)

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    return index


def normalize_timestamp_series(series):

    series = pd.to_datetime(
        series,
        errors="coerce",
    )

    if isinstance(
        series.dtype,
        pd.DatetimeTZDtype,
    ):
        series = (
            series
            .dt
            .tz_localize(None)
        )

    return series


# ============================================================
# LOAD RETURNS
# ============================================================

def load_returns():

    print("Loading portfolio returns...")

    returns = build_raw_return_matrix()

    returns.index = normalize_datetime_index(
        returns.index
    )

    returns = (
        returns
        .sort_index()
        .loc[
            ~returns.index.duplicated(
                keep="last"
            )
        ]
    )

    print(
        f"Return matrix: "
        f"{returns.shape[0]} rows × "
        f"{returns.shape[1]} assets"
    )

    print(
        f"Return period: "
        f"{returns.index.min().date()} "
        f"-> "
        f"{returns.index.max().date()}"
    )

    return returns


# ============================================================
# LOAD ADAPTIVE SIGNALS
# ============================================================

def load_adaptive_signals():

    print(
        "\nLoading adaptive allocation signals..."
    )

    if not ADAPTIVE_SIGNAL_PATH.exists():

        raise FileNotFoundError(
            f"Adaptive signal file not found:\n"
            f"{ADAPTIVE_SIGNAL_PATH}"
        )

    signals = pd.read_parquet(
        ADAPTIVE_SIGNAL_PATH
    )

    if "timestamp" not in signals.columns:

        raise ValueError(
            "Adaptive signal file does not contain "
            "'timestamp'."
        )

    signals["timestamp"] = (
        normalize_timestamp_series(
            signals["timestamp"]
        )
    )

    signals = (
        signals
        .dropna(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Adaptive signal dates: "
        f"{len(signals)}"
    )

    print(
        f"Signal period: "
        f"{signals['timestamp'].min().date()} "
        f"-> "
        f"{signals['timestamp'].max().date()}"
    )

    return signals


# ============================================================
# CLEAN BASELINE WEIGHTS
# ============================================================

def clean_weights(
    weights,
    assets,
):

    if isinstance(
        weights,
        pd.Series,
    ):

        result = weights.copy()

    else:

        result = pd.Series(
            weights,
            index=assets,
            dtype=float,
        )

    result = result.reindex(
        assets,
        fill_value=0.0,
    )

    result = pd.to_numeric(
        result,
        errors="coerce",
    ).fillna(0.0)

    result = result.clip(
        lower=0.0
    )

    total = float(
        result.sum()
    )

    if total <= 0:

        result[:] = (
            1.0 / len(result)
        )

    else:

        result /= total

    return result


# ============================================================
# BASELINE WEIGHTS
# ============================================================

def calculate_baseline_weights(
    strategy,
    covariance,
    assets,
):

    # --------------------------------------------------------
    # Equal Weight
    # --------------------------------------------------------

    if strategy == "EQUAL_WEIGHT":

        n_assets = len(
            assets
        )

        if n_assets == 0:

            raise ValueError(
                "Cannot construct Equal Weight "
                "portfolio with zero assets."
            )

        weights = pd.Series(
            1.0 / n_assets,
            index=assets,
            dtype=float,
        )

    # --------------------------------------------------------
    # Minimum Variance
    # --------------------------------------------------------

    elif strategy == "MIN_VARIANCE":

        weights = minimum_variance(
            covariance
        )

    # --------------------------------------------------------
    # Constrained Minimum Variance
    # --------------------------------------------------------

    elif (
        strategy
        == "CONSTRAINED_MIN_VARIANCE"
    ):

        weights = (
            constrained_minimum_variance(
                covariance,
                max_weight=(
                    CONSTRAINED_MAX_WEIGHT
                ),
            )
        )

    # --------------------------------------------------------
    # Risk Parity
    # --------------------------------------------------------

    elif strategy == "RISK_PARITY":

        weights = risk_parity(
            covariance
        )

    else:

        raise ValueError(
            f"Unknown baseline strategy: "
            f"{strategy}"
        )

    return clean_weights(
        weights,
        assets,
    )


# ============================================================
# ADAPTIVE WEIGHTS
# ============================================================

def extract_adaptive_weights(
    signal_row,
    assets,
):
    """
    Extract the adaptive risky-asset weights.

    IMPORTANT:
    Do NOT normalize these weights.

    The adaptive allocator intentionally leaves some
    capital in CASH. Therefore the sum of the 14 risky
    weights may be below 1.0.
    """

    weights = {}

    for asset in assets:

        column = (
            f"weight_{asset}"
        )

        if column in signal_row.index:

            weights[asset] = float(
                signal_row[column]
            )

        else:

            weights[asset] = 0.0

    weights = pd.Series(
        weights,
        index=assets,
        dtype=float,
    )

    weights = weights.clip(
        lower=0.0
    )

    risky_exposure = float(
        weights.sum()
    )

    if risky_exposure > (
        1.0 + 1e-8
    ):

        raise ValueError(
            "Adaptive risky exposure exceeds "
            f"100%: {risky_exposure:.6f}"
        )

    return weights


# ============================================================
# TURNOVER
# ============================================================

def calculate_turnover(
    previous_weights,
    current_weights,
):

    if previous_weights is None:

        # Starting from 100% cash.
        return 1.0

    all_assets = sorted(
        set(
            previous_weights.index
        )
        |
        set(
            current_weights.index
        )
    )

    previous = (
        previous_weights
        .reindex(
            all_assets,
            fill_value=0.0,
        )
    )

    current = (
        current_weights
        .reindex(
            all_assets,
            fill_value=0.0,
        )
    )

    return float(
        0.5
        * np.abs(
            current - previous
        ).sum()
    )


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def calculate_portfolio_return(
    asset_returns,
    weights,
):

    common_assets = (
        asset_returns.index
        .intersection(
            weights.index
        )
    )

    if len(common_assets) == 0:

        return 0.0

    r = (
        asset_returns[
            common_assets
        ]
        .fillna(0.0)
    )

    w = (
        weights[
            common_assets
        ]
        .fillna(0.0)
    )

    return float(
        (r * w).sum()
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    returns,
    turnover,
    transaction_costs,
):

    returns = (
        returns
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if returns.empty:

        return {}

    equity = (
        1.0 + returns
    ).cumprod()

    final_wealth = float(
        equity.iloc[-1]
    )

    years = (
        len(returns)
        / TRADING_DAYS
    )

    if years > 0:

        cagr = (
            final_wealth
            ** (1.0 / years)
            - 1.0
        )

    else:

        cagr = np.nan

    annualized_volatility = (
        returns.std(
            ddof=1
        )
        * np.sqrt(
            TRADING_DAYS
        )
    )

    if annualized_volatility > 0:

        sharpe = (
            returns.mean()
            * TRADING_DAYS
            / annualized_volatility
        )

    else:

        sharpe = np.nan

    downside = returns[
        returns < 0
    ]

    if len(downside) > 0:

        downside_deviation = (
            np.sqrt(
                np.mean(
                    downside ** 2
                )
            )
            * np.sqrt(
                TRADING_DAYS
            )
        )

        if (
            downside_deviation
            > 0
        ):

            sortino = (
                returns.mean()
                * TRADING_DAYS
                / downside_deviation
            )

        else:

            sortino = np.nan

    else:

        sortino = np.nan

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    if (
        max_drawdown < 0
        and np.isfinite(cagr)
    ):

        calmar = (
            cagr
            / abs(max_drawdown)
        )

    else:

        calmar = np.nan

    return {
        "CAGR": cagr,
        "Annualized Volatility":
            annualized_volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown":
            max_drawdown,
        "Calmar": calmar,
        "Final Wealth":
            final_wealth,
        "Total Turnover":
            float(
                turnover.sum()
            ),
        "Transaction Costs":
            float(
                transaction_costs.sum()
            ),
        "Observations":
            len(returns),
    }


# ============================================================
# BACKTEST ONE STRATEGY
# ============================================================

def backtest_strategy(
    strategy,
    returns,
    signals,
    assets,
):

    print(
        f"\nRunning: {strategy}"
    )

    records = []

    previous_weights = None

    signal_dates = list(
        signals["timestamp"]
    )

    # ========================================================
    # LOOP THROUGH REBALANCE DATES
    # ========================================================

    for i in range(
        len(signal_dates) - 1
    ):

        signal_date = pd.Timestamp(
            signal_dates[i]
        )

        next_signal_date = pd.Timestamp(
            signal_dates[i + 1]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Create signal_row FIRST for EVERY strategy.
        #
        # This completely eliminates the previous
        # UnboundLocalError.
        # ----------------------------------------------------

        signal_match = signals.loc[
            signals["timestamp"]
            == signal_date
        ]

        if signal_match.empty:

            raise ValueError(
                "Could not find signal row for "
                f"{signal_date.date()}."
            )

        signal_row = (
            signal_match.iloc[0]
        )

        # ----------------------------------------------------
        # Historical training data
        #
        # Only information available on or before
        # the rebalance date is used.
        # ----------------------------------------------------

        training_returns = (
            returns.loc[
                returns.index
                <= signal_date
            ]
            .copy()
        )

        if len(
            training_returns
        ) < TRAIN_WINDOW:

            print(
                f"  Skipping "
                f"{signal_date.date()}: "
                f"only "
                f"{len(training_returns)} "
                f"training observations."
            )

            continue

        training_returns = (
            training_returns
            .tail(
                TRAIN_WINDOW
            )
        )

        # ----------------------------------------------------
        # Covariance matrix
        # ----------------------------------------------------

        covariance = (
            training_returns
            .cov()
            * TRADING_DAYS
        )

        covariance = (
            covariance
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        # ----------------------------------------------------
        # Find assets with valid covariance
        # ----------------------------------------------------

        valid_assets = []

        for asset in assets:

            if asset not in (
                covariance.columns
            ):

                continue

            row = covariance.loc[
                asset
            ]

            if row.notna().all():

                valid_assets.append(
                    asset
                )

        covariance = (
            covariance.loc[
                valid_assets,
                valid_assets,
            ]
        )

        if covariance.empty:

            print(
                f"  Skipping "
                f"{signal_date.date()}: "
                "empty covariance matrix."
            )

            continue

        # ====================================================
        # GENERATE TARGET WEIGHTS
        # ====================================================

        if (
            strategy
            == "ADAPTIVE_HMM_VOL"
        ):

            # Adaptive weights are already generated by
            # the causal HMM + volatility allocator.

            weights = (
                extract_adaptive_weights(
                    signal_row,
                    assets,
                )
            )

        else:

            # Baseline portfolio is generated from the
            # same historical covariance information.

            weights = (
                calculate_baseline_weights(
                    strategy,
                    covariance,
                    valid_assets,
                )
            )

            weights = (
                weights.reindex(
                    assets,
                    fill_value=0.0,
                )
            )

            weights = clean_weights(
                weights,
                assets,
            )

        # ====================================================
        # TURNOVER
        # ====================================================

        if (
            strategy
            == "ADAPTIVE_HMM_VOL"
        ):

            # The adaptive allocator already calculated
            # turnover including movement into/out of CASH.

            turnover = float(
                signal_row[
                    "turnover"
                ]
            )

        else:

            turnover = (
                calculate_turnover(
                    previous_weights,
                    weights,
                )
            )

        transaction_cost = (
            turnover
            * TRANSACTION_COST
        )

        # ====================================================
        # HOLDING PERIOD
        # ====================================================

        holding_returns = (
            returns.loc[
                (
                    returns.index
                    > signal_date
                )
                &
                (
                    returns.index
                    <= next_signal_date
                )
            ]
            .copy()
        )

        if holding_returns.empty:

            previous_weights = (
                weights.copy()
            )

            continue

        # ====================================================
        # DAILY RETURNS
        # ====================================================

        for j, date in enumerate(
            holding_returns.index
        ):

            asset_returns = (
                holding_returns.loc[
                    date
                ]
            )

            gross_return = (
                calculate_portfolio_return(
                    asset_returns,
                    weights,
                )
            )

            # Transaction cost is charged only on the
            # first trading observation after rebalance.

            if j == 0:

                cost = (
                    transaction_cost
                )

                applied_turnover = (
                    turnover
                )

            else:

                cost = 0.0

                applied_turnover = 0.0

            net_return = (
                gross_return
                - cost
            )

            records.append(
                {
                    "timestamp":
                        date,

                    "signal_date":
                        signal_date,

                    "next_signal_date":
                        next_signal_date,

                    "strategy":
                        strategy,

                    "gross_return":
                        gross_return,

                    "transaction_cost":
                        cost,

                    "net_return":
                        net_return,

                    "turnover":
                        applied_turnover,

                    "current_regime":
                        signal_row[
                            "current_regime"
                        ],

                    "next_regime":
                        signal_row[
                            "next_regime"
                        ],
                }
            )

        # ----------------------------------------------------
        # Store weights for next rebalance.
        #
        # Adaptive weights intentionally sum below 1 when
        # CASH is present. Baseline weights sum to 1.
        # ----------------------------------------------------

        previous_weights = (
            weights.copy()
        )

    # ========================================================
    # CHECK OUTPUT
    # ========================================================

    if not records:

        raise ValueError(
            f"No backtest observations "
            f"generated for {strategy}."
        )

    performance = pd.DataFrame(
        records
    )

    performance = (
        performance
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    performance[
        "equity_curve"
    ] = (
        1.0
        + performance[
            "net_return"
        ]
    ).cumprod()

    performance[
        "running_max"
    ] = (
        performance[
            "equity_curve"
        ].cummax()
    )

    performance[
        "drawdown"
    ] = (
        performance[
            "equity_curve"
        ]
        /
        performance[
            "running_max"
        ]
        - 1.0
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        returns=performance[
            "net_return"
        ],
        turnover=performance[
            "turnover"
        ],
        transaction_costs=performance[
            "transaction_cost"
        ],
    )

    metrics[
        "Strategy"
    ] = strategy

    return (
        performance,
        metrics,
    )


# ============================================================
# MAIN UNIFIED BACKTEST
# ============================================================

def run_unified_backtest():

    print("=" * 70)
    print(
        "QuantOS UNIFIED FIVE-STRATEGY BACKTEST"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    returns = load_returns()

    signals = (
        load_adaptive_signals()
    )

    assets = list(
        returns.columns
    )

    print(
        f"\nPortfolio universe: "
        f"{len(assets)} assets"
    )

    print(
        "Assets:"
    )

    for asset in assets:

        print(
            f"  - {asset}"
        )

    # ========================================================
    # VERIFY FIRST SIGNAL
    # ========================================================

    first_signal = pd.Timestamp(
        signals[
            "timestamp"
        ].iloc[0]
    )

    available_training = int(
        (
            returns.index
            <= first_signal
        ).sum()
    )

    print(
        f"\nFirst common signal date: "
        f"{first_signal.date()}"
    )

    print(
        f"Training observations available: "
        f"{available_training}"
    )

    print(
        f"Required training window: "
        f"{TRAIN_WINDOW}"
    )

    if (
        available_training
        < TRAIN_WINDOW
    ):

        raise ValueError(
            "Insufficient historical data "
            "for the first signal date."
        )

    # ========================================================
    # STRATEGIES
    # ========================================================

    strategies = [
        "EQUAL_WEIGHT",
        "MIN_VARIANCE",
        "CONSTRAINED_MIN_VARIANCE",
        "RISK_PARITY",
        "ADAPTIVE_HMM_VOL",
    ]

    all_performance = []

    summary_records = []

    # ========================================================
    # RUN ALL STRATEGIES
    # ========================================================

    for strategy in strategies:

        performance, metrics = (
            backtest_strategy(
                strategy=strategy,
                returns=returns,
                signals=signals,
                assets=assets,
            )
        )

        all_performance.append(
            performance
        )

        summary_records.append(
            metrics
        )

    # ========================================================
    # COMBINE PERFORMANCE
    # ========================================================

    performance = pd.concat(
        all_performance,
        ignore_index=True,
    )

    performance = (
        performance
        .sort_values(
            [
                "timestamp",
                "strategy",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        summary_records
    )

    summary = summary[
        [
            "Strategy",
            "CAGR",
            "Annualized Volatility",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
            "Calmar",
            "Final Wealth",
            "Total Turnover",
            "Transaction Costs",
            "Observations",
        ]
    ]

    # ========================================================
    # EQUITY CURVES
    # ========================================================

    equity_curves = (
        performance[
            [
                "timestamp",
                "strategy",
                "equity_curve",
            ]
        ]
        .pivot(
            index="timestamp",
            columns="strategy",
            values="equity_curve",
        )
        .sort_index()
        .ffill()
    )

    # ========================================================
    # DRAWDOWN CURVES
    # ========================================================

    drawdowns = (
        performance[
            [
                "timestamp",
                "strategy",
                "drawdown",
            ]
        ]
        .pivot(
            index="timestamp",
            columns="strategy",
            values="drawdown",
        )
        .sort_index()
        .ffill()
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    performance.to_parquet(
        PERFORMANCE_PATH,
        index=False,
    )

    equity_curves.to_parquet(
        EQUITY_PATH
    )

    drawdowns.to_parquet(
        DRAWDOWN_PATH
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "UNIFIED BACKTEST COMPLETE"
    )
    print("=" * 70)

    print(
        "\nCommon evaluation period:"
    )

    print(
        f"  "
        f"{performance['timestamp'].min().date()}"
        f" -> "
        f"{performance['timestamp'].max().date()}"
    )

    print(
        "\nCommon observations per strategy:"
    )

    for strategy in strategies:

        count = int(
            (
                performance[
                    "strategy"
                ]
                == strategy
            ).sum()
        )

        print(
            f"  "
            f"{strategy:<30}"
            f"{count}"
        )

    print(
        "\nPerformance comparison:"
    )

    display_columns = [
        "Strategy",
        "CAGR",
        "Annualized Volatility",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Calmar",
        "Final Wealth",
        "Total Turnover",
        "Transaction Costs",
    ]

    display_table = (
        summary[
            display_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Format percentages / ratios
    # --------------------------------------------------------

    format_columns = [
        "CAGR",
        "Annualized Volatility",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Calmar",
        "Final Wealth",
        "Total Turnover",
        "Transaction Costs",
    ]

    for column in format_columns:

        display_table[
            column
        ] = (
            display_table[
                column
            ]
            .map(
                lambda x:
                f"{x:.4f}"
            )
        )

    print(
        display_table.to_string(
            index=False
        )
    )

    # ========================================================
    # FILES
    # ========================================================

    print(
        "\nSaved files:"
    )

    print(
        f"  Summary:\n"
        f"  {SUMMARY_PATH}"
    )

    print(
        f"\n  Performance:\n"
        f"  {PERFORMANCE_PATH}"
    )

    print(
        f"\n  Equity curves:\n"
        f"  {EQUITY_PATH}"
    )

    print(
        f"\n  Drawdowns:\n"
        f"  {DRAWDOWN_PATH}"
    )

    print("=" * 70)

    return (
        summary,
        performance,
        equity_curves,
        drawdowns,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_unified_backtest()