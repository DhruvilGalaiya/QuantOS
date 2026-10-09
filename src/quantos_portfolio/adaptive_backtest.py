from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .multi_market_returns import build_raw_return_matrix


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
    / "adaptive_backtest"
)

PERFORMANCE_PATH = (
    OUTPUT_DIR
    / "adaptive_performance.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "adaptive_backtest_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRADING_DAYS = 252

TRANSACTION_COST = 0.001


# ============================================================
# TIME NORMALIZATION
# ============================================================

def normalize_index(index):
    index = pd.to_datetime(index)

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    return index


# ============================================================
# LOAD DATA
# ============================================================

def load_returns():

    returns = build_raw_return_matrix()

    returns.index = normalize_index(returns.index)

    returns = (
        returns
        .sort_index()
        .loc[~returns.index.duplicated(keep="last")]
    )

    return returns


def load_adaptive_signals():

    if not ADAPTIVE_SIGNAL_PATH.exists():
        raise FileNotFoundError(
            f"Adaptive signal file not found:\n"
            f"{ADAPTIVE_SIGNAL_PATH}"
        )

    signals = pd.read_parquet(
        ADAPTIVE_SIGNAL_PATH
    )

    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"]
    )

    if isinstance(
        signals["timestamp"].dtype,
        pd.DatetimeTZDtype,
    ):
        signals["timestamp"] = (
            signals["timestamp"]
            .dt
            .tz_localize(None)
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

    return signals


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def calculate_portfolio_return(
    asset_returns: pd.Series,
    weights: pd.Series,
) -> float:

    common_assets = (
        asset_returns.index
        .intersection(weights.index)
    )

    if len(common_assets) == 0:
        return 0.0

    r = asset_returns[
        common_assets
    ].fillna(0.0)

    w = weights[
        common_assets
    ].fillna(0.0)

    return float(
        (r * w).sum()
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    transaction_costs: pd.Series,
) -> dict:

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    wealth = (
        1.0 + returns
    ).cumprod()

    total_years = (
        len(returns)
        / TRADING_DAYS
    )

    final_wealth = float(
        wealth.iloc[-1]
    )

    if total_years > 0:
        cagr = (
            final_wealth
            ** (1.0 / total_years)
            - 1.0
        )
    else:
        cagr = np.nan

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    mean_return = returns.mean()

    if annualized_volatility > 0:
        sharpe = (
            mean_return
            * TRADING_DAYS
            / annualized_volatility
        )
    else:
        sharpe = np.nan

    downside = returns[
        returns < 0
    ]

    if len(downside) > 0:

        downside_volatility = (
            np.sqrt(
                np.mean(
                    downside ** 2
                )
            )
            * np.sqrt(TRADING_DAYS)
        )

        if downside_volatility > 0:
            sortino = (
                mean_return
                * TRADING_DAYS
                / downside_volatility
            )
        else:
            sortino = np.nan

    else:
        sortino = np.nan

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    if max_drawdown < 0:
        calmar = (
            cagr
            / abs(max_drawdown)
        )
    else:
        calmar = np.nan

    total_turnover = float(
        turnover.sum()
    )

    total_transaction_cost = float(
        transaction_costs.sum()
    )

    return {
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar,
        "Final Wealth": final_wealth,
        "Total Turnover": total_turnover,
        "Total Transaction Cost": total_transaction_cost,
        "Observations": len(returns),
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest():

    print("=" * 70)
    print("QuantOS ADAPTIVE ALLOCATION BACKTEST")
    print("=" * 70)

    print("\nLoading portfolio returns...")

    returns = load_returns()

    print(
        f"Return matrix: "
        f"{returns.shape[0]} rows × "
        f"{returns.shape[1]} assets"
    )

    print(
        f"Return period: "
        f"{returns.index.min().date()} -> "
        f"{returns.index.max().date()}"
    )

    print("\nLoading adaptive allocation signals...")

    signals = load_adaptive_signals()

    print(
        f"Adaptive signals: "
        f"{len(signals)}"
    )

    print(
        f"Signal period: "
        f"{signals['timestamp'].min().date()} -> "
        f"{signals['timestamp'].max().date()}"
    )

    weight_columns = [
        column
        for column in signals.columns
        if column.startswith("weight_")
    ]

    if "weight_CASH" not in weight_columns:
        raise ValueError(
            "Adaptive signal file does not contain CASH weights."
        )

    # --------------------------------------------------------
    # Prepare signal weights
    # --------------------------------------------------------

    signal_weights = {}

    for _, row in signals.iterrows():

        timestamp = pd.Timestamp(
            row["timestamp"]
        )

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)

        weights = {}

        for column in weight_columns:

            asset = column.replace(
                "weight_",
                "",
            )

            weights[asset] = float(
                row[column]
            )

        signal_weights[timestamp] = pd.Series(
            weights,
            dtype=float,
        )

    signal_dates = sorted(
        signal_weights.keys()
    )

    if len(signal_dates) < 2:
        raise ValueError(
            "At least two adaptive signal dates are required."
        )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    records = []

    for i in range(
        len(signal_dates) - 1
    ):

        signal_date = signal_dates[i]

        next_signal_date = signal_dates[i + 1]

        weights = signal_weights[
            signal_date
        ]

        # ----------------------------------------------------
        # Returns AFTER the signal date and BEFORE next signal.
        #
        # This prevents look-ahead.
        # ----------------------------------------------------

        holding_returns = returns.loc[
            (returns.index > signal_date)
            & (returns.index <= next_signal_date)
        ].copy()

        if holding_returns.empty:
            continue

        # ----------------------------------------------------
        # Transaction cost occurs at rebalance.
        #
        # The cost is applied once on the first trading
        # observation following the signal date.
        # ----------------------------------------------------

        turnover = float(
            signals.loc[
                signals["timestamp"] == signal_date,
                "turnover",
            ].iloc[0]
        )

        transaction_cost = (
            turnover
            * TRANSACTION_COST
        )

        for j, date in enumerate(
            holding_returns.index
        ):

            asset_returns = holding_returns.loc[
                date
            ]

            portfolio_return = (
                calculate_portfolio_return(
                    asset_returns,
                    weights,
                )
            )

            cost = (
                transaction_cost
                if j == 0
                else 0.0
            )

            net_return = (
                portfolio_return
                - cost
            )

            records.append(
                {
                    "timestamp": date,
                    "signal_date": signal_date,
                    "next_signal_date": next_signal_date,
                    "gross_return": portfolio_return,
                    "transaction_cost": cost,
                    "net_return": net_return,
                    "turnover": (
                        turnover
                        if j == 0
                        else 0.0
                    ),
                    "current_regime": signals.loc[
                        signals["timestamp"]
                        == signal_date,
                        "current_regime",
                    ].iloc[0],
                    "next_regime": signals.loc[
                        signals["timestamp"]
                        == signal_date,
                        "next_regime",
                    ].iloc[0],
                }
            )

    performance = pd.DataFrame(
        records
    )

    if performance.empty:
        raise ValueError(
            "Backtest produced no observations."
        )

    performance = (
        performance
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Equity curve
    # --------------------------------------------------------

    performance["equity_curve"] = (
        1.0
        + performance["net_return"]
    ).cumprod()

    performance["running_max"] = (
        performance["equity_curve"]
        .cummax()
    )

    performance["drawdown"] = (
        performance["equity_curve"]
        / performance["running_max"]
        - 1.0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        returns=performance["net_return"],
        turnover=performance["turnover"],
        transaction_costs=performance[
            "transaction_cost"
        ],
    )

    summary = pd.DataFrame(
        [metrics]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance.to_parquet(
        PERFORMANCE_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADAPTIVE BACKTEST COMPLETE")
    print("=" * 70)

    print(
        f"\nBacktest period: "
        f"{performance['timestamp'].min().date()} -> "
        f"{performance['timestamp'].max().date()}"
    )

    print(
        f"Observations: "
        f"{len(performance)}"
    )

    print("\nPerformance metrics:")

    print(
        f"  CAGR                 : "
        f"{metrics['CAGR']:.4f}"
    )

    print(
        f"  Annualized Volatility: "
        f"{metrics['Annualized Volatility']:.4f}"
    )

    print(
        f"  Sharpe               : "
        f"{metrics['Sharpe']:.4f}"
    )

    print(
        f"  Sortino              : "
        f"{metrics['Sortino']:.4f}"
    )

    print(
        f"  Max Drawdown         : "
        f"{metrics['Max Drawdown']:.4f}"
    )

    print(
        f"  Calmar               : "
        f"{metrics['Calmar']:.4f}"
    )

    print(
        f"  Final Wealth         : "
        f"{metrics['Final Wealth']:.4f}"
    )

    print(
        f"  Total Turnover       : "
        f"{metrics['Total Turnover']:.4f}"
    )

    print(
        f"  Transaction Costs    : "
        f"{metrics['Total Transaction Cost']:.6f}"
    )

    print(
        f"\nSaved performance data to:\n"
        f"{PERFORMANCE_PATH}"
    )

    print(
        f"\nSaved summary to:\n"
        f"{SUMMARY_PATH}"
    )

    print("=" * 70)

    return performance, summary


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_backtest()