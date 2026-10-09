from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantos_portfolio.multi_market_returns import (
    build_portfolio_return_matrix,
)

from src.quantos_portfolio.allocation_engine import (
    equal_weight,
    minimum_variance,
    risk_parity,
)

from src.quantos_portfolio.constrained_allocation import (
    constrained_minimum_variance,
)


TRAIN_WINDOW = 504
REBALANCE_FREQUENCY = 21
TRANSACTION_COST = 0.001


def portfolio_volatility(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> float:

    covariance = (
        returns.cov()
        * 252
    )

    weights = weights.reindex(
        covariance.index
    )

    w = weights.to_numpy(
        dtype=float
    )

    cov = covariance.to_numpy(
        dtype=float
    )

    return float(
        np.sqrt(
            w @ cov @ w
        )
    )


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:

    combined = pd.concat(
        [
            old_weights,
            new_weights,
        ],
        axis=1,
    ).fillna(0.0)

    return float(
        0.5
        * np.abs(
            combined.iloc[:, 0]
            - combined.iloc[:, 1]
        ).sum()
    )


def get_strategy_weights(
    strategy: str,
    training_returns: pd.DataFrame,
) -> pd.Series:

    covariance = (
        training_returns.cov()
        * 252
    )

    if strategy == "EQUAL_WEIGHT":

        return equal_weight(
            covariance
        )

    if strategy == "MIN_VARIANCE":

        return minimum_variance(
            covariance
        )

    if strategy == "CONSTRAINED_MIN_VARIANCE":

        return constrained_minimum_variance(
            covariance,
            max_weight=0.25,
        )

    if strategy == "RISK_PARITY":

        return risk_parity(
            covariance
        )

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def walk_forward_backtest(
    returns: pd.DataFrame,
    strategy: str,
    train_window: int = TRAIN_WINDOW,
    rebalance_frequency: int = REBALANCE_FREQUENCY,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:

    returns = returns.copy()

    returns = returns.sort_index()

    # Remove the first undefined return.
    returns = returns.dropna(
        how="all"
    )

    strategies = []

    start = train_window

    current_weights = None

    records = []

    while start < len(returns):

        train_start = start - train_window

        train_end = start

        training = returns.iloc[
            train_start:train_end
        ].copy()

        # Use only dates where all assets have
        # a portfolio-ready return.
        training = training.dropna(
            how="any"
        )

        if len(training) < train_window * 0.8:

            start += rebalance_frequency

            continue

        new_weights = get_strategy_weights(
            strategy,
            training,
        )

        new_weights = new_weights.reindex(
            returns.columns
        ).fillna(0.0)

        new_weights = (
            new_weights
            / new_weights.sum()
        )

        if current_weights is None:

            turnover = 1.0

        else:

            turnover = calculate_turnover(
                current_weights,
                new_weights,
            )

        end = min(
            start + rebalance_frequency,
            len(returns),
        )

        holding_period = returns.iloc[
            start:end
        ].copy()

        for date, row in holding_period.iterrows():

            gross_return = float(
                row.fillna(0.0)
                @ new_weights
            )

            cost = 0.0

            if date == holding_period.index[0]:

                cost = (
                    turnover
                    * transaction_cost
                )

            net_return = (
                gross_return
                - cost
            )

            records.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "gross_return": gross_return,
                    "transaction_cost": cost,
                    "net_return": net_return,
                    "turnover": turnover
                    if date
                    == holding_period.index[0]
                    else 0.0,
                }
            )

        current_weights = new_weights.copy()

        start += rebalance_frequency

    result = pd.DataFrame(records)

    if result.empty:

        raise RuntimeError(
            f"No backtest observations "
            f"generated for {strategy}."
        )

    result = result.sort_values(
        "date"
    )

    result["wealth"] = (
        1.0
        * (1.0 + result["net_return"])
        .cumprod()
    )

    result["drawdown"] = (
        result["wealth"]
        / result["wealth"].cummax()
        - 1.0
    )

    return result


def summarize_backtest(
    result: pd.DataFrame,
) -> dict:

    returns = result["net_return"]

    periods = len(returns)

    years = periods / 252

    total_return = (
        result["wealth"].iloc[-1]
        - 1.0
    )

    if years > 0:

        cagr = (
            result["wealth"].iloc[-1]
            ** (1.0 / years)
            - 1.0
        )

    else:

        cagr = np.nan

    annualized_volatility = (
        returns.std()
        * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    downside = returns[
        returns < 0
    ]

    downside_volatility = (
        downside.std()
        * np.sqrt(252)
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        returns.mean()
        * 252
        / downside_volatility
        if downside_volatility
        and downside_volatility > 0
        else np.nan
    )

    max_drawdown = (
        result["drawdown"].min()
    )

    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    total_turnover = (
        result["turnover"].sum()
    )

    total_transaction_cost = (
        result["transaction_cost"].sum()
    )

    return {
        "CAGR": cagr,
        "Annualized Volatility":
            annualized_volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Maximum Drawdown":
            max_drawdown,
        "Calmar": calmar,
        "Total Return":
            total_return,
        "Final Wealth":
            result["wealth"].iloc[-1],
        "Total Turnover":
            total_turnover,
        "Transaction Cost":
            total_transaction_cost,
        "Observations":
            periods,
    }


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("QUANTOS WALK-FORWARD BACKTEST")
    print("=" * 80)

    returns = (
        build_portfolio_return_matrix()
    )

    strategies = [
        "EQUAL_WEIGHT",
        "MIN_VARIANCE",
        "CONSTRAINED_MIN_VARIANCE",
        "RISK_PARITY",
    ]

    summaries = {}

    for strategy in strategies:

        print()
        print(
            f"Running {strategy}..."
        )

        result = walk_forward_backtest(
            returns=returns,
            strategy=strategy,
        )

        summary = summarize_backtest(
            result
        )

        summaries[strategy] = summary

    summary_df = pd.DataFrame(
        summaries
    ).T

    print()
    print("=" * 80)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 80)

    print(
        summary_df.round(4)
    )
    output_path = (
    "data/regime/portfolio_backtest_summary.csv"
)

summary_df.to_csv(
    output_path
)
pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    200,
)

print()
print(
    f"Full results saved to: {output_path}"
)