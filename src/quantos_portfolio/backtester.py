from __future__ import annotations

import numpy as np
import pandas as pd

from .allocation_engine import (
    equal_weight,
    minimum_variance,
    risk_parity,
)
from .portfolio_analytics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
)


TRADING_DAYS = 252


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """
    One-way portfolio turnover.

    Turnover = sum(abs(new_weight - old_weight))
    """
    aligned = pd.concat(
        [old_weights, new_weights],
        axis=1,
    ).fillna(0.0)

    return float(
        np.abs(
            aligned.iloc[:, 1]
            - aligned.iloc[:, 0]
        ).sum()
    )


def apply_transaction_cost(
    portfolio_return: float,
    turnover: float,
    transaction_cost: float,
) -> float:
    """
    Deduct proportional transaction costs.

    transaction_cost:
        0.001 = 10 basis points
        0.0005 = 5 basis points
    """
    return (
        portfolio_return
        - turnover * transaction_cost
    )


def get_strategy_weights(
    strategy: str,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Calculate portfolio weights for a strategy.
    """
    strategy = strategy.upper()

    if strategy == "EQUAL_WEIGHT":
        return equal_weight(returns)

    if strategy == "MIN_VARIANCE":
        return minimum_variance(returns)

    if strategy == "RISK_PARITY":
        return risk_parity(returns)

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def walk_forward_backtest(
    returns: pd.DataFrame,
    strategy: str,
    train_window: int = 504,
    rebalance_frequency: int = 21,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Walk-forward portfolio backtest.

    At every rebalance date:

        1. Use only historical returns before the
           rebalance date.
        2. Estimate portfolio weights.
        3. Apply those weights to future returns.
        4. Deduct transaction costs.
        5. Repeat at the next rebalance date.

    train_window:
        Number of previous trading days used
        for optimization.

    rebalance_frequency:
        Number of trading days between rebalances.

    transaction_cost:
        Proportional transaction cost per unit turnover.
    """

    if returns.empty:
        raise ValueError(
            "Return matrix is empty."
        )

    returns = returns.copy()

    returns = returns.sort_index()

    returns = returns.dropna(
        how="any"
    )

    if len(returns) <= train_window:
        raise ValueError(
            "Not enough observations for "
            "the requested training window."
        )

    strategies = [
        "EQUAL_WEIGHT",
        "MIN_VARIANCE",
        "RISK_PARITY",
    ]

    if strategy.upper() not in strategies:
        raise ValueError(
            f"Strategy must be one of {strategies}"
        )

    records = []

    previous_weights = pd.Series(
        0.0,
        index=returns.columns,
    )

    for start in range(
        train_window,
        len(returns),
        rebalance_frequency,
    ):

        training_data = returns.iloc[
            start - train_window:start
        ]

        new_weights = get_strategy_weights(
            strategy,
            training_data,
        )

        new_weights = new_weights.reindex(
            returns.columns
        ).fillna(0.0)

        turnover = calculate_turnover(
            previous_weights,
            new_weights,
        )

        end = min(
            start + rebalance_frequency,
            len(returns),
        )

        holding_period = returns.iloc[
            start:end
        ]

        for date, row in holding_period.iterrows():

            gross_return = float(
                row.dot(new_weights)
            )

            # Transaction cost is charged only
            # on the first day of the new holding period.
            if date == holding_period.index[0]:
                net_return = apply_transaction_cost(
                    gross_return,
                    turnover,
                    transaction_cost,
                )
            else:
                net_return = gross_return

            records.append(
                {
                    "date": date,
                    "strategy": strategy.upper(),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "turnover": (
                        turnover
                        if date
                        == holding_period.index[0]
                        else 0.0
                    ),
                    "rebalance": (
                        date
                        == holding_period.index[0]
                    ),
                    "NIFTY_50_weight":
                        new_weights.get(
                            "NIFTY_50",
                            0.0,
                        ),
                    "NIFTY_BANK_weight":
                        new_weights.get(
                            "NIFTY_BANK",
                            0.0,
                        ),
                    "SENSEX_weight":
                        new_weights.get(
                            "SENSEX",
                            0.0,
                        ),
                }
            )

        previous_weights = new_weights.copy()

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError(
            "Backtest produced no observations."
        )

    result = result.set_index("date")

    result["wealth"] = (
        1.0
        + result["net_return"]
    ).cumprod()

    result["drawdown"] = (
        result["wealth"]
        / result["wealth"].cummax()
        - 1.0
    )

    return result


def summarize_backtest(
    result: pd.DataFrame,
) -> dict:
    """
    Calculate performance statistics
    from a walk-forward backtest.
    """
    returns = result["net_return"]

    return {
        "strategy": result["strategy"].iloc[0],
        "start_date": result.index.min(),
        "end_date": result.index.max(),
        "observations": len(result),
        "cagr": annualized_return(returns),
        "annualized_volatility":
            annualized_volatility(returns),
        "sharpe":
            sharpe_ratio(returns),
        "sortino":
            sortino_ratio(returns),
        "max_drawdown":
            max_drawdown(returns),
        "calmar":
            calmar_ratio(returns),
        "total_return":
            result["wealth"].iloc[-1] - 1.0,
        "final_wealth":
            result["wealth"].iloc[-1],
        "total_turnover":
            result["turnover"].sum(),
        "number_of_rebalances":
            int(result["rebalance"].sum()),
    }


def compare_strategies(
    returns: pd.DataFrame,
    train_window: int = 504,
    rebalance_frequency: int = 21,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Run all supported strategies and return
    a comparative performance table.
    """
    summaries = []

    strategies = [
        "EQUAL_WEIGHT",
        "MIN_VARIANCE",
        "RISK_PARITY",
    ]

    for strategy in strategies:

        result = walk_forward_backtest(
            returns=returns,
            strategy=strategy,
            train_window=train_window,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost,
        )

        summaries.append(
            summarize_backtest(result)
        )

    return pd.DataFrame(summaries).set_index(
        "strategy"
    )