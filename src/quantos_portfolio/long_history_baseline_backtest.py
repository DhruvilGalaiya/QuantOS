from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .long_history_universe import build_long_history_returns
from .long_history_risk import build_long_history_covariance


ROLLING_WINDOW = 504
REBALANCE_EVERY = 21
TRANSACTION_COST = 0.001


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def equal_weight(covariance: pd.DataFrame) -> pd.Series:

    assets = covariance.columns

    return pd.Series(
        1.0 / len(assets),
        index=assets,
    )


def clean_covariance(covariance: pd.DataFrame) -> np.ndarray:

    cov = covariance.to_numpy(dtype=float)

    # Symmetrize numerical noise.
    cov = (cov + cov.T) / 2.0

    # Ensure positive semi-definite covariance.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    eigenvalues = np.maximum(
        eigenvalues,
        1e-10,
    )

    cov = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    return cov


def minimum_variance(
    covariance: pd.DataFrame,
    max_weight: float | None = None,
) -> pd.Series:

    assets = covariance.columns
    n = len(assets)

    cov = clean_covariance(covariance)

    x0 = np.ones(n) / n

    if max_weight is None:
        upper_bound = 1.0
    else:
        upper_bound = max_weight

    bounds = [
        (0.0, upper_bound)
        for _ in range(n)
    ]

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }

    def objective(w):
        return float(w @ cov @ w)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Minimum variance optimization failed: "
            f"{result.message}"
        )

    weights = np.clip(
        result.x,
        0.0,
        None,
    )

    weights /= weights.sum()

    return pd.Series(
        weights,
        index=assets,
    )


def risk_parity(
    covariance: pd.DataFrame,
) -> pd.Series:

    assets = covariance.columns
    n = len(assets)

    cov = clean_covariance(covariance)

    x0 = np.ones(n) / n

    def risk_contributions(w):

        portfolio_variance = (
            w @ cov @ w
        )

        portfolio_vol = np.sqrt(
            max(portfolio_variance, 1e-16)
        )

        marginal_risk = (
            cov @ w
        )

        component_risk = (
            w * marginal_risk
        )

        return (
            component_risk / portfolio_vol
        )

    def objective(w):

        rc = risk_contributions(w)

        target = rc.sum() / n

        return np.sum(
            (rc - target) ** 2
        )

    bounds = [
        (1e-8, 1.0)
        for _ in range(n)
    ]

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Risk parity optimization failed: "
            f"{result.message}"
        )

    weights = np.clip(
        result.x,
        0.0,
        None,
    )

    weights /= weights.sum()

    return pd.Series(
        weights,
        index=assets,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    returns: pd.Series,
    turnover: float,
    transaction_cost: float,
) -> dict:

    returns = returns.dropna()

    if returns.empty:
        return {}

    wealth = (1.0 + returns).cumprod()

    n_days = len(returns)

    years = n_days / 252.0

    final_wealth = float(
        wealth.iloc[-1]
    )

    cagr = (
        final_wealth ** (1.0 / years)
        - 1.0
    )

    annualized_vol = (
        returns.std(ddof=1)
        * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(252)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    downside = returns[returns < 0]

    if len(downside) > 0:
        downside_vol = (
            np.sqrt(
                np.mean(
                    downside ** 2
                )
            )
            * np.sqrt(252)
        )

        sortino = (
            returns.mean()
            * 252
            / downside_vol
            if downside_vol > 0
            else np.nan
        )
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

    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    return {
        "CAGR": cagr,
        "Volatility": annualized_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar,
        "Final Wealth": final_wealth,
        "Turnover": turnover,
        "Transaction Cost": transaction_cost,
        "Observations": n_days,
    }


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def run_strategy(
    strategy_name: str,
    returns: pd.DataFrame,
    covariance_matrices: dict,
) -> tuple[pd.Series, dict]:

    returns = returns.sort_index()

    valid_rebalance_dates = [
        date
        for date in covariance_matrices
        if date in returns.index
    ]

    valid_rebalance_dates.sort()

    if not valid_rebalance_dates:
        raise RuntimeError(
            "No valid rebalance dates found."
        )

    # Rebalance every 21 observations.
    rebalance_dates = valid_rebalance_dates[
        ::REBALANCE_EVERY
    ]

    # Make sure the final available date is included.
    if rebalance_dates[-1] != valid_rebalance_dates[-1]:
        rebalance_dates.append(
            valid_rebalance_dates[-1]
        )

    period_returns = []

    total_turnover = 0.0
    total_transaction_cost = 0.0

    for i, rebalance_date in enumerate(
        rebalance_dates
    ):

        covariance = covariance_matrices[
            rebalance_date
        ]

        # ----------------------------------------------------
        # ALIGN ASSETS
        # ----------------------------------------------------

        available_assets = [
            asset
            for asset in returns.columns
            if asset in covariance.index
        ]

        covariance = covariance.loc[
            available_assets,
            available_assets,
        ]

        # ----------------------------------------------------
        # CALCULATE TARGET WEIGHTS
        #
        # The covariance matrix for rebalance_date
        # was calculated using observations BEFORE
        # rebalance_date.
        # ----------------------------------------------------

        if strategy_name == "Equal Weight":

            target_weights = equal_weight(
                covariance
            )

        elif strategy_name == "Minimum Variance":

            target_weights = minimum_variance(
                covariance
            )

        elif strategy_name == "Constrained Min Variance":

            target_weights = minimum_variance(
                covariance,
                max_weight=0.25,
            )

        elif strategy_name == "Risk Parity":

            target_weights = risk_parity(
                covariance
            )

        else:

            raise ValueError(
                f"Unknown strategy: {strategy_name}"
            )

        # ----------------------------------------------------
        # TURNOVER
        # ----------------------------------------------------

        if i == 0:

            # Initial portfolio construction.
            turnover = float(
                target_weights.abs().sum()
            )

        else:

            previous_weights = previous_target_weights.reindex(
                target_weights.index,
                fill_value=0.0,
            )

            turnover = float(
                (
                    target_weights
                    - previous_weights
                )
                .abs()
                .sum()
            )

        transaction_cost = (
            turnover
            * TRANSACTION_COST
        )

        total_turnover += turnover
        total_transaction_cost += (
            transaction_cost
        )

        # ----------------------------------------------------
        # HOLDING PERIOD
        #
        # New weights apply from this rebalance date
        # until the day BEFORE the next rebalance.
        #
        # Therefore each daily return belongs to exactly
        # one portfolio holding period.
        # ----------------------------------------------------

        start_idx = returns.index.get_loc(
            rebalance_date
        )

        if i + 1 < len(rebalance_dates):

            next_rebalance = (
                rebalance_dates[i + 1]
            )

            next_idx = returns.index.get_loc(
                next_rebalance
            )

            end_idx = next_idx - 1

        else:

            end_idx = len(returns) - 1

        if end_idx < start_idx:
            continue

        holding_returns = returns.iloc[
            start_idx:end_idx + 1
        ][target_weights.index]

        # ----------------------------------------------------
        # DIFFERENT MARKET CALENDARS
        #
        # Indian and US assets do not trade on identical
        # dates. A closed market contributes zero return
        # for that day while its portfolio weight remains
        # unchanged.
        # ----------------------------------------------------

        holding_returns = (
            holding_returns.fillna(0.0)
        )

        portfolio_returns = (
            holding_returns
            @ target_weights
        )

        # ----------------------------------------------------
        # TRANSACTION COST
        #
        # Charge the rebalance cost once at the beginning
        # of each new holding period.
        # ----------------------------------------------------

        if len(portfolio_returns) > 0:

            portfolio_returns.iloc[0] -= (
                transaction_cost
            )

        period_returns.append(
            portfolio_returns
        )

        previous_target_weights = (
            target_weights.copy()
        )

    # --------------------------------------------------------
    # COMBINE ALL HOLDING PERIODS
    # --------------------------------------------------------

    if not period_returns:
        raise RuntimeError(
            f"No portfolio returns generated "
            f"for {strategy_name}"
        )

    result = pd.concat(
        period_returns
    ).sort_index()

    # Defensive check: every date should occur exactly once.
    if result.index.duplicated().any():

        duplicated_dates = (
            result.index[
                result.index.duplicated()
            ]
            .unique()
        )

        raise RuntimeError(
            "Duplicate portfolio return dates "
            f"detected: {duplicated_dates[:5]}"
        )

    metrics = calculate_metrics(
        result,
        total_turnover,
        total_transaction_cost,
    )

    return result, metrics

# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 80)
    print("QUANTOS LONG-HISTORY BASELINE ALLOCATION BACKTEST")
    print("=" * 80)

    returns = build_long_history_returns()

    print(
    "\nNon-missing return observations:",
    int(returns.notna().sum().sum())
     )

    print(
    "Assets:",
    list(returns.columns)
)

    covariance_matrices = (
        build_long_history_covariance(
            returns,
            window=ROLLING_WINDOW,
        )
    )

    print("\nDataset:")
    print("Assets:", len(returns.columns))
    print("Observations:", len(returns))
    print(
        "Date range:",
        returns.index.min().date(),
        "->",
        returns.index.max().date(),
    )

    print("\nRisk model:")
    print(
        "Covariance window:",
        ROLLING_WINDOW,
    )

    print(
        "Rebalance frequency:",
        REBALANCE_EVERY,
        "observations",
    )

    print(
        "Transaction cost:",
        TRANSACTION_COST * 10000,
        "bps",
    )

    strategies = [
        "Equal Weight",
        "Minimum Variance",
        "Constrained Min Variance",
        "Risk Parity",
    ]

    results = {}
    metrics = {}

    for strategy in strategies:

        print(
            f"\nRunning: {strategy}"
        )

        strategy_returns, strategy_metrics = (
            run_strategy(
                strategy,
                returns,
                covariance_matrices,
            )
        )

        results[strategy] = strategy_returns
        metrics[strategy] = strategy_metrics

        print(
            "Observations:",
            strategy_metrics["Observations"],
        )

        print(
            "CAGR:",
            f"{strategy_metrics['CAGR']:.4f}",
        )

        print(
            "Volatility:",
            f"{strategy_metrics['Volatility']:.4f}",
        )

        print(
            "Sharpe:",
            f"{strategy_metrics['Sharpe']:.4f}",
        )

        print(
            "Sortino:",
            f"{strategy_metrics['Sortino']:.4f}",
        )

        print(
            "Max Drawdown:",
            f"{strategy_metrics['Max Drawdown']:.4f}",
        )

        print(
            "Calmar:",
            f"{strategy_metrics['Calmar']:.4f}",
        )

        print(
            "Final Wealth:",
            f"{strategy_metrics['Final Wealth']:.4f}",
        )

        print(
            "Turnover:",
            f"{strategy_metrics['Turnover']:.4f}",
        )

        print(
            "Transaction Cost:",
            f"{strategy_metrics['Transaction Cost']:.6f}",
        )

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    summary = pd.DataFrame(metrics).T

    print("\n")
    print("=" * 80)
    print("LONG-HISTORY BASELINE SUMMARY")
    print("=" * 80)

    print(
        summary[
            [
                "CAGR",
                "Volatility",
                "Sharpe",
                "Sortino",
                "Max Drawdown",
                "Calmar",
                "Final Wealth",
                "Turnover",
                "Transaction Cost",
            ]
        ].round(4)
    )

    print("=" * 80)


if __name__ == "__main__":
    main()