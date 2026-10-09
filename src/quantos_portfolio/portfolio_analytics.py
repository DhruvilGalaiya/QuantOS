from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    """
    Geometric annualized return based on daily simple returns.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    cumulative = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS

    if cumulative <= 0 or years <= 0:
        return np.nan

    return cumulative ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility from daily returns.
    """
    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio.

    risk_free_rate is expressed as an annual decimal.
    Example: 6% = 0.06
    """
    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - daily_rf

    volatility = excess.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return excess.mean() / volatility * np.sqrt(TRADING_DAYS)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sortino ratio using downside deviation.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - daily_rf

    downside = excess[excess < 0]

    if len(downside) == 0:
        return np.nan

    downside_deviation = np.sqrt(
        np.mean(np.square(downside))
    ) * np.sqrt(TRADING_DAYS)

    if downside_deviation == 0:
        return np.nan

    return excess.mean() * TRADING_DAYS / downside_deviation


def equity_curve(returns: pd.Series) -> pd.Series:
    """
    Growth of one unit invested at the beginning.
    """
    returns = returns.fillna(0.0)

    return (1.0 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown.
    Returned as a negative decimal.
    """
    curve = equity_curve(returns)

    if curve.empty:
        return np.nan

    peak = curve.cummax()
    drawdown = curve / peak - 1.0

    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    """
    Annualized return divided by absolute maximum drawdown.
    """
    ann_return = annualized_return(returns)
    mdd = max_drawdown(returns)

    if np.isnan(ann_return) or np.isnan(mdd) or mdd == 0:
        return np.nan

    return ann_return / abs(mdd)


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Historical Value-at-Risk.

    Returned as a positive loss magnitude.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    quantile = returns.quantile(1.0 - confidence)

    return max(0.0, -quantile)


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Historical Conditional VaR / Expected Shortfall.

    Average loss beyond the VaR threshold.
    Returned as a positive loss magnitude.
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    threshold = returns.quantile(1.0 - confidence)
    tail = returns[returns <= threshold]

    if tail.empty:
        return np.nan

    return max(0.0, -tail.mean())


def beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Beta of an asset relative to a benchmark.
    """
    aligned = pd.concat(
        [asset_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        return np.nan

    asset = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]

    benchmark_variance = benchmark.var(ddof=1)

    if benchmark_variance == 0:
        return np.nan

    covariance = asset.cov(benchmark)

    return covariance / benchmark_variance


def correlation_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pearson correlation matrix.
    """
    return returns.corr()


def covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Covariance matrix.

    If annualize=True, covariance is scaled by 252.
    """
    covariance = returns.cov()

    if annualize:
        covariance = covariance * TRADING_DAYS

    return covariance


def portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Construct portfolio returns from asset weights.

    Missing asset columns raise an error.
    """
    weights = {
        symbol: float(weight)
        for symbol, weight in weights.items()
    }

    missing = [
        symbol
        for symbol in weights
        if symbol not in returns.columns
    ]

    if missing:
        raise ValueError(
            f"Assets missing from return matrix: {missing}"
        )

    weight_vector = pd.Series(weights, dtype=float)

    selected = returns[list(weights.keys())]

    return selected.mul(weight_vector, axis=1).sum(axis=1, min_count=1)


def risk_contribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Marginal/component contribution to portfolio volatility.

    Returns each asset's percentage contribution
    to total portfolio volatility.
    """
    weight_vector = pd.Series(
        weights,
        dtype=float,
    )

    covariance = covariance_matrix(
        returns[list(weights.keys())],
        annualize=True,
    )

    w = weight_vector.values
    sigma = covariance.values

    portfolio_variance = w.T @ sigma @ w

    if portfolio_variance <= 0:
        return pd.Series(
            np.nan,
            index=weight_vector.index,
        )

    portfolio_volatility = np.sqrt(portfolio_variance)

    marginal_contribution = sigma @ w / portfolio_volatility

    component_contribution = (
        w * marginal_contribution
    )

    contribution_pct = (
        component_contribution
        / portfolio_volatility
    )

    return pd.Series(
        contribution_pct,
        index=weight_vector.index,
    )


def portfolio_summary(
    returns: pd.DataFrame,
    weights: dict[str, float],
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Complete portfolio analytics summary.
    """
    p_returns = portfolio_returns(
        returns,
        weights,
    )

    summary = {
        "annualized_return": annualized_return(p_returns),
        "annualized_volatility": annualized_volatility(p_returns),
        "sharpe_ratio": sharpe_ratio(
            p_returns,
            risk_free_rate,
        ),
        "sortino_ratio": sortino_ratio(
            p_returns,
            risk_free_rate,
        ),
        "max_drawdown": max_drawdown(p_returns),
        "calmar_ratio": calmar_ratio(p_returns),
        "var_95": historical_var(
            p_returns,
            confidence=0.95,
        ),
        "cvar_95": historical_cvar(
            p_returns,
            confidence=0.95,
        ),
    }

    if benchmark is not None:
        summary["beta"] = beta(
            p_returns,
            benchmark,
        )
    else:
        summary["beta"] = np.nan

    summary["portfolio_volatility"] = annualized_volatility(
        p_returns
    )

    summary["risk_contribution"] = (
        risk_contribution(
            returns,
            weights,
        )
        .to_dict()
    )

    return summary