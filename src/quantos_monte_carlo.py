"""
QuantOS Monte Carlo Simulation Engine
=====================================

Purpose
-------
A practical forward-path simulation engine for QuantOS.

The engine is designed to answer:

1. What range of future outcomes is plausible?
2. What is the probability of reaching a target?
3. What is the probability of hitting a stop-loss?
4. What is the probability of finishing below the starting value?
5. What does the terminal return distribution look like?
6. What is the expected maximum drawdown?
7. How sensitive are outcomes to volatility, jumps and market regimes?
8. Do simulated 90% bands have reasonable historical coverage?

Supported models
----------------
1. Student-t benchmark
2. GARCH(1,1)-Student-t
3. GARCH(1,1)-Student-t + empirical Merton jumps
4. Moving block bootstrap
5. HMM regime switching

Important
---------
This module does NOT make investment recommendations.

It produces probabilistic scenarios and quantitative risk information
for a human decision-maker.

All simulations use reproducible random seeds.

Author: QuantOS
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf


# ============================================================================
# CONSTANTS
# ============================================================================

TRADING_DAYS = 252

DEFAULT_HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GARCHResult:
    """
    Fitted GARCH(1,1) model.
    """

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float

    last_variance: float
    unconditional_variance: float

    standardized_residuals: np.ndarray

    log_likelihood: float
    converged: bool


@dataclass
class JumpResult:
    """
    Empirical jump-process parameters.
    """

    lambda_daily: float
    jump_mean: float
    jump_std: float
    jump_count: int
    threshold_sigma: float


@dataclass
class SimulationResult:
    """
    Container for simulation outputs.
    """

    model: str
    paths: np.ndarray
    wealth_paths: np.ndarray
    terminal_returns: np.ndarray
    terminal_wealth: np.ndarray
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]


# ============================================================================
# BASIC DATA UTILITIES
# ============================================================================

def clean_returns(
    returns: pd.Series | pd.DataFrame | np.ndarray
) -> np.ndarray:
    """
    Convert returns into a clean NumPy array.

    Returns are expected in decimal form.

    Example:
        1% -> 0.01
        -2% -> -0.02
    """

    if isinstance(returns, pd.DataFrame):
        arr = returns.values.astype(float)

    elif isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float).reshape(-1, 1)

    else:
        arr = np.asarray(returns, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return arr


def prices_to_returns(
    prices: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """
    Convert price series/matrix to simple daily returns.
    """

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.copy()
    prices = prices.sort_index()

    returns = prices.pct_change()

    return returns.dropna(how="all")


def log_returns(
    prices: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """
    Convert prices to log returns.
    """

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.sort_index()

    return np.log(prices / prices.shift(1)).dropna(how="all")


def annualized_volatility(
    returns: np.ndarray
) -> float:
    """
    Annualized volatility.
    """

    r = np.asarray(returns, dtype=float)

    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


# ============================================================================
# COVARIANCE / CORRELATION
# ============================================================================

def ledoit_wolf_covariance(
    returns: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    This is used instead of raw sample covariance when multiple assets
    are simulated together.

    It is particularly useful when:
        - number of assets is non-trivial
        - correlations are unstable
        - assets have different volatility scales
    """

    arr = clean_returns(returns)

    model = LedoitWolf()
    model.fit(arr)

    covariance = model.covariance_

    return covariance


def safe_cholesky(
    covariance: np.ndarray,
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Robust Cholesky decomposition.

    Adds a tiny diagonal adjustment if numerical precision makes
    the covariance matrix non-positive-definite.
    """

    covariance = np.asarray(covariance, dtype=float)

    covariance = (
        covariance + covariance.T
    ) / 2.0

    try:
        return np.linalg.cholesky(covariance)

    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(covariance)

        eigvals = np.maximum(eigvals, jitter)

        repaired = (
            eigvecs
            @ np.diag(eigvals)
            @ eigvecs.T
        )

        repaired = (
            repaired + repaired.T
        ) / 2.0

        return np.linalg.cholesky(repaired)


# ============================================================================
# STUDENT-T INNOVATIONS
# ============================================================================

def standardized_student_t(
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    degrees_of_freedom: float,
) -> np.ndarray:
    """
    Generate Student-t innovations with approximately unit variance.

    Standard Student-t variance is:

        nu / (nu - 2)

    Therefore we rescale it so the resulting innovation has variance ~1.
    """

    nu = max(float(degrees_of_freedom), 2.05)

    z = rng.standard_t(
        df=nu,
        size=shape,
    )

    scale = np.sqrt((nu - 2.0) / nu)

    return z * scale


def multivariate_student_t_shocks(
    rng: np.random.Generator,
    n_paths: int,
    horizon: int,
    covariance: np.ndarray,
    degrees_of_freedom: float,
) -> np.ndarray:
    """
    Generate correlated multivariate Student-t shocks.

    Construction:

        Z ~ N(0, Sigma)

        U ~ ChiSquare(nu)

        T = Z / sqrt(U / nu)

    followed by variance normalization.
    """

    n_assets = covariance.shape[0]

    L = safe_cholesky(covariance)

    normal = rng.normal(
        size=(n_paths, horizon, n_assets)
    )

    correlated_normal = (
        normal @ L.T
    )

    chi = rng.chisquare(
        degrees_of_freedom,
        size=(n_paths, horizon, 1),
    )

    t_scale = np.sqrt(
        chi / degrees_of_freedom
    )

    shocks = correlated_normal / t_scale

    variance_adjustment = np.sqrt(
        (degrees_of_freedom - 2.0)
        / degrees_of_freedom
    )

    shocks *= variance_adjustment

    return shocks


# ============================================================================
# STUDENT-T BASELINE
# ============================================================================

def simulate_student_t(
    returns: pd.DataFrame | np.ndarray,
    horizon: int,
    n_paths: int = 10_000,
    degrees_of_freedom: float = 8.0,
    seed: int = 42,
    mean_shrink: float = 0.0,
) -> np.ndarray:
    """
    Static covariance Student-t Monte Carlo.

    This is deliberately retained as a benchmark.

    It does NOT attempt to model changing volatility.

    Parameters
    ----------
    mean_shrink:
        Shrinks historical mean toward zero.

        0.0 = full historical mean
        1.0 = zero expected daily return
    """

    arr = clean_returns(returns)

    mu = np.mean(arr, axis=0)

    mu = (
        1.0 - mean_shrink
    ) * mu

    covariance = ledoit_wolf_covariance(arr)

    rng = np.random.default_rng(seed)

    shocks = multivariate_student_t_shocks(
        rng=rng,
        n_paths=n_paths,
        horizon=horizon,
        covariance=covariance,
        degrees_of_freedom=degrees_of_freedom,
    )

    simulated = shocks + mu.reshape(1, 1, -1)

    return simulated


# ============================================================================
# GARCH(1,1)
# ============================================================================

def _garch_variance_path(
    residuals: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Compute GARCH conditional variance sequence.
    """

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    n = len(residuals)

    variance = np.zeros(n)

    initial_variance = np.var(
        residuals,
        ddof=1,
    )

    initial_variance = max(
        initial_variance,
        1e-12,
    )

    variance[0] = initial_variance

    for i in range(1, n):

        variance[i] = (
            omega
            + alpha * residuals[i - 1] ** 2
            + beta * variance[i - 1]
        )

        variance[i] = max(
            variance[i],
            1e-12,
        )

    return variance


def fit_garch(
    returns: pd.Series | np.ndarray,
) -> GARCHResult:
    """
    Fit a GARCH(1,1) model using Gaussian QMLE.

    Student-t degrees of freedom are estimated afterward from
    standardized residuals.

    Model:

        sigma_t^2 =
            omega
            + alpha * epsilon_(t-1)^2
            + beta * sigma_(t-1)^2
    """

    r = np.asarray(
        returns,
        dtype=float,
    ).reshape(-1)

    r = r[np.isfinite(r)]

    if len(r) < 250:
        raise ValueError(
            "GARCH requires at least 250 observations."
        )

    mu = float(np.mean(r))

    residuals = r - mu

    sample_variance = float(
        np.var(
            residuals,
            ddof=1,
        )
    )

    sample_variance = max(
        sample_variance,
        1e-10,
    )

    def objective(params):

        omega, alpha, beta = params

        if omega <= 0:
            return 1e20

        if alpha < 0 or beta < 0:
            return 1e20

        if alpha + beta >= 0.9999:
            return 1e20

        variance = _garch_variance_path(
            residuals,
            omega,
            alpha,
            beta,
        )

        ll = (
            np.log(variance)
            + residuals ** 2 / variance
        )

        return 0.5 * np.sum(ll)

    initial = np.array([
        sample_variance * 0.05,
        0.05,
        0.90,
    ])

    bounds = [
        (1e-12, sample_variance),
        (1e-6, 0.30),
        (1e-6, 0.999),
    ]

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 500,
        },
    )

    omega, alpha, beta = result.x

    variance = _garch_variance_path(
        residuals,
        omega,
        alpha,
        beta,
    )

    standardized = residuals / np.sqrt(
        variance
    )

    # ------------------------------------------------------------------
    # Estimate Student-t degrees of freedom
    # ------------------------------------------------------------------

    best_nu = 8.0
    best_ll = -np.inf

    candidate_nu = np.linspace(
        4.0,
        30.0,
        53,
    )

    for nu in candidate_nu:

        scale = np.sqrt(
            (nu - 2.0) / nu
        )

        density = (
            student_t.logpdf(
                standardized / scale,
                df=nu,
            )
            - np.log(scale)
        )

        ll = float(
            np.sum(density)
        )

        if ll > best_ll:

            best_ll = ll
            best_nu = float(nu)

    unconditional_variance = omega / max(
        1.0 - alpha - beta,
        1e-8,
    )

    log_likelihood = -float(
        objective(result.x)
    )

    return GARCHResult(
        mu=mu,
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        nu=float(best_nu),
        last_variance=float(variance[-1]),
        unconditional_variance=float(
            unconditional_variance
        ),
        standardized_residuals=standardized,
        log_likelihood=log_likelihood,
        converged=bool(result.success),
    )


# ============================================================================
# GARCH SIMULATION
# ============================================================================

def simulate_garch_t(
    returns: pd.Series | np.ndarray,
    horizon: int,
    n_paths: int = 10_000,
    seed: int = 42,
    atm_iv: Optional[float] = None,
    iv_weight: float = 0.50,
    mean_shrink: float = 0.0,
) -> Tuple[np.ndarray, GARCHResult]:
    """
    Simulate GARCH(1,1)-Student-t paths.

    If ATM IV is supplied:

        initial variance =
            (1 - iv_weight) * GARCH variance
            + iv_weight * IV variance

    ATM IV must be annualized decimal volatility.

    Example:

        18% IV -> 0.18
    """

    r = np.asarray(
        returns,
        dtype=float,
    ).reshape(-1)

    model = fit_garch(r)

    rng = np.random.default_rng(seed)

    mu = model.mu * (
        1.0 - mean_shrink
    )

    variance0 = model.last_variance

    # ---------------------------------------------------------------
    # Optional implied-volatility anchor
    # ---------------------------------------------------------------

    if atm_iv is not None:

        if not 0 < atm_iv < 5:
            raise ValueError(
                "ATM IV must be supplied as annualized decimal volatility."
            )

        iv_daily_variance = (
            atm_iv ** 2
            / TRADING_DAYS
        )

        iv_weight = float(
            np.clip(
                iv_weight,
                0.0,
                1.0,
            )
        )

        variance0 = (
            (1.0 - iv_weight)
            * variance0
            + iv_weight
            * iv_daily_variance
        )

    paths = np.zeros(
        (
            n_paths,
            horizon,
        ),
        dtype=float,
    )

    variance = np.full(
        n_paths,
        variance0,
        dtype=float,
    )

    for day in range(horizon):

        z = standardized_student_t(
            rng,
            (n_paths,),
            model.nu,
        )

        sigma = np.sqrt(
            np.maximum(
                variance,
                1e-12,
            )
        )

        daily_return = (
            mu
            + sigma * z
        )

        # Prevent impossible simple returns.
        daily_return = np.maximum(
            daily_return,
            -0.999,
        )

        paths[:, day] = daily_return

        residual = (
            daily_return
            - mu
        )

        variance = (
            model.omega
            + model.alpha
            * residual ** 2
            + model.beta
            * variance
        )

        variance = np.maximum(
            variance,
            1e-12,
        )

    return paths, model


# ============================================================================
# EMPIRICAL JUMP ESTIMATION
# ============================================================================

def estimate_jump_parameters(
    returns: pd.Series | np.ndarray,
    threshold_sigma: float = 3.0,
) -> JumpResult:
    """
    Estimate an empirical jump process.

    A jump is defined as an observation whose standardized return
    exceeds threshold_sigma standard deviations.

    This is intentionally empirical rather than pretending that a
    simple threshold estimator is a full MLE for a Merton model.
    """

    r = np.asarray(
        returns,
        dtype=float,
    ).reshape(-1)

    r = r[np.isfinite(r)]

    mu = np.mean(r)
    sigma = np.std(
        r,
        ddof=1,
    )

    sigma = max(
        sigma,
        1e-12,
    )

    standardized = (
        r - mu
    ) / sigma

    jump_mask = (
        np.abs(
            standardized
        )
        >= threshold_sigma
    )

    jumps = r[jump_mask]

    if len(jumps) == 0:

        return JumpResult(
            lambda_daily=0.0,
            jump_mean=0.0,
            jump_std=0.0,
            jump_count=0,
            threshold_sigma=threshold_sigma,
        )

    lambda_daily = (
        len(jumps)
        / len(r)
    )

    return JumpResult(
        lambda_daily=float(
            lambda_daily
        ),
        jump_mean=float(
            np.mean(jumps)
        ),
        jump_std=float(
            np.std(
                jumps,
                ddof=1,
            )
            if len(jumps) > 1
            else 0.0
        ),
        jump_count=int(
            len(jumps)
        ),
        threshold_sigma=threshold_sigma,
    )


# ============================================================================
# GARCH + MERTON JUMPS
# ============================================================================

def simulate_merton_jump_garch(
    returns: pd.Series | np.ndarray,
    horizon: int,
    n_paths: int = 10_000,
    seed: int = 42,
    atm_iv: Optional[float] = None,
    iv_weight: float = 0.50,
    threshold_sigma: float = 3.0,
) -> Tuple[
    np.ndarray,
    GARCHResult,
    JumpResult,
]:
    """
    Simulate:

        GARCH volatility
        +
        Student-t innovations
        +
        empirical Merton-style jumps

    This is intended to be the primary single-asset QuantOS
    forward-path model.
    """

    r = np.asarray(
        returns,
        dtype=float,
    ).reshape(-1)

    garch = fit_garch(r)

    jumps = estimate_jump_parameters(
        r,
        threshold_sigma=threshold_sigma,
    )

    rng = np.random.default_rng(
        seed
    )

    variance0 = garch.last_variance

    if atm_iv is not None:

        iv_daily_variance = (
            atm_iv ** 2
            / TRADING_DAYS
        )

        iv_weight = float(
            np.clip(
                iv_weight,
                0.0,
                1.0,
            )
        )

        variance0 = (
            (1.0 - iv_weight)
            * variance0
            + iv_weight
            * iv_daily_variance
        )

    paths = np.zeros(
        (
            n_paths,
            horizon,
        ),
        dtype=float,
    )

    variance = np.full(
        n_paths,
        variance0,
    )

    # Jump compensation.
    #
    # This prevents the jump component from mechanically shifting
    # expected return simply because jumps were added.
    if jumps.lambda_daily > 0:

        jump_compensation = (
            jumps.lambda_daily
            * (
                np.exp(
                    jumps.jump_mean
                    + 0.5
                    * jumps.jump_std ** 2
                )
                - 1.0
            )
        )

    else:

        jump_compensation = 0.0

    for day in range(horizon):

        z = standardized_student_t(
            rng,
            (n_paths,),
            garch.nu,
        )

        sigma = np.sqrt(
            np.maximum(
                variance,
                1e-12,
            )
        )

        diffusion = (
            garch.mu
            - jump_compensation
            + sigma * z
        )

        # ---------------------------------------------------------------
        # Poisson jump arrivals
        # ---------------------------------------------------------------

        jump_count = rng.poisson(
            jumps.lambda_daily,
            size=n_paths,
        )

        jump_component = np.zeros(
            n_paths
        )

        active = (
            jump_count > 0
        )

        if np.any(active):

            k = jump_count[active]

            jump_component[active] = (
                k * jumps.jump_mean
                + np.sqrt(k)
                * jumps.jump_std
                * rng.normal(
                    size=np.sum(active)
                )
            )

        daily_return = (
            diffusion
            + jump_component
        )

        daily_return = np.maximum(
            daily_return,
            -0.999,
        )

        paths[:, day] = daily_return

        residual = (
            daily_return
            - garch.mu
            - jump_component
        )

        variance = (
            garch.omega
            + garch.alpha
            * residual ** 2
            + garch.beta
            * variance
        )

        variance = np.maximum(
            variance,
            1e-12,
        )

    return (
        paths,
        garch,
        jumps,
    )


# ============================================================================
# MOVING BLOCK BOOTSTRAP
# ============================================================================

def simulate_block_bootstrap(
    returns: pd.DataFrame | np.ndarray,
    horizon: int,
    n_paths: int = 10_000,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Moving block bootstrap.

    Entire return vectors are sampled together.

    This preserves:
        - cross-asset correlation
        - empirical fat tails
        - local volatility clustering
        - historical dependence within blocks
    """

    arr = clean_returns(
        returns
    )

    n_obs, n_assets = arr.shape

    if n_obs < block_size + 10:
        raise ValueError(
            "Not enough observations for requested bootstrap block size."
        )

    rng = np.random.default_rng(
        seed
    )

    max_start = (
        n_obs
        - block_size
    )

    n_blocks = int(
        np.ceil(
            horizon
            / block_size
        )
    )

    output = np.empty(
        (
            n_paths,
            n_blocks * block_size,
            n_assets,
        ),
        dtype=float,
    )

    for block in range(
        n_blocks
    ):

        starts = rng.integers(
            0,
            max_start + 1,
            size=n_paths,
        )

        for path in range(
            n_paths
        ):

            output[
                path,
                block * block_size:
                (block + 1) * block_size,
                :
            ] = arr[
                starts[path]:
                starts[path] + block_size,
                :
            ]

    return output[
        :,
        :horizon,
        :
    ]


# ============================================================================
# HMM REGIME STATISTICS
# ============================================================================

def fit_regime_statistics(
    returns: pd.DataFrame | np.ndarray,
    regime_labels: np.ndarray,
    n_states: int = 3,
) -> Dict[int, Dict[str, Any]]:
    """
    Estimate mean/covariance for each regime.

    Small regimes fall back toward global covariance through
    Ledoit-Wolf estimation.
    """

    arr = clean_returns(
        returns
    )

    labels = np.asarray(
        regime_labels
    ).reshape(-1)

    if len(labels) != len(arr):
        raise ValueError(
            "regime_labels and returns must have identical length."
        )

    global_mean = np.mean(
        arr,
        axis=0,
    )

    global_cov = ledoit_wolf_covariance(
        arr
    )

    results = {}

    for state in range(
        n_states
    ):

        mask = (
            labels == state
        )

        state_returns = arr[
            mask
        ]

        if len(state_returns) < 30:

            results[state] = {
                "mean": global_mean.copy(),
                "covariance": global_cov.copy(),
                "observations": int(
                    len(state_returns)
                ),
                "fallback": True,
            }

            continue

        state_mean = np.mean(
            state_returns,
            axis=0,
        )

        state_cov = ledoit_wolf_covariance(
            state_returns
        )

        results[state] = {
            "mean": state_mean,
            "covariance": state_cov,
            "observations": int(
                len(state_returns)
            ),
            "fallback": False,
        }

    return results


# ============================================================================
# HMM REGIME-SWITCHING SIMULATION
# ============================================================================

def simulate_hmm_regime_switching(
    returns: pd.DataFrame | np.ndarray,
    regime_labels: np.ndarray,
    transition_matrix: np.ndarray,
    current_probabilities: np.ndarray,
    horizon: int,
    n_paths: int = 10_000,
    degrees_of_freedom: float = 8.0,
    seed: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    HMM-conditioned Monte Carlo simulation.

    Each simulated day:

        current regime
            ↓
        transition matrix
            ↓
        sampled next regime
            ↓
        state-specific mean/covariance
            ↓
        correlated Student-t return

    This creates actual regime-dependent future paths rather than
    merely printing the current HMM label beside a Monte Carlo chart.
    """

    arr = clean_returns(
        returns
    )

    transition = np.asarray(
        transition_matrix,
        dtype=float,
    )

    current_probabilities = np.asarray(
        current_probabilities,
        dtype=float,
    )

    transition = np.clip(
        transition,
        0.0,
        None,
    )

    transition /= np.maximum(
        transition.sum(axis=1, keepdims=True),
        1e-12,
    )

    current_probabilities = np.clip(
        current_probabilities,
        0.0,
        None,
    )

    current_probabilities /= max(
        current_probabilities.sum(),
        1e-12,
    )

    n_states = transition.shape[0]
    n_assets = arr.shape[1]

    stats = fit_regime_statistics(
        arr,
        regime_labels,
        n_states=n_states,
    )

    rng = np.random.default_rng(
        seed
    )

    paths = np.zeros(
        (
            n_paths,
            horizon,
            n_assets,
        )
    )

    state_paths = np.zeros(
        (
            n_paths,
            horizon,
        ),
        dtype=int,
    )

    # ---------------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------------

    initial_states = rng.choice(
        n_states,
        size=n_paths,
        p=current_probabilities,
    )

    current_states = initial_states

    for day in range(
        horizon
    ):

        next_states = np.zeros(
            n_paths,
            dtype=int,
        )

        for state in range(
            n_states
        ):

            mask = (
                current_states
                == state
            )

            count = int(
                np.sum(mask)
            )

            if count == 0:
                continue

            next_states[mask] = (
                np.array([
                    rng.choice(
                        n_states,
                        p=transition[state],
                    )
                    for _ in range(count)
                ])
            )

        current_states = next_states

        state_paths[
            :,
            day
        ] = current_states

        # -----------------------------------------------------------
        # Simulate each regime separately
        # -----------------------------------------------------------

        for state in range(
            n_states
        ):

            mask = (
                current_states
                == state
            )

            count = int(
                np.sum(mask)
            )

            if count == 0:
                continue

            covariance = stats[
                state
            ]["covariance"]

            mean = stats[
                state
            ]["mean"]

            shocks = multivariate_student_t_shocks(
                rng=rng,
                n_paths=count,
                horizon=1,
                covariance=covariance,
                degrees_of_freedom=degrees_of_freedom,
            )[:, 0, :]

            state_returns = (
                mean
                + shocks
            )

            state_returns = np.maximum(
                state_returns,
                -0.999,
            )

            paths[
                mask,
                day,
                :
            ] = state_returns

    return (
        paths,
        state_paths,
    )


# ============================================================================
# PATH / WEALTH CONSTRUCTION
# ============================================================================

def returns_to_paths(
    simulated_returns: np.ndarray,
    initial_value: float = 1.0,
) -> np.ndarray:
    """
    Convert daily simple returns into price/index paths.
    """

    r = np.asarray(
        simulated_returns,
        dtype=float,
    )

    if r.ndim == 2:

        growth = (
            1.0 + r
        )

        return initial_value * np.cumprod(
            growth,
            axis=1,
        )

    if r.ndim == 3:

        growth = (
            1.0 + r
        )

        return initial_value * np.cumprod(
            growth,
            axis=1,
        )

    raise ValueError(
        "simulated_returns must be 2D or 3D."
    )


def apply_cashflows(
    portfolio_returns: np.ndarray,
    initial_capital: float = 1_000_000.0,
    monthly_sip: float = 0.0,
    monthly_withdrawal: float = 0.0,
) -> np.ndarray:
    """
    Apply monthly SIP/withdrawal cash flows to simulated wealth.

    Cash flows are applied approximately every 21 trading days.

    Positive:
        monthly_sip

    Negative:
        monthly_withdrawal
    """

    r = np.asarray(
        portfolio_returns,
        dtype=float,
    )

    if r.ndim != 2:
        raise ValueError(
            "portfolio_returns must be 2D."
        )

    n_paths, horizon = r.shape

    wealth = np.zeros(
        (
            n_paths,
            horizon,
        ),
        dtype=float,
    )

    current = np.full(
        n_paths,
        initial_capital,
        dtype=float,
    )

    for day in range(
        horizon
    ):

        current *= (
            1.0 + r[:, day]
        )

        # Apply monthly cash flow after the daily return.
        if (
            day > 0
            and day % 21 == 0
        ):

            current += (
                monthly_sip
                - monthly_withdrawal
            )

            current = np.maximum(
                current,
                0.0,
            )

        wealth[
            :,
            day
        ] = current

    return wealth


# ============================================================================
# PORTFOLIO AGGREGATION
# ============================================================================

def portfolio_returns_from_assets(
    asset_returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Convert multi-asset simulated returns into portfolio returns.

    Parameters
    ----------
    asset_returns:
        Shape:
            (paths, horizon, assets)

    weights:
        Shape:
            (assets,)
    """

    asset_returns = np.asarray(
        asset_returns,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if asset_returns.ndim != 3:
        raise ValueError(
            "asset_returns must be 3D."
        )

    if asset_returns.shape[2] != len(weights):
        raise ValueError(
            "Number of weights must match number of assets."
        )

    weights = weights / np.sum(
        weights
    )

    return np.sum(
        asset_returns
        * weights.reshape(1, 1, -1),
        axis=2,
    )


# ============================================================================
# RISK STATISTICS
# ============================================================================

def max_drawdown(
    wealth_paths: np.ndarray,
) -> np.ndarray:
    """
    Maximum drawdown for each simulation path.
    """

    wealth_paths = np.asarray(
        wealth_paths,
        dtype=float,
    )

    running_max = np.maximum.accumulate(
        wealth_paths,
        axis=1,
    )

    drawdown = (
        wealth_paths
        / np.maximum(
            running_max,
            1e-12,
        )
        - 1.0
    )

    return np.min(
        drawdown,
        axis=1,
    )


def var_cvar(
    returns: np.ndarray,
    confidence: float,
) -> Tuple[float, float]:
    """
    Historical Monte Carlo VaR and CVaR.

    Returned as positive loss percentages.

    Example:

        VaR = 0.10

    means approximately a 10% loss threshold.
    """

    alpha = 1.0 - confidence

    var_quantile = np.quantile(
        returns,
        alpha,
    )

    tail = returns[
        returns <= var_quantile
    ]

    if len(tail) == 0:

        cvar = var_quantile

    else:

        cvar = np.mean(
            tail
        )

    return (
        float(-var_quantile),
        float(-cvar),
    )


def summarize_simulation(
    paths: np.ndarray,
    initial_value: float = 1.0,
    target_pct: Optional[float] = None,
    stop_pct: Optional[float] = None,
    wealth_paths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Convert raw simulation paths into decision-useful statistics.

    The summary deliberately contains distributions rather than a
    single point prediction.
    """

    paths = np.asarray(
        paths,
        dtype=float,
    )

    if paths.ndim != 2:
        raise ValueError(
            "summarize_simulation expects 2D paths."
        )

    if wealth_paths is None:

        wealth_paths = paths

    terminal = paths[:, -1]

    terminal_returns = (
        terminal
        / initial_value
        - 1.0
    )

    mdd = max_drawdown(
        wealth_paths
    )

    var95, cvar95 = var_cvar(
        terminal_returns,
        0.95,
    )

    var99, cvar99 = var_cvar(
        terminal_returns,
        0.99,
    )

    statistics = {

        # -----------------------------------------------------------
        # Terminal distribution
        # -----------------------------------------------------------

        "terminal_p05": float(
            np.quantile(
                terminal_returns,
                0.05,
            )
        ),

        "terminal_p25": float(
            np.quantile(
                terminal_returns,
                0.25,
            )
        ),

        "terminal_median": float(
            np.quantile(
                terminal_returns,
                0.50,
            )
        ),

        "terminal_p75": float(
            np.quantile(
                terminal_returns,
                0.75,
            )
        ),

        "terminal_p95": float(
            np.quantile(
                terminal_returns,
                0.95,
            )
        ),

        "terminal_mean": float(
            np.mean(
                terminal_returns
            )
        ),

        # -----------------------------------------------------------
        # Tail risk
        # -----------------------------------------------------------

        "VaR_95": var95,
        "CVaR_95": cvar95,
        "VaR_99": var99,
        "CVaR_99": cvar99,

        # -----------------------------------------------------------
        # Drawdown
        # -----------------------------------------------------------

        "max_drawdown_mean": float(
            np.mean(mdd)
        ),

        "max_drawdown_p05": float(
            np.quantile(
                mdd,
                0.05,
            )
        ),

        "max_drawdown_median": float(
            np.quantile(
                mdd,
                0.50,
            )
        ),

        "max_drawdown_p95": float(
            np.quantile(
                mdd,
                0.95,
            )
        ),

        # -----------------------------------------------------------
        # Downside probability
        # -----------------------------------------------------------

        "prob_finish_below_start": float(
            np.mean(
                terminal_returns < 0.0
            )
        ),

        # -----------------------------------------------------------
        # Expected terminal wealth
        # -----------------------------------------------------------

        "terminal_wealth_mean": float(
            np.mean(
                wealth_paths[:, -1]
            )
        ),

        "terminal_wealth_median": float(
            np.median(
                wealth_paths[:, -1]
            )
        ),

        "terminal_wealth_p05": float(
            np.quantile(
                wealth_paths[:, -1],
                0.05,
            )
        ),

        "terminal_wealth_p95": float(
            np.quantile(
                wealth_paths[:, -1],
                0.95,
            )
        ),
    }

    # ---------------------------------------------------------------
    # Path-dependent target / stop probabilities
    # ---------------------------------------------------------------

    if target_pct is not None:

        target_level = (
            initial_value
            * (1.0 + target_pct)
        )

        hit_target = np.any(
            paths >= target_level,
            axis=1,
        )

        statistics[
            "prob_hit_target"
        ] = float(
            np.mean(
                hit_target
            )
        )

    else:

        statistics[
            "prob_hit_target"
        ] = None

    if stop_pct is not None:

        stop_level = (
            initial_value
            * (1.0 - stop_pct)
        )

        hit_stop = np.any(
            paths <= stop_level,
            axis=1,
        )

        statistics[
            "prob_hit_stop"
        ] = float(
            np.mean(
                hit_stop
            )
        )

    else:

        statistics[
            "prob_hit_stop"
        ] = None

    return statistics


# ============================================================================
# FAN CHART DATA
# ============================================================================

def fan_chart_dataframe(
    paths: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """
    Create percentile data for the dashboard fan chart.

    Percentiles:

        5%
        25%
        50%
        75%
        95%
    """

    paths = np.asarray(
        paths,
        dtype=float,
    )

    if dates is None:

        dates = pd.RangeIndex(
            start=1,
            stop=paths.shape[1] + 1,
        )

    return pd.DataFrame(
        {
            "date": dates,

            "p05": initial_value
            * np.quantile(
                paths,
                0.05,
                axis=0,
            ),

            "p25": initial_value
            * np.quantile(
                paths,
                0.25,
                axis=0,
            ),

            "p50": initial_value
            * np.quantile(
                paths,
                0.50,
                axis=0,
            ),

            "p75": initial_value
            * np.quantile(
                paths,
                0.75,
                axis=0,
            ),

            "p95": initial_value
            * np.quantile(
                paths,
                0.95,
                axis=0,
            ),
        }
    )


# ============================================================================
# COMPLETE MODEL DISPATCHER
# ============================================================================

def simulate_model(
    model: str,
    returns: pd.Series | pd.DataFrame | np.ndarray,
    horizon: int,
    n_paths: int = 10_000,
    seed: int = 42,
    atm_iv: Optional[float] = None,
    iv_weight: float = 0.50,
    block_size: int = 20,
    regime_labels: Optional[np.ndarray] = None,
    transition_matrix: Optional[np.ndarray] = None,
    current_probabilities: Optional[np.ndarray] = None,
    degrees_of_freedom: float = 8.0,
) -> SimulationResult:
    """
    Unified Monte Carlo interface.

    Models:

        "student_t"
        "garch_t"
        "garch_t_jump"
        "bootstrap"
        "hmm"
    """

    model_key = (
        model
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    metadata = {
        "model": model_key,
        "horizon_days": int(horizon),
        "paths": int(n_paths),
        "seed": int(seed),
    }

    # ================================================================
    # STUDENT-T
    # ================================================================

    if model_key in {
        "student_t",
        "student_t_benchmark",
    }:

        simulated = simulate_student_t(
            returns=returns,
            horizon=horizon,
            n_paths=n_paths,
            degrees_of_freedom=degrees_of_freedom,
            seed=seed,
        )

        metadata[
            "degrees_of_freedom"
        ] = degrees_of_freedom

    # ================================================================
    # GARCH-T
    # ================================================================

    elif model_key in {
        "garch_t",
        "garch",
    }:

        if isinstance(
            returns,
            pd.DataFrame,
        ):

            if returns.shape[1] != 1:

                raise ValueError(
                    "GARCH-T is a single-asset model. "
                    "Use portfolio aggregation or HMM/multivariate models "
                    "for multiple assets."
                )

            returns_1d = returns.iloc[:, 0]

        else:

            returns_1d = returns

        simulated, garch = simulate_garch_t(
            returns=returns_1d,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
            atm_iv=atm_iv,
            iv_weight=iv_weight,
        )

        metadata.update(
            {
                "garch_mu": garch.mu,
                "garch_omega": garch.omega,
                "garch_alpha": garch.alpha,
                "garch_beta": garch.beta,
                "garch_nu": garch.nu,
                "garch_current_vol": np.sqrt(
                    garch.last_variance
                ) * np.sqrt(TRADING_DAYS),
            }
        )

    # ================================================================
    # GARCH + JUMPS
    # ================================================================

    elif model_key in {
        "garch_t_jump",
        "garch_jump",
        "merton_jump_garch",
    }:

        if isinstance(
            returns,
            pd.DataFrame,
        ):

            if returns.shape[1] != 1:

                raise ValueError(
                    "GARCH + jumps is a single-asset model."
                )

            returns_1d = returns.iloc[:, 0]

        else:

            returns_1d = returns

        (
            simulated,
            garch,
            jumps,
        ) = simulate_merton_jump_garch(
            returns=returns_1d,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
            atm_iv=atm_iv,
            iv_weight=iv_weight,
        )

        metadata.update(
            {
                "garch_nu": garch.nu,
                "garch_current_vol": np.sqrt(
                    garch.last_variance
                ) * np.sqrt(TRADING_DAYS),
                "jump_lambda_daily": jumps.lambda_daily,
                "jump_mean": jumps.jump_mean,
                "jump_std": jumps.jump_std,
                "jump_count": jumps.jump_count,
            }
        )

    # ================================================================
    # BLOCK BOOTSTRAP
    # ================================================================

    elif model_key in {
        "bootstrap",
        "block_bootstrap",
    }:

        simulated = simulate_block_bootstrap(
            returns=returns,
            horizon=horizon,
            n_paths=n_paths,
            block_size=block_size,
            seed=seed,
        )

        metadata[
            "block_size"
        ] = block_size

    # ================================================================
    # HMM
    # ================================================================

    elif model_key in {
        "hmm",
        "hmm_regime",
        "regime_switching",
    }:

        if regime_labels is None:
            raise ValueError(
                "HMM simulation requires regime_labels."
            )

        if transition_matrix is None:
            raise ValueError(
                "HMM simulation requires transition_matrix."
            )

        if current_probabilities is None:
            raise ValueError(
                "HMM simulation requires current_probabilities."
            )

        (
            simulated,
            state_paths,
        ) = simulate_hmm_regime_switching(
            returns=returns,
            regime_labels=regime_labels,
            transition_matrix=transition_matrix,
            current_probabilities=current_probabilities,
            horizon=horizon,
            n_paths=n_paths,
            degrees_of_freedom=degrees_of_freedom,
            seed=seed,
        )

        metadata[
            "state_paths"
        ] = state_paths

    else:

        raise ValueError(
            f"Unknown Monte Carlo model: {model}"
        )

    # ================================================================
    # PORTFOLIO / SINGLE-ASSET CONVERSION
    # ================================================================

    if simulated.ndim == 3:

        # Equal-weight fallback.
        #
        # The dashboard can replace this with user-selected weights
        # before calling summarize_simulation.
        n_assets = simulated.shape[2]

        weights = np.ones(
            n_assets
        ) / n_assets

        portfolio_returns = (
            portfolio_returns_from_assets(
                simulated,
                weights,
            )
        )

    else:

        portfolio_returns = simulated

    paths = returns_to_paths(
        portfolio_returns,
        initial_value=1.0,
    )

    wealth = apply_cashflows(
        portfolio_returns,
        initial_capital=1.0,
    )

    statistics = summarize_simulation(
        paths=paths,
        initial_value=1.0,
        wealth_paths=wealth,
    )

    terminal_returns = (
        paths[:, -1] - 1.0
    )

    terminal_wealth = (
        wealth[:, -1]
    )

    return SimulationResult(
        model=model_key,
        paths=paths,
        wealth_paths=wealth,
        terminal_returns=terminal_returns,
        terminal_wealth=terminal_wealth,
        statistics=statistics,
        metadata=metadata,
    )


# ============================================================================
# USER-FACING PORTFOLIO SIMULATION
# ============================================================================

def simulate_portfolio(
    returns: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    model: str = "garch_t",
    horizon: int = 63,
    n_paths: int = 10_000,
    initial_capital: float = 1_000_000.0,
    monthly_sip: float = 0.0,
    monthly_withdrawal: float = 0.0,
    target_pct: Optional[float] = None,
    stop_pct: Optional[float] = None,
    seed: int = 42,
    atm_iv: Optional[float] = None,
    iv_weight: float = 0.50,
    block_size: int = 20,
    regime_labels: Optional[np.ndarray] = None,
    transition_matrix: Optional[np.ndarray] = None,
    current_probabilities: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    Main portfolio-level simulation entry point.

    For multi-asset portfolios, the preferred models are:

        - Student-t benchmark
        - Block bootstrap
        - HMM regime switching

    GARCH-T and GARCH-T + jumps are intended primarily for
    single-underlying simulations.
    """

    if not isinstance(
        returns,
        pd.DataFrame,
    ):

        raise TypeError(
            "simulate_portfolio expects a pandas DataFrame."
        )

    returns = returns.dropna()

    if len(returns) < 250:

        raise ValueError(
            "At least 250 daily observations are required."
        )

    n_assets = returns.shape[1]

    if weights is None:

        weights = np.ones(
            n_assets
        ) / n_assets

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if len(weights) != n_assets:

        raise ValueError(
            "Number of weights must match number of assets."
        )

    if np.sum(
        np.abs(weights)
    ) == 0:

        raise ValueError(
            "Portfolio weights cannot all be zero."
        )

    weights = (
        weights
        / np.sum(weights)
    )

    model_key = (
        model.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    # ================================================================
    # SINGLE ASSET GARCH MODELS
    # ================================================================

    if model_key in {
        "garch_t",
        "garch",
        "garch_t_jump",
        "garch_jump",
        "merton_jump_garch",
    }:

        if n_assets != 1:

            raise ValueError(
                "GARCH models in this engine operate on one underlying "
                "at a time. Use Student-t, bootstrap or HMM for "
                "multi-asset portfolio simulation."
            )

        result = simulate_model(
            model=model_key,
            returns=returns.iloc[:, 0],
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
            atm_iv=atm_iv,
            iv_weight=iv_weight,
            block_size=block_size,
            regime_labels=regime_labels,
            transition_matrix=transition_matrix,
            current_probabilities=current_probabilities,
        )

        portfolio_returns = (
            result.paths
            / np.vstack(
                [
                    np.ones(n_paths),
                    result.paths[:, :-1],
                ]
            )
            - 1.0
        )

    # ================================================================
    # MULTI-ASSET MODELS
    # ================================================================

    else:

        simulated = simulate_model(
            model=model_key,
            returns=returns,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
            atm_iv=atm_iv,
            iv_weight=iv_weight,
            block_size=block_size,
            regime_labels=regime_labels,
            transition_matrix=transition_matrix,
            current_probabilities=current_probabilities,
        )

        # The generic dispatcher defaults to equal weights internally,
        # so for portfolio simulation we regenerate the underlying
        # simulation when custom weights are required.

        if model_key in {
            "student_t",
            "student_t_benchmark",
        }:

            asset_returns = simulate_student_t(
                returns=returns,
                horizon=horizon,
                n_paths=n_paths,
                seed=seed,
            )

        elif model_key in {
            "bootstrap",
            "block_bootstrap",
        }:

            asset_returns = simulate_block_bootstrap(
                returns=returns,
                horizon=horizon,
                n_paths=n_paths,
                block_size=block_size,
                seed=seed,
            )

        elif model_key in {
            "hmm",
            "hmm_regime",
            "regime_switching",
        }:

            asset_returns, _ = (
                simulate_hmm_regime_switching(
                    returns=returns,
                    regime_labels=regime_labels,
                    transition_matrix=transition_matrix,
                    current_probabilities=current_probabilities,
                    horizon=horizon,
                    n_paths=n_paths,
                    seed=seed,
                )
            )

        else:

            raise ValueError(
                f"Unsupported portfolio model: {model}"
            )

        portfolio_returns = (
            portfolio_returns_from_assets(
                asset_returns,
                weights,
            )
        )

        paths = returns_to_paths(
            portfolio_returns,
            initial_value=1.0,
        )

        wealth = apply_cashflows(
            portfolio_returns,
            initial_capital=initial_capital,
            monthly_sip=monthly_sip,
            monthly_withdrawal=monthly_withdrawal,
        )

        statistics = summarize_simulation(
            paths=paths,
            initial_value=1.0,
            target_pct=target_pct,
            stop_pct=stop_pct,
            wealth_paths=wealth,
        )

        statistics[
            "initial_capital"
        ] = initial_capital

        statistics[
            "monthly_sip"
        ] = monthly_sip

        statistics[
            "monthly_withdrawal"
        ] = monthly_withdrawal

        return SimulationResult(
            model=model_key,
            paths=paths,
            wealth_paths=wealth,
            terminal_returns=(
                paths[:, -1] - 1.0
            ),
            terminal_wealth=(
                wealth[:, -1]
            ),
            statistics=statistics,
            metadata={
                "assets": list(
                    returns.columns
                ),
                "weights": weights.tolist(),
                "horizon_days": horizon,
                "n_paths": n_paths,
                "seed": seed,
            },
        )

    # ================================================================
    # SINGLE-ASSET CASHFLOW + TARGET/STOP HANDLING
    # ================================================================

    single_returns = (
        portfolio_returns
    )

    wealth = apply_cashflows(
        single_returns,
        initial_capital=initial_capital,
        monthly_sip=monthly_sip,
        monthly_withdrawal=monthly_withdrawal,
    )

    price_paths = returns_to_paths(
        single_returns,
        initial_value=1.0,
    )

    statistics = summarize_simulation(
        paths=price_paths,
        initial_value=1.0,
        target_pct=target_pct,
        stop_pct=stop_pct,
        wealth_paths=wealth,
    )

    statistics[
        "initial_capital"
    ] = initial_capital

    statistics[
        "monthly_sip"
    ] = monthly_sip

    statistics[
        "monthly_withdrawal"
    ] = monthly_withdrawal

    return SimulationResult(
        model=model_key,
        paths=price_paths,
        wealth_paths=wealth,
        terminal_returns=(
            price_paths[:, -1] - 1.0
        ),
        terminal_wealth=(
            wealth[:, -1]
        ),
        statistics=statistics,
        metadata={
            "assets": list(
                returns.columns
            ),
            "weights": weights.tolist(),
            "horizon_days": horizon,
            "n_paths": n_paths,
            "seed": seed,
        },
    )


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    returns: pd.Series | pd.DataFrame,
    horizon: int,
    models: Optional[list[str]] = None,
    n_paths: int = 10_000,
    seed: int = 42,
    atm_iv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run several simulation models using the same historical data.

    This is important because the point is not to declare one model
    magically correct. The user should be able to see how assumptions
    change the distribution of outcomes.
    """

    if models is None:

        models = [
            "student_t",
            "garch_t",
            "garch_t_jump",
            "bootstrap",
        ]

    results = []

    for i, model in enumerate(
        models
    ):

        result = simulate_model(
            model=model,
            returns=returns,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed + i,
            atm_iv=atm_iv,
        )

        s = result.statistics

        results.append(
            {
                "Model": model,
                "P05": s[
                    "terminal_p05"
                ],
                "P25": s[
                    "terminal_p25"
                ],
                "Median": s[
                    "terminal_median"
                ],
                "P75": s[
                    "terminal_p75"
                ],
                "P95": s[
                    "terminal_p95"
                ],
                "VaR95": s[
                    "VaR_95"
                ],
                "CVaR95": s[
                    "CVaR_95"
                ],
                "VaR99": s[
                    "VaR_99"
                ],
                "CVaR99": s[
                    "CVaR_99"
                ],
                "Mean_Max_Drawdown": s[
                    "max_drawdown_mean"
                ],
                "P_Below_Start": s[
                    "prob_finish_below_start"
                ],
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================================
# HISTORICAL 90% BAND VALIDATION
# ============================================================================

def validate_90pct_band(
    returns: pd.Series,
    horizon: int = 21,
    model: str = "garch_t",
    n_paths: int = 2_000,
    step: int = 21,
    min_train: int = 750,
    seed: int = 42,
    atm_iv_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Rolling-origin validation of the 90% Monte Carlo interval.

    Procedure:

        1. Train only on data available at the forecast origin.
        2. Simulate future horizon.
        3. Calculate 5th/95th percentile terminal outcomes.
        4. Observe the actual future outcome.
        5. Repeat through history.

    A nominal 90% interval should be interpreted as a calibration
    target, not a guarantee.

    The result reports:
        - empirical coverage
        - expected nominal coverage
        - number of forecast origins
        - lower/upper violations
    """

    r = pd.Series(
        returns
    ).dropna()

    if len(r) < (
        min_train
        + horizon
    ):

        raise ValueError(
            "Not enough history for requested validation."
        )

    observations = []

    origin_indices = range(
        min_train,
        len(r) - horizon + 1,
        step,
    )

    for origin_number, origin in enumerate(
        origin_indices
    ):

        train = r.iloc[
            :origin
        ]

        future = r.iloc[
            origin:
            origin + horizon
        ]

        if len(future) < horizon:
            break

        iv = None

        if atm_iv_series is not None:

            try:

                iv = float(
                    atm_iv_series.iloc[
                        origin - 1
                    ]
                )

            except Exception:
                iv = None

        result = simulate_model(
            model=model,
            returns=train,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed + origin_number,
            atm_iv=iv,
        )

        simulated_terminal = (
            result.paths[:, -1]
            - 1.0
        )

        lower = float(
            np.quantile(
                simulated_terminal,
                0.05,
            )
        )

        upper = float(
            np.quantile(
                simulated_terminal,
                0.95,
            )
        )

        actual = float(
            np.prod(
                1.0
                + future.values
            )
            - 1.0
        )

        inside = (
            lower <= actual <= upper
        )

        observations.append(
            {
                "origin": r.index[
                    origin - 1
                ],
                "actual": actual,
                "lower_5": lower,
                "upper_95": upper,
                "inside_90pct_band": bool(
                    inside
                ),
            }
        )

    validation_df = pd.DataFrame(
        observations
    )

    if validation_df.empty:

        raise ValueError(
            "No validation observations were generated."
        )

    coverage = float(
        validation_df[
            "inside_90pct_band"
        ].mean()
    )

    lower_violations = int(
        np.sum(
            validation_df["actual"]
            < validation_df["lower_5"]
        )
    )

    upper_violations = int(
        np.sum(
            validation_df["actual"]
            > validation_df["upper_95"]
        )
    )

    return {
        "model": model,
        "horizon_days": horizon,
        "nominal_coverage": 0.90,
        "observed_coverage": coverage,
        "forecast_origins": len(
            validation_df
        ),
        "lower_violations": lower_violations,
        "upper_violations": upper_violations,
        "validation": validation_df,
    }


# ============================================================================
# CONVENIENCE HELPERS
# ============================================================================

def horizon_from_label(
    label: str
) -> int:
    """
    Convert UI horizon labels into trading days.
    """

    key = (
        label
        .strip()
        .upper()
    )

    if key not in DEFAULT_HORIZONS:

        raise ValueError(
            f"Unsupported horizon: {label}. "
            f"Use one of {list(DEFAULT_HORIZONS)}."
        )

    return DEFAULT_HORIZONS[
        key
    ]


def percentiles_to_dataframe(
    result: SimulationResult,
) -> pd.DataFrame:
    """
    Convert simulation output into a dashboard-friendly summary table.
    """

    s = result.statistics

    return pd.DataFrame(
        [
            {
                "Metric": "5th Percentile",
                "Value": s[
                    "terminal_p05"
                ],
            },
            {
                "Metric": "25th Percentile",
                "Value": s[
                    "terminal_p25"
                ],
            },
            {
                "Metric": "Median",
                "Value": s[
                    "terminal_median"
                ],
            },
            {
                "Metric": "75th Percentile",
                "Value": s[
                    "terminal_p75"
                ],
            },
            {
                "Metric": "95th Percentile",
                "Value": s[
                    "terminal_p95"
                ],
            },
            {
                "Metric": "VaR 95%",
                "Value": s[
                    "VaR_95"
                ],
            },
            {
                "Metric": "CVaR 95%",
                "Value": s[
                    "CVaR_95"
                ],
            },
            {
                "Metric": "VaR 99%",
                "Value": s[
                    "VaR_99"
                ],
            },
            {
                "Metric": "CVaR 99%",
                "Value": s[
                    "CVaR_99"
                ],
            },
            {
                "Metric": "Mean Maximum Drawdown",
                "Value": s[
                    "max_drawdown_mean"
                ],
            },
            {
                "Metric": "Probability Below Start",
                "Value": s[
                    "prob_finish_below_start"
                ],
            },
        ]
    )


# ============================================================================
# SIMPLE SMOKE TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("QuantOS MONTE CARLO ENGINE")
    print("=" * 72)

    rng = np.random.default_rng(
        123
    )

    # Synthetic test data only.
    #
    # This is NOT a market result.
    # It exists purely to verify that the engine executes.

    synthetic_returns = pd.Series(
        rng.normal(
            0.0004,
            0.012,
            1500,
        )
    )

    print(
        f"Historical observations : "
        f"{len(synthetic_returns)}"
    )

    print(
        "\nRunning GARCH-t simulation..."
    )

    result = simulate_model(
        model="garch_t",
        returns=synthetic_returns,
        horizon=63,
        n_paths=2_000,
        seed=42,
    )

    print(
        "\nMODEL:"
    )

    print(
        result.model
    )

    print(
        "\nTERMINAL DISTRIBUTION:"
    )

    print(
        f"P05    : "
        f"{result.statistics['terminal_p05']:.2%}"
    )

    print(
        f"P25    : "
        f"{result.statistics['terminal_p25']:.2%}"
    )

    print(
        f"Median : "
        f"{result.statistics['terminal_median']:.2%}"
    )

    print(
        f"P75    : "
        f"{result.statistics['terminal_p75']:.2%}"
    )

    print(
        f"P95    : "
        f"{result.statistics['terminal_p95']:.2%}"
    )

    print(
        "\nTAIL RISK:"
    )

    print(
        f"VaR 95%  : "
        f"{result.statistics['VaR_95']:.2%}"
    )

    print(
        f"CVaR 95% : "
        f"{result.statistics['CVaR_95']:.2%}"
    )

    print(
        f"VaR 99%  : "
        f"{result.statistics['VaR_99']:.2%}"
    )

    print(
        f"CVaR 99% : "
        f"{result.statistics['CVaR_99']:.2%}"
    )

    print(
        "\nDRAWDOWN:"
    )

    print(
        f"Mean Max Drawdown : "
        f"{result.statistics['max_drawdown_mean']:.2%}"
    )

    print(
        f"P(Below Start)    : "
        f"{result.statistics['prob_finish_below_start']:.2%}"
    )

    print(
        "\nENGINE TEST COMPLETE."
    )