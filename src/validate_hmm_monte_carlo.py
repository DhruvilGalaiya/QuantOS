"""
QuantOS - HMM Monte Carlo Historical Validation

Purpose
-------
Quick historical validation of a regime-switching Monte Carlo model.

The model:
    1. Uses the historical causal HMM state available at each date.
    2. Estimates a transition matrix using only states known up to that date.
    3. Estimates return distributions separately for BULL / SIDE / BEAR.
    4. Simulates future regime paths.
    5. Samples returns from the corresponding historical regime.
    6. Compares the simulated 90% terminal range with the actual future return.

This is a research validation artifact.

It intentionally does NOT modify the production V10 HMM.
It also does not use today's transition matrix to predict historical periods.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PRICE_PATH = Path(
    "data/regime/daily/nifty_50.parquet"
)

HMM_PATH = Path(
    "data/regime/live_predictions/"
    "historical_hmm_states_nifty_50.csv"
)

OUTPUT_DIR = Path(
    "data/portfolio/monte_carlo_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = [
    "hmm_regime_switching",
]

HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
}

N_PATHS = 2000

MIN_TRAIN = 756

SEED = 42


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_price_data():

    if not PRICE_PATH.exists():

        raise FileNotFoundError(
            f"Missing NIFTY price data:\n"
            f"{PRICE_PATH}"
        )

    df = pd.read_parquet(
        PRICE_PATH
    ).copy()

    required = [
        "timestamp",
        "close",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing price columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df[
            [
                "timestamp",
                "close",
            ]
        ]
        .dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df["return"] = (
        df["close"]
        .pct_change()
    )

    df = df.dropna(
        subset=["return"]
    ).reset_index(drop=True)

    return df


# ============================================================
# LOAD HISTORICAL HMM STATES
# ============================================================

def load_hmm_states():

    if not HMM_PATH.exists():

        raise FileNotFoundError(
            f"Missing historical HMM states:\n"
            f"{HMM_PATH}"
        )

    df = pd.read_csv(
        HMM_PATH
    )

    required = [
        "timestamp",
        "current_regime",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing HMM columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df[
            [
                "timestamp",
                "current_regime",
            ]
        ]
        .dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    valid_states = {
        "BULL",
        "SIDE",
        "BEAR",
    }

    df = df[
        df["current_regime"].isin(
            valid_states
        )
    ].reset_index(drop=True)

    return df


# ============================================================
# COMBINE PRICE + HMM DATA
# ============================================================

def build_dataset():

    prices = load_price_data()

    states = load_hmm_states()

    df = prices.merge(
        states,
        on="timestamp",
        how="inner",
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TRANSITION MATRIX
# ============================================================

def estimate_transition_matrix(
    states
):
    """
    Estimate a causal empirical transition matrix.

    Only observations supplied to this function are used.
    """

    state_names = [
        "BULL",
        "SIDE",
        "BEAR",
    ]

    state_to_int = {
        "BULL": 0,
        "SIDE": 1,
        "BEAR": 2,
    }

    matrix = np.zeros(
        (
            3,
            3,
        ),
        dtype=float,
    )

    for previous, current in zip(
        states[:-1],
        states[1:],
    ):

        if (
            previous in state_to_int
            and current in state_to_int
        ):

            i = state_to_int[
                previous
            ]

            j = state_to_int[
                current
            ]

            matrix[i, j] += 1.0

    # Laplace smoothing prevents a state from having
    # impossible transitions simply because a transition
    # did not occur in a finite historical sample.
    matrix += 1.0

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    matrix = (
        matrix
        / row_sums
    )

    return matrix


# ============================================================
# REGIME RETURN POOLS
# ============================================================

def build_return_pools(
    historical_df
):
    """
    Build empirical return pools for each regime.

    Only historical returns available at the forecast origin
    should be passed here.
    """

    pools = {}

    for regime in [
        "BULL",
        "SIDE",
        "BEAR",
    ]:

        values = (
            historical_df.loc[
                historical_df[
                    "current_regime"
                ] == regime,
                "return",
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        pools[regime] = values

    return pools


# ============================================================
# SIMULATE REGIME-SWITCHING PATHS
# ============================================================

def simulate_hmm_paths(
    historical_df,
    horizon,
    n_paths,
    seed,
):
    """
    Simulate future paths using:

        current regime
              ↓
        transition matrix
              ↓
        next regime
              ↓
        empirical return distribution
              ↓
        repeat
    """

    rng = np.random.default_rng(
        seed
    )

    state_names = [
        "BULL",
        "SIDE",
        "BEAR",
    ]

    state_to_int = {
        "BULL": 0,
        "SIDE": 1,
        "BEAR": 2,
    }

    int_to_state = {
        0: "BULL",
        1: "SIDE",
        2: "BEAR",
    }

    states = (
        historical_df[
            "current_regime"
        ]
        .to_numpy()
    )

    transition_matrix = (
        estimate_transition_matrix(
            states
        )
    )

    return_pools = (
        build_return_pools(
            historical_df
        )
    )

    # Make sure every state has enough data.
    for regime in state_names:

        if len(
            return_pools[regime]
        ) < 20:

            raise ValueError(
                f"Not enough historical "
                f"{regime} returns for "
                f"simulation."
            )

    current_state_name = (
        states[-1]
    )

    current_state = state_to_int[
        current_state_name
    ]

    wealth = np.ones(
        (
            n_paths,
            horizon + 1,
        ),
        dtype=float,
    )

    state_paths = np.zeros(
        (
            n_paths,
            horizon,
        ),
        dtype=np.int8,
    )

    for path_index in range(
        n_paths
    ):

        state = current_state

        for day in range(
            horizon
        ):

            probabilities = (
                transition_matrix[
                    state
                ]
            )

            next_state = rng.choice(
                3,
                p=probabilities,
            )

            regime = int_to_state[
                next_state
            ]

            pool = return_pools[
                regime
            ]

            sampled_return = rng.choice(
                pool
            )

            # Prevent impossible total loss.
            sampled_return = max(
                sampled_return,
                -0.999,
            )

            wealth[
                path_index,
                day + 1
            ] = (
                wealth[
                    path_index,
                    day
                ]
                * (
                    1.0
                    + sampled_return
                )
            )

            state_paths[
                path_index,
                day
            ] = next_state

            state = next_state

    metadata = {
        "transition_matrix":
            transition_matrix,

        "current_regime":
            current_state_name,

        "state_paths":
            state_paths,
    }

    return wealth, metadata


# ============================================================
# SINGLE HISTORICAL FORECAST
# ============================================================

def run_forecast(
    historical_df,
    horizon,
    n_paths,
    seed,
):

    paths, metadata = (
        simulate_hmm_paths(
            historical_df=historical_df,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
        )
    )

    terminal_returns = (
        paths[:, -1] - 1.0
    )

    lower = float(
        np.percentile(
            terminal_returns,
            5,
        )
    )

    median = float(
        np.percentile(
            terminal_returns,
            50,
        )
    )

    upper = float(
        np.percentile(
            terminal_returns,
            95,
        )
    )

    return {
        "lower_5": lower,
        "median": median,
        "upper_95": upper,
        "metadata": metadata,
    }


# ============================================================
# WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    successes,
    total,
):
    """
    95% Wilson interval for a binomial proportion.
    """

    if total == 0:

        return (
            np.nan,
            np.nan,
        )

    z = 1.959963984540054

    p = (
        successes
        / total
    )

    denominator = (
        1.0
        + z * z / total
    )

    center = (
        p
        + z * z / (
            2.0 * total
        )
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (
                    1.0 - p
                )
                / total
            )
            + (
                z * z
                / (
                    4.0
                    * total
                    * total
                )
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center - margin,
        ),
        min(
            1.0,
            center + margin,
        ),
    )


# ============================================================
# VALIDATE ONE HORIZON
# ============================================================

def validate_horizon(
    df,
    horizon_name,
    horizon,
):

    results = []

    # We need enough HMM history plus a future horizon.
    if len(df) <= (
        MIN_TRAIN
        + horizon
    ):

        raise ValueError(
            f"Not enough observations "
            f"for {horizon_name}."
        )

    origin = MIN_TRAIN

    forecast_number = 0

    while (
        origin + horizon
        <= len(df)
    ):

        historical = (
            df.iloc[:origin]
            .copy()
        )

        future = (
            df.iloc[
                origin:
                origin + horizon
            ]
            .copy()
        )

        # Actual future return.
        actual_return = (
            future["close"].iloc[-1]
            / historical["close"].iloc[-1]
            - 1.0
        )

        forecast_date = (
            historical["timestamp"].iloc[-1]
        )

        actual_end_date = (
            future["timestamp"].iloc[-1]
        )

        try:

            forecast = run_forecast(
                historical_df=historical,
                horizon=horizon,
                n_paths=N_PATHS,
                seed=(
                    SEED
                    + forecast_number
                ),
            )

        except Exception as exc:

            print(
                f"WARNING: HMM forecast failed "
                f"at {forecast_date}: {exc}"
            )

            origin += horizon

            forecast_number += 1

            continue

        lower = forecast[
            "lower_5"
        ]

        median = forecast[
            "median"
        ]

        upper = forecast[
            "upper_95"
        ]

        inside = (
            lower
            <= actual_return
            <= upper
        )

        lower_violation = (
            actual_return
            < lower
        )

        upper_violation = (
            actual_return
            > upper
        )

        results.append(
            {
                "model":
                    "HMM Regime Switching",

                "horizon":
                    horizon_name,

                "forecast_date":
                    forecast_date,

                "actual_end_date":
                    actual_end_date,

                "actual_return":
                    actual_return,

                "lower_5":
                    lower,

                "median":
                    median,

                "upper_95":
                    upper,

                "inside_90pct_band":
                    inside,

                "lower_violation":
                    lower_violation,

                "upper_violation":
                    upper_violation,

                "band_width":
                    upper - lower,

                "starting_regime":
                    forecast[
                        "metadata"
                    ][
                        "current_regime"
                    ],
            }
        )

        origin += horizon

        forecast_number += 1

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results
):

    summaries = []

    for (
        model,
        horizon,
    ), group in results.groupby(
        [
            "model",
            "horizon",
        ]
    ):

        total = len(group)

        inside = int(
            group[
                "inside_90pct_band"
            ].sum()
        )

        coverage = (
            inside
            / total
        )

        ci_low, ci_high = (
            wilson_interval(
                inside,
                total,
            )
        )

        summaries.append(
            {
                "model":
                    model,

                "horizon":
                    horizon,

                "forecasts":
                    total,

                "inside_band":
                    f"{inside}/{total}",

                "target_coverage":
                    0.90,

                "observed_coverage":
                    coverage,

                "coverage_error":
                    coverage - 0.90,

                "coverage_ci_low":
                    ci_low,

                "coverage_ci_high":
                    ci_high,

                "lower_violation_rate":
                    group[
                        "lower_violation"
                    ].mean(),

                "upper_violation_rate":
                    group[
                        "upper_violation"
                    ].mean(),

                "average_band_width":
                    group[
                        "band_width"
                    ].mean(),

                "median_band_width":
                    group[
                        "band_width"
                    ].median(),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "QUANTOS HMM MONTE CARLO VALIDATION"
    )
    print("=" * 78)

    print()

    print(
        f"Price data : {PRICE_PATH}"
    )

    print(
        f"HMM data   : {HMM_PATH}"
    )

    print(
        f"Simulation paths : {N_PATHS}"
    )

    print(
        f"Minimum training : {MIN_TRAIN}"
    )

    print()

    df = build_dataset()

    print(
        "MARKET + HMM DATA"
    )

    print("-" * 78)

    print(
        f"Observations : {len(df):,}"
    )

    print(
        f"Period       : "
        f"{df['timestamp'].iloc[0].date()} "
        f"→ "
        f"{df['timestamp'].iloc[-1].date()}"
    )

    print()

    print(
        "HISTORICAL REGIME COUNTS"
    )

    print("-" * 78)

    counts = (
        df[
            "current_regime"
        ]
        .value_counts()
        .reindex(
            [
                "BULL",
                "SIDE",
                "BEAR",
            ],
            fill_value=0,
        )
    )

    for regime, count in (
        counts.items()
    ):

        print(
            f"{regime:>5} : "
            f"{count:>5}"
        )

    all_results = []

    for (
        horizon_name,
        horizon,
    ) in HORIZONS.items():

        print()
        print("=" * 78)

        print(
            f"VALIDATING {horizon_name} "
            f"({horizon} trading days)"
        )

        print("=" * 78)

        result = validate_horizon(
            df=df,
            horizon_name=horizon_name,
            horizon=horizon,
        )

        if result.empty:

            print(
                "No successful forecasts."
            )

            continue

        all_results.append(
            result
        )

        inside = int(
            result[
                "inside_90pct_band"
            ].sum()
        )

        total = len(
            result
        )

        coverage = (
            inside
            / total
            * 100
        )

        print(
            f"Forecasts : {total}"
        )

        print(
            f"Inside 90% range : "
            f"{inside}/{total}"
        )

        print(
            f"Observed coverage : "
            f"{coverage:.2f}%"
        )

        print(
            f"Average range width : "
            f"{result['band_width'].mean() * 100:.2f}%"
        )

    if not all_results:

        raise RuntimeError(
            "No successful HMM "
            "validation forecasts."
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = summarize(
        results
    )

    # --------------------------------------------------------
    # Save detailed results.
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "hmm_monte_carlo_validation_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "hmm_monte_carlo_validation_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Display summary.
    # --------------------------------------------------------

    display = summary.copy()

    percentage_columns = [
        "target_coverage",
        "observed_coverage",
        "coverage_error",
        "coverage_ci_low",
        "coverage_ci_high",
        "lower_violation_rate",
        "upper_violation_rate",
        "average_band_width",
        "median_band_width",
    ]

    for column in percentage_columns:

        display[column] = (
            display[column]
            * 100
        )

    display = display.round(
        {
            column: 2
            for column in percentage_columns
        }
    )

    print()
    print("=" * 78)
    print(
        "HMM MONTE CARLO VALIDATION SUMMARY"
    )
    print("=" * 78)

    print()

    print(
        display.to_string(
            index=False
        )
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "- Coverage near 90% means the simulated "
        "90% range has approximately the intended "
        "historical coverage."
    )

    print(
        "- The coverage interval shows the uncertainty "
        "around the measured historical coverage."
    )

    print(
        "- Longer horizons have fewer validation "
        "forecasts, so their coverage estimates are "
        "less precise."
    )

    print(
        "- This HMM model is a research comparison, "
        "not a replacement for the production V10 HMM."
    )

    print()

    print(
        "FILES SAVED"
    )

    print("-" * 78)

    print(
        f"Detailed results : "
        f"{results_path}"
    )

    print(
        f"Summary          : "
        f"{summary_path}"
    )

    print()

    print(
        "HMM MONTE CARLO VALIDATION COMPLETE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()