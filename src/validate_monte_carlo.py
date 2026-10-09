"""
QuantOS Monte Carlo Historical Validation

Purpose
-------
Tests whether Monte Carlo forecast intervals have reasonable historical
coverage.

The validation is rolling-origin:

    past data
        ↓
    fit model
        ↓
    simulate future
        ↓
    compare simulated range with actual future
        ↓
    move forward
        ↓
    repeat

Important:
- No future data is used when fitting each forecast.
- Validation is performed on NIFTY 50.
- HMM is intentionally excluded from this first validation pass.
- Coverage confidence intervals are reported so small samples are not
  presented with false precision.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quantos_monte_carlo import simulate_model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/regime/daily/nifty_50.parquet")

OUTPUT_DIR = Path(
    "data/portfolio/monte_carlo_validation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "student_t",
    "garch_t",
    "garch_t_jump",
    "bootstrap",
]

HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
}

# 2,000 paths is sufficient for historical calibration.
N_PATHS = 2000

# Approximately 3 years of trading observations.
MIN_TRAIN = 756

# Use non-overlapping forecast windows.
STEP_BY_HORIZON = True

SEED = 42


# ============================================================
# DATA
# ============================================================

def load_nifty_returns():
    """Load NIFTY 50 and create clean daily simple returns."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"NIFTY data not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH).copy()

    if "timestamp" not in df.columns:
        raise ValueError(
            "Expected 'timestamp' column."
        )

    if "close" not in df.columns:
        raise ValueError(
            "Expected 'close' column."
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

    df["return"] = df["close"].pct_change()

    df = df.dropna(
        subset=["return"]
    ).reset_index(drop=True)

    return df


# ============================================================
# SINGLE FORECAST
# ============================================================

def run_single_forecast(
    returns,
    model,
    horizon,
    n_paths,
    seed,
):
    """
    Generate a Monte Carlo forecast.

    simulate_model() returns a SimulationResult object.

    Its paths are wealth/value paths beginning at approximately
    1.0, so terminal wealth must be converted into a terminal
    return before comparing it with the actual historical return.
    """

    result = simulate_model(
        returns=returns,
        model=model,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
    )

    paths = result.paths

    if paths is None:
        raise ValueError(
            f"Model '{model}' returned no simulation paths."
        )

    paths = np.asarray(paths)

    if paths.ndim != 2:
        raise ValueError(
            f"Unexpected path shape for {model}: "
            f"{paths.shape}"
        )

    # Convert terminal wealth/value to terminal return.
    terminal_returns = paths[:, -1] - 1.0

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
    }


# ============================================================
# ROLLING VALIDATION
# ============================================================

def validate_model(
    df,
    model,
    horizon_name,
    horizon,
    n_paths=N_PATHS,
):
    """
    Perform rolling-origin validation for one model/horizon.
    """

    results = []

    n = len(df)

    if n <= MIN_TRAIN + horizon:
        raise ValueError(
            f"Not enough observations for "
            f"{model} / {horizon_name}."
        )

    origin = MIN_TRAIN
    forecast_number = 0

    while origin + horizon <= n:

        train = df.iloc[:origin].copy()

        future = df.iloc[
            origin:origin + horizon
        ].copy()

        train_returns = (
            train["return"]
            .to_numpy(dtype=float)
        )

        actual_return = (
            future["close"].iloc[-1]
            / train["close"].iloc[-1]
            - 1.0
        )

        forecast_date = (
            train["timestamp"].iloc[-1]
        )

        actual_end_date = (
            future["timestamp"].iloc[-1]
        )

        try:

            forecast = run_single_forecast(
                returns=train_returns,
                model=model,
                horizon=horizon,
                n_paths=n_paths,
                seed=SEED + forecast_number,
            )

        except Exception as exc:

            print(
                f"WARNING: forecast failed "
                f"{model} {horizon_name} "
                f"at {forecast_date}: {exc}"
            )

            if STEP_BY_HORIZON:
                origin += horizon
            else:
                origin += 1

            forecast_number += 1

            continue

        lower = forecast["lower_5"]
        upper = forecast["upper_95"]

        inside_band = (
            actual_return >= lower
            and actual_return <= upper
        )

        lower_violation = (
            actual_return < lower
        )

        upper_violation = (
            actual_return > upper
        )

        results.append(
            {
                "model": model,
                "horizon": horizon_name,
                "forecast_date": forecast_date,
                "actual_end_date": actual_end_date,
                "actual_return": actual_return,
                "lower_5": lower,
                "median": forecast["median"],
                "upper_95": upper,
                "inside_90pct_band": inside_band,
                "lower_violation": lower_violation,
                "upper_violation": upper_violation,
                "band_width": upper - lower,
            }
        )

        forecast_number += 1

        if STEP_BY_HORIZON:
            origin += horizon
        else:
            origin += 1

    return pd.DataFrame(results)


# ============================================================
# BINOMIAL CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    successes,
    total,
    confidence=0.95,
):
    """
    Wilson confidence interval for a binomial proportion.

    Used instead of a normal approximation because some of the
    validation samples, particularly 6M, are small.
    """

    if total <= 0:
        return np.nan, np.nan

    z_table = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }

    z = z_table.get(
        confidence,
        1.959963984540054,
    )

    p = successes / total

    denominator = (
        1.0
        + (z ** 2) / total
    )

    center = (
        p
        + (z ** 2) / (2.0 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1.0 - p) / total
                + (z ** 2) / (4.0 * total ** 2)
            )
        )
        / denominator
    )

    lower = max(
        0.0,
        center - margin,
    )

    upper = min(
        1.0,
        center + margin,
    )

    return lower, upper


# ============================================================
# SUMMARY
# ============================================================

def summarize_validation(results):
    """Create model-level validation statistics."""

    if results.empty:
        return pd.DataFrame()

    summaries = []

    grouped = results.groupby(
        [
            "model",
            "horizon",
        ],
        sort=False,
    )

    for (
        model,
        horizon,
    ), group in grouped:

        total = len(group)

        successes = int(
            group[
                "inside_90pct_band"
            ].sum()
        )

        coverage = (
            successes / total
        )

        coverage_low, coverage_high = (
            wilson_interval(
                successes,
                total,
                confidence=0.95,
            )
        )

        lower_rate = (
            group[
                "lower_violation"
            ].mean()
        )

        upper_rate = (
            group[
                "upper_violation"
            ].mean()
        )

        average_width = (
            group[
                "band_width"
            ].mean()
        )

        median_width = (
            group[
                "band_width"
            ].median()
        )

        summaries.append(
            {
                "model": model,
                "horizon": horizon,

                "forecasts": total,

                "inside_band": (
                    f"{successes}/{total}"
                ),

                "target_coverage": 0.90,

                "observed_coverage": coverage,

                "coverage_error": (
                    coverage - 0.90
                ),

                "coverage_ci_low": (
                    coverage_low
                ),

                "coverage_ci_high": (
                    coverage_high
                ),

                "lower_violation_rate": (
                    lower_rate
                ),

                "upper_violation_rate": (
                    upper_rate
                ),

                "average_band_width": (
                    average_width
                ),

                "median_band_width": (
                    median_width
                ),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_summary(summary):

    print()
    print("=" * 78)
    print(
        "QUANTOS MONTE CARLO HISTORICAL VALIDATION"
    )
    print("=" * 78)

    print()

    print(
        "Question being tested:"
    )

    print(
        "When the model says that roughly 90% "
        "of outcomes should fall inside its "
        "forecast range, did that happen historically?"
    )

    print()

    print(
        "Target coverage: 90%"
    )

    print(
        "Coverage interval: 95% Wilson confidence interval"
    )

    print()

    if summary.empty:

        print(
            "No validation results."
        )

        return

    display = summary.copy()

    # Convert proportions to percentages.
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
            display[column] * 100
        )

    display = display.round(
        {
            "target_coverage": 2,
            "observed_coverage": 2,
            "coverage_error": 2,
            "coverage_ci_low": 2,
            "coverage_ci_high": 2,
            "lower_violation_rate": 2,
            "upper_violation_rate": 2,
            "average_band_width": 2,
            "median_band_width": 2,
        }
    )

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
        "- 'Inside' shows how many historical outcomes "
        "fell inside the simulated 90% range."
    )

    print(
        "- 'Observed coverage' is the percentage of "
        "historical outcomes inside that range."
    )

    print(
        "- The 95% coverage interval shows the uncertainty "
        "around the observed coverage."
    )

    print(
        "- With only 15 six-month forecasts, the interval "
        "will naturally be wide."
    )

    print(
        "- Coverage near 90% is consistent with the intended "
        "90% forecast range."
    )

    print(
        "- Coverage materially below 90% suggests the range "
        "may be too narrow."
    )

    print(
        "- Coverage materially above 90% suggests the range "
        "may be unnecessarily wide."
    )

    print(
        "- Band width measures how much uncertainty the model "
        "is putting around the outcome."
    )

    print(
        "- Lower and upper violation rates show whether "
        "errors are concentrated on either side."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "QUANTOS MONTE CARLO VALIDATION"
    )
    print("=" * 78)

    print()

    print(
        f"Data: {DATA_PATH}"
    )

    print(
        f"Simulation paths per forecast: "
        f"{N_PATHS}"
    )

    print(
        f"Minimum training observations: "
        f"{MIN_TRAIN}"
    )

    df = load_nifty_returns()

    print()
    print(
        "MARKET DATA"
    )
    print("-" * 78)

    print(
        f"Observations : "
        f"{len(df):,}"
    )

    print(
        f"Period       : "
        f"{df['timestamp'].iloc[0].date()} "
        f"→ "
        f"{df['timestamp'].iloc[-1].date()}"
    )

    print(
        f"Latest close : "
        f"₹{df['close'].iloc[-1]:,.2f}"
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

        for model in MODELS:

            print()

            print(
                f"Running {model}..."
            )

            result = validate_model(
                df=df,
                model=model,
                horizon_name=horizon_name,
                horizon=horizon,
                n_paths=N_PATHS,
            )

            if result.empty:

                print(
                    "No successful forecasts."
                )

                continue

            all_results.append(
                result
            )

            coverage = (
                result[
                    "inside_90pct_band"
                ].mean()
                * 100
            )

            inside = int(
                result[
                    "inside_90pct_band"
                ].sum()
            )

            total = len(result)

            print(
                f"Forecasts : {total}"
            )

            print(
                f"Inside 90% range : "
                f"{inside}/{total}"
            )

            print(
                f"Observed 90% coverage : "
                f"{coverage:.2f}%"
            )

            print(
                f"Average range width : "
                f"{result['band_width'].mean() * 100:.2f}%"
            )

    if not all_results:

        raise RuntimeError(
            "No validation forecasts were generated."
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = summarize_validation(
        results
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_path = (
        OUTPUT_DIR
        / "monte_carlo_validation_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "monte_carlo_validation_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_summary(
        summary
    )

    print()
    print("=" * 78)
    print(
        "FILES SAVED"
    )
    print("=" * 78)

    print()

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
        "VALIDATION COMPLETE."
    )


if __name__ == "__main__":
    main()