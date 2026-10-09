"""
QuantOS Real NIFTY-50 Monte Carlo Test

This script connects the Monte Carlo engine to the actual
QuantOS NIFTY-50 historical dataset.

It is a research test only.
It does not modify the dashboard or any model artefacts.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from quantos_monte_carlo import (
    simulate_model,
    summarize_simulation,
    fan_chart_dataframe,
)


# ============================================================================
# CONFIG
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRICE_FILE = (
    ROOT
    / "data"
    / "regime"
    / "daily"
    / "nifty_50.parquet"
)

HMM_FILE = (
    ROOT
    / "data"
    / "regime"
    / "live_predictions"
    / "nifty_50.json"
)

N_PATHS = 10_000

HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

SEED = 42


# ============================================================================
# LOAD NIFTY DATA
# ============================================================================

def load_nifty():

    df = pd.read_parquet(
        PRICE_FILE
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
        .drop_duplicates(
            subset="timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    df["return"] = (
        df["close"]
        .pct_change()
    )

    df = df.dropna(
        subset=["return"]
    )

    return df


# ============================================================================
# LOAD CURRENT HMM STATE
# ============================================================================

def load_hmm():

    with open(
        HMM_FILE,
        "r",
    ) as f:

        data = json.load(f)

    probabilities = data[
        "probabilities"
    ]

    transition = np.asarray(
        data[
            "transition_matrix"
        ][
            "data"
        ],
        dtype=float,
    )

    regime_map = {
        "BULL": 0,
        "SIDE": 1,
        "BEAR": 2,
    }

    current_probabilities = np.array(
        [
            probabilities["BULL"],
            probabilities["SIDE"],
            probabilities["BEAR"],
        ],
        dtype=float,
    )

    return (
        data,
        regime_map,
        transition,
        current_probabilities,
    )


# ============================================================================
# BUILD HISTORICAL HMM LABELS
# ============================================================================

def load_historical_hmm_labels(
    nifty_df,
    regime_map,
    hmm_data,
):

    daily_output = pd.DataFrame(
        hmm_data["daily_output"]
    )

    if daily_output.empty:

        raise ValueError(
            "HMM daily_output is empty."
        )

    daily_output["date"] = pd.to_datetime(
        daily_output["date"]
    )

    daily_output[
        "regime_id"
    ] = daily_output[
        "current_regime"
    ].map(
        regime_map
    )

    # The current live HMM JSON stores the recent daily history.
    #
    # Align it with the real NIFTY returns.
    labels = daily_output[
        [
            "date",
            "regime_id",
        ]
    ].dropna()

    labels = labels.rename(
        columns={
            "date": "timestamp"
        }
    )

    merged = nifty_df.merge(
        labels,
        on="timestamp",
        how="inner",
    )

    if len(merged) < 100:

        raise ValueError(
            "Too few aligned HMM observations."
        )

    return merged


# ============================================================================
# PRINT HEADER
# ============================================================================

print("=" * 80)
print("QUANTOS REAL NIFTY-50 MONTE CARLO TEST")
print("=" * 80)

print(
    f"\nPrice file : {PRICE_FILE}"
)

# ============================================================================
# LOAD DATA
# ============================================================================

nifty = load_nifty()

latest_date = nifty[
    "timestamp"
].iloc[-1]

latest_close = float(
    nifty[
        "close"
    ].iloc[-1]
)

returns = nifty[
    "return"
]

print(
    "\nREAL MARKET DATA"
)

print(
    f"Observations : {len(nifty):,}"
)

print(
    f"Period       : "
    f"{nifty['timestamp'].iloc[0].date()} "
    f"→ "
    f"{latest_date.date()}"
)

print(
    f"Latest Close : ₹{latest_close:,.2f}"
)

# ============================================================================
# BASIC HISTORICAL VOLATILITY
# ============================================================================

rv_21 = (
    returns.tail(21)
    .std(ddof=1)
    * np.sqrt(252)
)

rv_63 = (
    returns.tail(63)
    .std(ddof=1)
    * np.sqrt(252)
)

rv_252 = (
    returns.tail(252)
    .std(ddof=1)
    * np.sqrt(252)
)

print(
    "\nREALIZED VOLATILITY"
)

print(
    f"21D  : {rv_21:.2%}"
)

print(
    f"63D  : {rv_63:.2%}"
)

print(
    f"252D : {rv_252:.2%}"
)

# ============================================================================
# HMM
# ============================================================================

(
    hmm_data,
    regime_map,
    transition_matrix,
    current_probabilities,
) = load_hmm()

print(
    "\nCURRENT HMM STATE"
)

print(
    f"Latest HMM date : "
    f"{hmm_data['latest_date']}"
)

print(
    f"Current regime  : "
    f"{hmm_data['current_regime']}"
)

print(
    f"P(BULL)         : "
    f"{current_probabilities[0]:.2%}"
)

print(
    f"P(SIDE)         : "
    f"{current_probabilities[1]:.2%}"
)

print(
    f"P(BEAR)         : "
    f"{current_probabilities[2]:.2%}"
)

print(
    "\nHMM TRANSITION MATRIX"
)

transition_df = pd.DataFrame(
    transition_matrix,
    index=[
        "BULL",
        "SIDE",
        "BEAR",
    ],
    columns=[
        "BULL",
        "SIDE",
        "BEAR",
    ],
)

print(
    transition_df.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

# ============================================================================
# HISTORICAL HMM LABELS
# ============================================================================

hmm_aligned = load_historical_hmm_labels(
    nifty,
    regime_map,
    hmm_data,
)

print(
    "\nHMM TRAINING SAMPLE FOR SIMULATION"
)

print(
    f"Aligned observations : "
    f"{len(hmm_aligned):,}"
)

print(
    f"From : "
    f"{hmm_aligned['timestamp'].iloc[0].date()}"
)

print(
    f"To   : "
    f"{hmm_aligned['timestamp'].iloc[-1].date()}"
)

# ============================================================================
# MODEL RUNNER
# ============================================================================

def run_model(
    model,
    horizon,
    seed_offset=0,
):

    kwargs = {
        "model": model,
        "returns": returns,
        "horizon": horizon,
        "n_paths": N_PATHS,
        "seed": SEED + seed_offset,
    }

    if model == "hmm":

        # Use the HMM-aligned historical sample.
        hmm_returns = hmm_aligned[
            "return"
        ]

        kwargs["returns"] = (
            hmm_returns
        )

        kwargs[
            "regime_labels"
        ] = hmm_aligned[
            "regime_id"
        ].to_numpy()

        kwargs[
            "transition_matrix"
        ] = transition_matrix

        kwargs[
            "current_probabilities"
        ] = current_probabilities

    return simulate_model(
        **kwargs
    )


# ============================================================================
# RUN MODELS
# ============================================================================

MODELS = [
    "student_t",
    "garch_t",
    "garch_t_jump",
    "bootstrap",
    "hmm",
]

all_results = {}


for horizon_name, horizon_days in HORIZONS.items():

    print("\n")
    print("=" * 80)
    print(
        f"{horizon_name} FORWARD SIMULATION "
        f"({horizon_days} trading days)"
    )
    print("=" * 80)

    rows = []

    for model_number, model in enumerate(
        MODELS
    ):

        print(
            f"\nRunning {model}..."
        )

        try:

            result = run_model(
                model=model,
                horizon=horizon_days,
                seed_offset=(
                    model_number
                    + horizon_days
                ),
            )

            stats = result.statistics

            rows.append(
                {
                    "Model": model,
                    "P05": stats[
                        "terminal_p05"
                    ],
                    "P25": stats[
                        "terminal_p25"
                    ],
                    "Median": stats[
                        "terminal_median"
                    ],
                    "P75": stats[
                        "terminal_p75"
                    ],
                    "P95": stats[
                        "terminal_p95"
                    ],
                    "VaR95": stats[
                        "VaR_95"
                    ],
                    "CVaR95": stats[
                        "CVaR_95"
                    ],
                    "VaR99": stats[
                        "VaR_99"
                    ],
                    "CVaR99": stats[
                        "CVaR_99"
                    ],
                    "MeanMaxDD": stats[
                        "max_drawdown_mean"
                    ],
                    "P_Below_Start": stats[
                        "prob_finish_below_start"
                    ],
                }
            )

            # Save first model's result for later inspection.
            all_results[
                (
                    horizon_name,
                    model,
                )
            ] = result

        except Exception as e:

            print(
                f"ERROR: {type(e).__name__}: {e}"
            )

    comparison = pd.DataFrame(
        rows
    )

    if not comparison.empty:

        display_df = comparison.copy()

        percentage_columns = [
            "P05",
            "P25",
            "Median",
            "P75",
            "P95",
            "VaR95",
            "CVaR95",
            "VaR99",
            "CVaR99",
            "MeanMaxDD",
            "P_Below_Start",
        ]

        for column in percentage_columns:

            display_df[column] = (
                display_df[column]
                .map(
                    lambda x:
                    f"{x:.2%}"
                )
            )

        print(
            "\nMODEL COMPARISON"
        )

        print(
            display_df.to_string(
                index=False
            )
        )

# ============================================================================
# REAL NIFTY GARCH-JUMP DETAIL
# ============================================================================

print("\n")
print("=" * 80)
print("PRIMARY SINGLE-ASSET MODEL: GARCH-T + JUMPS")
print("=" * 80)

primary = all_results.get(
    ("6M", "garch_t_jump")
)

if primary is not None:

    stats = primary.statistics

    print(
        "\n6-MONTH TERMINAL DISTRIBUTION"
    )

    print(
        f"5th percentile  : "
        f"{stats['terminal_p05']:.2%}"
    )

    print(
        f"25th percentile : "
        f"{stats['terminal_p25']:.2%}"
    )

    print(
        f"Median          : "
        f"{stats['terminal_median']:.2%}"
    )

    print(
        f"75th percentile : "
        f"{stats['terminal_p75']:.2%}"
    )

    print(
        f"95th percentile : "
        f"{stats['terminal_p95']:.2%}"
    )

    print(
        "\nTAIL RISK"
    )

    print(
        f"VaR 95%  : "
        f"{stats['VaR_95']:.2%}"
    )

    print(
        f"CVaR 95% : "
        f"{stats['CVaR_95']:.2%}"
    )

    print(
        f"VaR 99%  : "
        f"{stats['VaR_99']:.2%}"
    )

    print(
        f"CVaR 99% : "
        f"{stats['CVaR_99']:.2%}"
    )

    print(
        "\nDRAWDOWN"
    )

    print(
        f"Mean maximum drawdown : "
        f"{stats['max_drawdown_mean']:.2%}"
    )

    print(
        f"95th percentile DD    : "
        f"{stats['max_drawdown_p05']:.2%}"
    )

    print(
        "\nDOWNSIDE"
    )

    print(
        f"Probability below start : "
        f"{stats['prob_finish_below_start']:.2%}"
    )

# ============================================================================
# FAN CHART DATA
# ============================================================================

if primary is not None:

    fan = fan_chart_dataframe(
        primary.paths
    )

    output_file = (
        ROOT
        / "data"
        / "portfolio"
        / "monte_carlo_nifty_6m_fan.csv"
    )

    fan.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nFan chart data saved:"
    )

    print(
        output_file
    )

# ============================================================================
# COMPLETE
# ============================================================================

print("\n")
print("=" * 80)
print("REAL NIFTY MONTE CARLO TEST COMPLETE")
print("=" * 80)