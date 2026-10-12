from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import quantos_monte_carlo as mc


# ============================================================
# PATHS
# ============================================================

PRICE_PATH = ROOT / "data/regime/daily/nifty_50.parquet"

INPUT_PATH = (
    ROOT
    / "data/portfolio/monte_carlo_inputs/"
    / "nifty_50_monte_carlo_inputs.json"
)

HMM_PATH = (
    ROOT
    / "data/regime/live_predictions/"
    / "nifty_50.json"
)

OUTPUT_DIR = ROOT / "data/portfolio/monte_carlo_outputs"

OUTPUT_JSON = (
    OUTPUT_DIR
    / "nifty_50_monte_carlo_output.json"
)


# ============================================================
# CONFIG
# ============================================================

N_PATHS = 10_000
SEED = 42

PATH_SAMPLE_COUNT = 60

HORIZONS = {
    "1M": "1M",
    "3M": "3M",
    "6M": "6M",
    "1Y": "1Y",
}

MODELS = [
    "student_t",
    "garch_t",
    "garch_jump",
    "bootstrap",
    "hmm",
]

STATE_NAMES = [
    "BULL",
    "SIDE",
    "BEAR",
]

# ============================================================
# PATH VISUALIZATION
# ============================================================

def build_path_visualization(
    paths: np.ndarray,
    initial_value: float,
) -> dict:
    """
    Prepare a compact representation of Monte Carlo paths
    for frontend visualization.

    The simulation itself still uses all N_PATHS paths.
    Only representative paths and percentile trajectories
    are stored in the output artifact.
    """

    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2 or paths.shape[0] == 0:
        raise ValueError(
            "Monte Carlo paths must be a non-empty 2D array."
        )

    # Normalized paths start at 1.0.
    # Convert them back to actual NIFTY price levels.
    price_paths = paths * float(initial_value)

    n_steps = price_paths.shape[1]

    days = list(range(n_steps))

    # --------------------------------------------------------
    # Representative paths
    #
    # Select paths according to terminal outcome rank rather
    # than simply taking the first 60 random paths. This gives
    # the chart a useful spread across the simulated outcomes.
    # --------------------------------------------------------

    terminal_values = price_paths[:, -1]

    order = np.argsort(terminal_values)

    sample_count = min(
        PATH_SAMPLE_COUNT,
        len(order),
    )

    sample_positions = np.linspace(
        0,
        len(order) - 1,
        sample_count,
        dtype=int,
    )

    selected_indices = order[
        sample_positions
    ]

    representative_paths = (
        price_paths[selected_indices]
        .tolist()
    )

    # --------------------------------------------------------
    # Percentile trajectories
    #
    # These are calculated independently at every time step.
    # They form the probability fan on the frontend.
    # --------------------------------------------------------

    percentile_paths = {}

    for percentile, key in [
        (5, "p05"),
        (25, "p25"),
        (50, "median"),
        (75, "p75"),
        (95, "p95"),
    ]:
        percentile_paths[key] = (
            np.percentile(
                price_paths,
                percentile,
                axis=0,
            )
            .tolist()
        )

    return {
        "days": days,
        "paths": representative_paths,
        "percentile_paths": percentile_paths,
        "path_count": int(sample_count),
        "total_simulation_paths": int(paths.shape[0]),
    }

# ============================================================
# LOADERS
# ============================================================

def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_PATH).copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return df


def load_inputs() -> dict:
    with open(INPUT_PATH, "r") as f:
        return json.load(f)


def load_hmm() -> dict:
    if not HMM_PATH.exists():
        raise FileNotFoundError(
            f"Fresh HMM output not found: {HMM_PATH}"
        )

    with open(HMM_PATH, "r") as f:
        return json.load(f)


# ============================================================
# RETURNS
# ============================================================

def get_returns(prices: pd.DataFrame) -> pd.Series:
    returns = prices["close"].pct_change()

    returns = (
        returns
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    return returns


# ============================================================
# HMM EXTRACTION
# ============================================================

def _normalize_probabilities(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = values.sum()

    if total <= 0:
        raise ValueError(
            "HMM probabilities contain no positive mass."
        )

    return values / total


def _extract_probability_dict(
    obj: dict,
) -> np.ndarray | None:

    if not isinstance(obj, dict):
        return None

    values = []

    for state in STATE_NAMES:

        value = None

        if state in obj:
            value = obj[state]

        elif state.lower() in obj:
            value = obj[state.lower()]

        if value is None:
            return None

        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None

    return _normalize_probabilities(
        np.asarray(values, dtype=float)
    )


def get_current_probabilities(
    hmm: dict,
) -> np.ndarray:

    # --------------------------------------------------------
    # 1. Direct probabilities
    # --------------------------------------------------------

    candidates = [
        hmm.get("probabilities"),
        hmm.get("next_day_probabilities"),
        hmm.get("next_probabilities"),
    ]

    for candidate in candidates:

        if isinstance(candidate, dict):

            result = _extract_probability_dict(
                candidate
            )

            if result is not None:
                return result


    # --------------------------------------------------------
    # 2. Nested production/summary/model structures
    # --------------------------------------------------------

    for parent_key in [
        "summary",
        "production_output",
        "hmm",
        "model",
    ]:

        parent = hmm.get(parent_key)

        if not isinstance(parent, dict):
            continue

        for key in [
            "probabilities",
            "next_day_probabilities",
            "next_probabilities",
        ]:

            candidate = parent.get(key)

            if isinstance(candidate, dict):

                result = _extract_probability_dict(
                    candidate
                )

                if result is not None:
                    return result


    # --------------------------------------------------------
    # 3. Daily history fallback
    # --------------------------------------------------------

    histories = [
        hmm.get("daily_history"),
        hmm.get("history"),
        hmm.get("daily"),
    ]

    for parent_key in [
        "summary",
        "production_output",
        "hmm",
        "model",
    ]:

        parent = hmm.get(parent_key)

        if isinstance(parent, dict):

            histories.extend([
                parent.get("daily_history"),
                parent.get("history"),
                parent.get("daily"),
            ])


    for history in histories:

        if not isinstance(history, list):
            continue

        if not history:
            continue

        latest = history[-1]

        if not isinstance(latest, dict):
            continue

        # Current JSON format.
        direct = _extract_probability_dict(
            latest
        )

        if direct is not None:
            return direct

        # Explicit field format.
        candidates = [
            {
                "BULL": latest.get(
                    "bull_probability"
                ),
                "SIDE": latest.get(
                    "side_probability"
                ),
                "BEAR": latest.get(
                    "bear_probability"
                ),
            },
            {
                "BULL": latest.get(
                    "bull_prob"
                ),
                "SIDE": latest.get(
                    "side_prob"
                ),
                "BEAR": latest.get(
                    "bear_prob"
                ),
            },
        ]

        for candidate in candidates:

            if all(
                value is not None
                for value in candidate.values()
            ):

                return _normalize_probabilities(
                    np.asarray(
                        [
                            candidate["BULL"],
                            candidate["SIDE"],
                            candidate["BEAR"],
                        ],
                        dtype=float,
                    )
                )


    raise ValueError(
        "Could not extract HMM state probabilities "
        f"from {HMM_PATH}"
    )


def _matrix_from_records(
    records,
) -> np.ndarray | None:

    if not isinstance(records, list):
        return None

    if len(records) != 3:
        return None

    if not all(
        isinstance(row, dict)
        for row in records
    ):
        return None

    matrix = np.zeros(
        (3, 3),
        dtype=float,
    )

    for i, row in enumerate(records):

        for j, state in enumerate(
            STATE_NAMES
        ):

            value = None

            for key in [
                state,
                state.lower(),
            ]:

                if key in row:
                    value = row[key]
                    break

            if value is not None:
                matrix[i, j] = float(value)

    if np.any(matrix < 0):
        return None

    row_sums = matrix.sum(axis=1)

    if np.any(row_sums <= 0):
        return None

    matrix = (
        matrix
        / row_sums[:, None]
    )

    return matrix


def _matrix_from_dict(
    obj: dict,
) -> np.ndarray | None:

    if not isinstance(obj, dict):
        return None

    matrix = np.zeros(
        (3, 3),
        dtype=float,
    )

    found_rows = 0

    for i, source in enumerate(
        STATE_NAMES
    ):

        row = None

        if source in obj:
            row = obj[source]

        elif source.lower() in obj:
            row = obj[source.lower()]

        if not isinstance(row, dict):
            continue

        found_rows += 1

        for j, target in enumerate(
            STATE_NAMES
        ):

            value = None

            if target in row:
                value = row[target]

            elif target.lower() in row:
                value = row[target.lower()]

            if value is not None:
                matrix[i, j] = float(value)

    if found_rows != 3:
        return None

    if np.any(matrix < 0):
        return None

    row_sums = matrix.sum(axis=1)

    if np.any(row_sums <= 0):
        return None

    matrix = (
        matrix
        / row_sums[:, None]
    )

    return matrix


def get_transition_matrix(
    hmm: dict,
) -> np.ndarray:

    candidates = []

    # --------------------------------------------------------
    # Direct candidates
    # --------------------------------------------------------

    for key in [
        "transition_matrix",
        "reference_transition_matrix",
        "matrix",
    ]:

        if key in hmm:
            candidates.append(
                hmm[key]
            )


    # --------------------------------------------------------
    # Nested candidates
    # --------------------------------------------------------

    for parent_key in [
        "summary",
        "production_output",
        "hmm",
        "model",
    ]:

        parent = hmm.get(parent_key)

        if not isinstance(parent, dict):
            continue

        for key in [
            "transition_matrix",
            "reference_transition_matrix",
            "matrix",
        ]:

            if key in parent:
                candidates.append(
                    parent[key]
                )


    # --------------------------------------------------------
    # Try every representation
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate is None:
            continue


        # Direct numeric list / ndarray.
        try:

            arr = np.asarray(
                candidate,
                dtype=float,
            )

            if arr.shape == (3, 3):

                if np.all(arr >= 0):

                    row_sums = arr.sum(
                        axis=1
                    )

                    if np.all(
                        row_sums > 0
                    ):

                        return (
                            arr
                            / row_sums[:, None]
                        )

        except (TypeError, ValueError):
            pass


        # {"data": [[...], [...], [...]]}
        if isinstance(
            candidate,
            dict
        ):

            if "data" in candidate:

                try:

                    arr = np.asarray(
                        candidate["data"],
                        dtype=float,
                    )

                    if arr.shape == (3, 3):

                        row_sums = arr.sum(
                            axis=1
                        )

                        if np.all(
                            arr >= 0
                        ) and np.all(
                            row_sums > 0
                        ):

                            return (
                                arr
                                / row_sums[:, None]
                            )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass


            # Dict-of-dicts.
            result = _matrix_from_dict(
                candidate
            )

            if result is not None:
                return result


        # List of dict records.
        result = _matrix_from_records(
            candidate
        )

        if result is not None:
            return result


    raise ValueError(
        "Could not extract the 3x3 HMM "
        "transition matrix from "
        f"{HMM_PATH}"
    )


# ============================================================
# HMM HISTORICAL LABELS
# ============================================================

def build_hmm_labels(
    prices: pd.DataFrame,
) -> np.ndarray:

    """
    Reconstruct the simple reference regime used by
    the QuantOS HMM simulation interface.

    The production V10 HMM is NOT retrained here.

    Labels are explicitly aligned to the return series.
    """

    close = prices["close"].astype(float)

    log_return_20 = np.log(
        close
        / close.shift(20)
    )

    labels = pd.Series(
        np.nan,
        index=prices.index,
        dtype=object,
    )

    labels[
        log_return_20 > 0.05
    ] = "BULL"

    labels[
        log_return_20 < -0.05
    ] = "BEAR"

    middle = (
        log_return_20.notna()
        & (log_return_20 <= 0.05)
        & (log_return_20 >= -0.05)
    )

    labels[middle] = "SIDE"

    # --------------------------------------------------------
    # CRITICAL ALIGNMENT:
    #
    # pct_change() creates observations beginning at the
    # second price row. Therefore HMM labels must use the
    # exact same index.
    # --------------------------------------------------------

    returns = (
        close
        .pct_change()
        .dropna()
    )

    labels = labels.reindex(
        returns.index
    )

    valid = labels.notna()

    return labels.loc[
        valid
    ].to_numpy()


def align_returns_to_hmm_labels(
    prices: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray]:

    close = prices["close"].astype(float)

    returns = (
        close
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    log_return_20 = np.log(
        close
        / close.shift(20)
    )

    labels = pd.Series(
        np.nan,
        index=close.index,
        dtype=object,
    )

    labels[
        log_return_20 > 0.05
    ] = "BULL"

    labels[
        log_return_20 < -0.05
    ] = "BEAR"

    middle = (
        log_return_20.notna()
        & (log_return_20 <= 0.05)
        & (log_return_20 >= -0.05)
    )

    labels[middle] = "SIDE"

    aligned = pd.concat(
        [
            returns.rename("return"),
            labels.rename("regime"),
        ],
        axis=1,
    )

    aligned = aligned.dropna()

    return (
        aligned["return"].astype(float),
        aligned["regime"].to_numpy(),
    )


# ============================================================
# DISPLAY
# ============================================================

def model_name(
    model: str,
) -> str:

    names = {
        "student_t": "Student-t",
        "garch_t": "GARCH-t",
        "garch_jump": (
            "GARCH-t + empirical jumps"
        ),
        "bootstrap": "Block bootstrap",
        "hmm": "HMM regime switching",
    }

    return names.get(
        model,
        model,
    )


def clean_number(value):

    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        value = value.item()

    if isinstance(value, float):

        if not np.isfinite(value):
            return None

    return value


def clean_dict(obj):

    if isinstance(obj, dict):

        return {
            str(k): clean_dict(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            clean_dict(v)
            for v in obj
        ]

    return clean_number(obj)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices = load_prices()
    inputs = load_inputs()
    fresh_hmm = load_hmm()

    if prices.empty:
        raise RuntimeError(
            "NIFTY price dataset is empty."
        )


    # --------------------------------------------------------
    # Current market data
    # --------------------------------------------------------

    returns = get_returns(
        prices
    )

    latest_date = prices[
        "timestamp"
    ].iloc[-1]

    initial_value = float(
        prices["close"].iloc[-1]
    )


    # --------------------------------------------------------
    # Monte Carlo input artifact
    # --------------------------------------------------------

    input_hmm = inputs.get(
        "hmm",
        {},
    )

    garch = inputs.get(
        "garch",
        {},
    )

    options = inputs.get(
        "options",
        {},
    )

    atm_iv = options.get(
        "atm_iv"
    )


    # --------------------------------------------------------
    # FRESH production HMM
    #
    # This is intentionally read directly from the current
    # live HMM artifact so the Monte Carlo cannot silently
    # use stale transition probabilities.
    # --------------------------------------------------------

    current_probabilities = (
        get_current_probabilities(
            fresh_hmm
        )
    )

    transition_matrix = (
        get_transition_matrix(
            fresh_hmm
        )
    )


    # --------------------------------------------------------
    # HMM labels aligned to returns
    # --------------------------------------------------------

    hmm_returns, regime_labels = (
        align_returns_to_hmm_labels(
            prices
        )
    )


    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(hmm_returns) != len(
        regime_labels
    ):
        raise RuntimeError(
            "HMM returns and regime labels "
            "are not aligned."
        )

    if transition_matrix.shape != (
        3,
        3,
    ):
        raise RuntimeError(
            "HMM transition matrix must be 3x3."
        )

    if not np.isclose(
        transition_matrix.sum(
            axis=1
        ),
        1.0,
        atol=1e-6,
    ).all():
        raise RuntimeError(
            "HMM transition matrix rows "
            "do not sum to 1."
        )


    # --------------------------------------------------------
    # Console header
    # --------------------------------------------------------

    print("=" * 70)
    print("QuantOS NIFTY MONTE CARLO")
    print("=" * 70)

    print(
        f"Market date       : "
        f"{latest_date.date()}"
    )

    print(
        f"Starting value    : "
        f"₹{initial_value:,.2f}"
    )

    print(
        f"Current regime    : "
        f"{fresh_hmm.get('current_regime')}"
    )

    print(
        f"Next regime       : "
        f"{fresh_hmm.get('next_regime')}"
    )

    print(
        f"P(next BEAR)     : "
        f"{current_probabilities[2]:.2%}"
    )

    print(
        f"GARCH 21D        : "
        f"{float(garch.get('forecast_vol_21d')):.2%}"
    )

    print(
        f"ATM IV            : "
        f"{atm_iv if atm_iv is not None else 'unavailable'}"
    )

    print(
        f"Historical returns : "
        f"{len(returns)}"
    )

    print(
        f"Simulation paths  : "
        f"{N_PATHS:,}"
    )

    print()

    print(
        "HMM next-day probabilities:"
    )

    print(
        f"  BULL : "
        f"{current_probabilities[0]:.2%}"
    )

    print(
        f"  SIDE : "
        f"{current_probabilities[1]:.2%}"
    )

    print(
        f"  BEAR : "
        f"{current_probabilities[2]:.2%}"
    )

    print()

    print(
        "HMM transition matrix:"
    )

    print(
        pd.DataFrame(
            transition_matrix,
            index=STATE_NAMES,
            columns=STATE_NAMES,
        ).round(4)
    )

    print()


    # --------------------------------------------------------
    # Output structure
    # --------------------------------------------------------

    output = {

        "metadata": {

            "asset": "NIFTY_50",

            "market_date": (
                latest_date.strftime(
                    "%Y-%m-%d"
                )
            ),

            "starting_value": (
                initial_value
            ),

            "n_paths": N_PATHS,

            "seed": SEED,

            "historical_observations": (
                len(returns)
            ),

            "current_regime": (
                fresh_hmm.get(
                    "current_regime"
                )
            ),

            "next_regime": (
                fresh_hmm.get(
                    "next_regime"
                )
            ),

            "next_day_probabilities": {

                "BULL": float(
                    current_probabilities[0]
                ),

                "SIDE": float(
                    current_probabilities[1]
                ),

                "BEAR": float(
                    current_probabilities[2]
                ),
            },

            "hmm_transition_matrix": (
                transition_matrix
                .tolist()
            ),

            "garch_forecast": garch,

            "atm_iv": atm_iv,
        },

        "horizons": {},
    }


    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for horizon_label, horizon_name in (
        HORIZONS.items()
    ):

        horizon = mc.horizon_from_label(
            horizon_name
        )

        print("-" * 70)

        print(
            f"{horizon_label} "
            f"({horizon} trading days)"
        )

        print("-" * 70)

        output[
            "horizons"
        ][horizon_label] = {}


        for model_index, model in enumerate(
            MODELS
        ):

            print(
                f"Running "
                f"{model_name(model)}..."
            )


            # ------------------------------------------------
            # Deterministic but independent seed
            # ------------------------------------------------

            model_seed = (
                SEED
                + horizon
                + model_index * 1000
            )


            # ------------------------------------------------
            # HMM uses aligned HMM observations.
            #
            # Other models use the complete return series.
            # ------------------------------------------------

            simulation_returns = (
                hmm_returns
                if model == "hmm"
                else returns
            )


            simulation_labels = (
                regime_labels
                if model == "hmm"
                else None
            )


            simulation_transition = (
                transition_matrix
                if model == "hmm"
                else None
            )


            simulation_probabilities = (
                current_probabilities
                if model == "hmm"
                else None
            )


            # ------------------------------------------------
            # RUN MODEL
            # ------------------------------------------------

            result = mc.simulate_model(

                model=model,

                returns=simulation_returns,

                horizon=horizon,

                n_paths=N_PATHS,

                seed=model_seed,

                atm_iv=atm_iv,

                iv_weight=0.5,

                block_size=20,

                regime_labels=simulation_labels,

                transition_matrix=(
                    simulation_transition
                ),

                current_probabilities=(
                    simulation_probabilities
                ),

                degrees_of_freedom=8.0,
            )

                        # ------------------------------------------------
            # Extract simulated paths
            # ------------------------------------------------

            paths = np.asarray(
                result.paths,
                dtype=float,
            )

            if paths.ndim != 2:
                raise RuntimeError(
                    "Monte Carlo paths must be a 2D array. "
                    f"Received shape: {paths.shape}"
                )

            if paths.shape[0] != N_PATHS:
                raise RuntimeError(
                    "Unexpected number of Monte Carlo paths. "
                    f"Expected {N_PATHS}, got {paths.shape[0]}"
                )

            # ------------------------------------------------
            # Frontend path visualization
            # ------------------------------------------------

            path_visualization = build_path_visualization(
                paths=paths,
                initial_value=initial_value,
            )

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            summary = mc.summarize_simulation(
                paths=paths,
                initial_value=1.0,
                target_pct=0.10,
                stop_pct=0.10,
            )

            # ------------------------------------------------
            # Terminal return distribution
            # ------------------------------------------------

            terminal_returns = (
                paths[:, -1] - 1.0
            )

            percentiles = {
                "p05": float(
                    np.percentile(
                        terminal_returns,
                        5,
                    )
                ),
                "p25": float(
                    np.percentile(
                        terminal_returns,
                        25,
                    )
                ),
                "median": float(
                    np.percentile(
                        terminal_returns,
                        50,
                    )
                ),
                "p75": float(
                    np.percentile(
                        terminal_returns,
                        75,
                    )
                ),
                "p95": float(
                    np.percentile(
                        terminal_returns,
                        95,
                    )
                ),
            }

            # ------------------------------------------------
            # Convert returns to actual NIFTY price levels
            # ------------------------------------------------

            outcome_levels = {
                key: initial_value * (
                    1.0 + value
                )
                for key, value in percentiles.items()
            }

            # ------------------------------------------------
            # Save model output
            # ------------------------------------------------

            model_output = {
                "model": model_name(model),

                "horizon_days": horizon,

                "summary": clean_dict(
                    summary
                ),

                "terminal_return_percentiles": (
                    percentiles
                ),

                "possible_outcome_levels": (
                    outcome_levels
                ),

                "risk_probabilities": {
                    "finish_below_start": (
                        summary.get(
                            "prob_finish_below_start"
                        )
                    ),

                    "hit_plus_10pct": (
                        summary.get(
                            "prob_hit_target"
                        )
                    ),

                    "hit_minus_10pct": (
                        summary.get(
                            "prob_hit_stop"
                        )
                    ),
                },

                "path_visualization": (
                    path_visualization
                ),
            }

            output[
                "horizons"
            ][horizon_label][model] = model_output
            


            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"  P05    : "
                f"{percentiles['p05']:.2%}"
            )

            print(
                f"  Median : "
                f"{percentiles['median']:.2%}"
            )

            print(
                f"  P95    : "
                f"{percentiles['p95']:.2%}"
            )

            below_start = summary.get(
            "prob_finish_below_start"
)

            if below_start is None:
             print("  Below start : n/a")
            else:
             print(
        f"  Below start : "
        f"{below_start:.2%}"
    )
            print()
    
        # ========================================================
    # FINAL OUTPUT VALIDATION
    # ========================================================

    for horizon_label in HORIZONS:
        horizon_output = output[
            "horizons"
        ].get(horizon_label)

        if not isinstance(
            horizon_output,
            dict,
        ):
            raise RuntimeError(
                f"Invalid horizon output: "
                f"{horizon_label}"
            )

        for model in MODELS:
            model_output = horizon_output.get(
                model
            )

            if not isinstance(
                model_output,
                dict,
            ):
                raise RuntimeError(
                    f"Invalid model output: "
                    f"{horizon_label}/{model}"
                )

            visualization = model_output.get(
                "path_visualization"
            )

            if not isinstance(
                visualization,
                dict,
            ):
                raise RuntimeError(
                    f"Missing path_visualization: "
                    f"{horizon_label}/{model}"
                )

            if not isinstance(
                visualization.get("paths"),
                list,
            ):
                raise RuntimeError(
                    f"Invalid visualization paths: "
                    f"{horizon_label}/{model}"
                )

            if not isinstance(
                visualization.get(
                    "percentile_paths"
                ),
                dict,
            ):
                raise RuntimeError(
                    f"Invalid percentile paths: "
                    f"{horizon_label}/{model}"
                )


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_JSON,
        "w",
    ) as f:

        json.dump(
            clean_dict(output),
            f,
            indent=2,
        )


    print("=" * 70)
    print("MONTE CARLO COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_JSON}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()