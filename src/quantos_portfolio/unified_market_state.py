"""
QuantOS Unified Market State
============================

Combines existing QuantOS research outputs into one transparent
market-state object.

Sources:
    1. HMM regime / conditional volatility
    2. GARCH / HAR volatility forecasts
    3. NIFTY 50 option-implied volatility surface
    4. NIFTY 50 SABR surface

Important:
    The sources may have different observation timestamps.
    They are NOT artificially aligned.

The resulting state preserves:
    - HMM observation date
    - volatility-model observation date
    - options surface generation timestamp

No arbitrary weighted "QuantOS score" is created.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ================================================================
# PATHS
# ================================================================

HMM_PATH = Path(
    "data/portfolio/hmm_volatility_bridge/"
    "current_quantos_market_state.csv"
)

VOL_PATH = Path(
    "data/regime/volatility_predictions/"
    "nifty_50.json"
)

IV_PATH = Path(
    "data/regime/volatility_surface/"
    "nifty_50_iv_surface.json"
)

SABR_PATH = Path(
    "data/regime/volatility_surface/"
    "nifty_50_sabr_surface.json"
)

OUTPUT_DIR = Path(
    "data/portfolio/unified_market_state"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    /
    "current_market_state.csv"
)


# ================================================================
# HELPERS
# ================================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def safe_float(value):
    """
    Convert a value to float where possible.
    """
    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return np.nan

    except (
        TypeError,
        ValueError,
    ):
        return np.nan


def nearest_surface_iv(
    surface,
    expiry,
    target_strike,
):
    """
    Find the IV observation nearest to the requested strike
    for a specific expiry.
    """

    rows = [
        row
        for row in surface
        if row.get("expiry") == expiry
    ]

    if not rows:
        return np.nan

    valid = []

    for row in rows:

        strike = safe_float(
            row.get("strike")
        )

        iv = safe_float(
            row.get("implied_volatility")
        )

        if (
            np.isfinite(strike)
            and np.isfinite(iv)
        ):

            valid.append(
                (
                    abs(
                        strike
                        -
                        target_strike
                    ),
                    iv,
                    row
                )
            )

    if not valid:
        return np.nan

    valid.sort(
        key=lambda x: x[0]
    )

    return valid[0][1]


def find_atm_iv(
    surface,
    expiry,
    atm_strike,
):
    """
    Prefer the explicit is_atm observation.
    Otherwise use the nearest strike.
    """

    rows = [
        row
        for row in surface
        if row.get("expiry") == expiry
    ]

    for row in rows:

        if row.get("is_atm") is True:

            iv = safe_float(
                row.get(
                    "implied_volatility"
                )
            )

            if np.isfinite(iv):
                return iv

    return nearest_surface_iv(
        surface,
        expiry,
        atm_strike
    )


def calculate_skew(
    surface,
    expiry,
    atm_strike,
):
    """
    Calculate a transparent local option skew:

        25-delta is NOT available in the supplied surface.

    Therefore we use:
        average IV near -3% moneyness
        minus ATM IV

    This is explicitly moneyness-based skew, not
    a conventional 25-delta risk reversal.
    """

    rows = [
        row
        for row in surface
        if row.get("expiry") == expiry
    ]

    atm_iv = find_atm_iv(
        surface,
        expiry,
        atm_strike
    )

    if not np.isfinite(atm_iv):
        return np.nan

    otm_puts = []

    for row in rows:

        moneyness = safe_float(
            row.get("moneyness")
        )

        iv = safe_float(
            row.get(
                "implied_volatility"
            )
        )

        if (
            np.isfinite(moneyness)
            and np.isfinite(iv)
            and -0.035
            <= moneyness
            <= -0.025
        ):

            otm_puts.append(
                iv
            )

    if not otm_puts:
        return np.nan

    return (
        float(np.mean(otm_puts))
        -
        atm_iv
    )


def calculate_term_structure(
    surface,
    expiries,
    atm_strike,
):
    """
    Calculate the difference between the nearest available
    expiry ATM IV and the next expiry ATM IV.
    """

    if len(expiries) < 2:
        return np.nan

    first_iv = find_atm_iv(
        surface,
        expiries[0],
        atm_strike
    )

    second_iv = find_atm_iv(
        surface,
        expiries[1],
        atm_strike
    )

    if (
        not np.isfinite(first_iv)
        or
        not np.isfinite(second_iv)
    ):
        return np.nan

    return (
        second_iv
        -
        first_iv
    )


def find_sabr_atm_values(
    sabr_surface,
    expiry,
    atm_strike,
):
    """
    Extract SABR IV and parameters near ATM.

    The SABR JSON may contain surface rows rather than a separate
    parameter object, so this function handles both common layouts.
    """

    if not isinstance(
        sabr_surface,
        dict
    ):
        return {
            "sabr_iv": np.nan,
            "sabr_alpha": np.nan,
            "sabr_beta": np.nan,
            "sabr_rho": np.nan,
            "sabr_nu": np.nan,
        }

    # ------------------------------------------------------------
    # Case 1: explicit SABR parameter objects
    # ------------------------------------------------------------

    parameter_containers = []

    for key in [
        "parameters",
        "sabr_parameters",
        "calibration",
        "calibrations",
        "surfaces",
    ]:

        value = sabr_surface.get(
            key
        )

        if value is not None:
            parameter_containers.append(
                value
            )

    # ------------------------------------------------------------
    # Case 2: surface rows
    # ------------------------------------------------------------

    rows = []

    for key in [
        "surface",
        "data",
        "rows",
    ]:

        value = sabr_surface.get(
            key
        )

        if isinstance(
            value,
            list
        ):

            rows.extend(
                value
            )

    matching_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("expiry") == expiry
    ]

    if matching_rows:

        closest = None
        closest_distance = np.inf

        for row in matching_rows:

            strike = safe_float(
                row.get("strike")
            )

            if not np.isfinite(
                strike
            ):
                continue

            distance = abs(
                strike
                -
                atm_strike
            )

            if distance < closest_distance:

                closest_distance = distance
                closest = row

        if closest is not None:

            return {
                "sabr_iv":
                    safe_float(
                        closest.get(
                            "sabr_iv"
                        )
                    ),

                "sabr_alpha":
                    safe_float(
                        closest.get(
                            "alpha"
                        )
                    ),

                "sabr_beta":
                    safe_float(
                        closest.get(
                            "beta"
                        )
                    ),

                "sabr_rho":
                    safe_float(
                        closest.get(
                            "rho"
                        )
                    ),

                "sabr_nu":
                    safe_float(
                        closest.get(
                            "nu"
                        )
                    ),
            }

    # ------------------------------------------------------------
    # Search nested parameter dictionaries.
    # ------------------------------------------------------------

    def search_dict(obj):

        if isinstance(
            obj,
            dict
        ):

            lower = {
                str(k).lower(): v
                for k, v in obj.items()
            }

            result = {}

            for key in [
                "alpha",
                "beta",
                "rho",
                "nu",
            ]:

                if key in lower:

                    result[
                        f"sabr_{key}"
                    ] = safe_float(
                        lower[key]
                    )

            if result:

                return result

            for value in obj.values():

                found = search_dict(
                    value
                )

                if found:
                    return found

        elif isinstance(
            obj,
            list
        ):

            for value in obj:

                found = search_dict(
                    value
                )

                if found:
                    return found

        return {}

    parameters = search_dict(
        sabr_surface
    )

    return {
        "sabr_iv": np.nan,
        "sabr_alpha":
            parameters.get(
                "sabr_alpha",
                np.nan
            ),
        "sabr_beta":
            parameters.get(
                "sabr_beta",
                np.nan
            ),
        "sabr_rho":
            parameters.get(
                "sabr_rho",
                np.nan
            ),
        "sabr_nu":
            parameters.get(
                "sabr_nu",
                np.nan
            ),
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("QUANTOS UNIFIED MARKET STATE")
    print("=" * 70)

    # ============================================================
    # CHECK FILES
    # ============================================================

    for path in [
        HMM_PATH,
        VOL_PATH,
        IV_PATH,
        SABR_PATH,
    ]:

        require_file(
            path
        )

    # ============================================================
    # LOAD HMM
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING HMM STATE")
    print("=" * 70)

    hmm = pd.read_csv(
        HMM_PATH
    )

    if hmm.empty:
        raise ValueError(
            "HMM state file is empty."
        )

    hmm_row = (
        hmm.iloc[-1]
    )

    hmm_date = (
        pd.to_datetime(
            hmm_row["date"]
        )
    )

    hmm_regime = str(
        hmm_row["regime"]
    )

    p_bull = safe_float(
        hmm_row["p_bull"]
    )

    p_side = safe_float(
        hmm_row["p_side"]
    )

    p_bear = safe_float(
        hmm_row["p_bear"]
    )

    current_vol_5d = safe_float(
        hmm_row["current_vol_5d"]
    )

    current_vol_20d = safe_float(
        hmm_row["current_vol_20d"]
    )

    current_vol_60d = safe_float(
        hmm_row["current_vol_60d"]
    )

    regime_conditioned_21d_vol = (
        safe_float(
            hmm_row[
                "expected_forward_21d_vol"
            ]
        )
    )

    # ============================================================
    # LOAD GARCH / HAR
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING GARCH / HAR")
    print("=" * 70)

    with open(
        VOL_PATH,
        "r"
    ) as f:

        volatility = json.load(
            f
        )

    live = volatility.get(
        "live",
        {}
    )

    garch = live.get(
        "garch",
        {}
    )

    har = live.get(
        "har",
        {}
    )

    volatility_date = (
        pd.to_datetime(
            live.get(
                "latest_date"
            )
        )
    )

    latest_close = safe_float(
        live.get(
            "latest_close"
        )
    )

    realized_21d = safe_float(
        live.get(
            "current_realized_vol_21d"
        )
    )

    long_run_21d = safe_float(
        live.get(
            "long_run_vol_21d"
        )
    )

    volatility_ratio = safe_float(
        live.get(
            "volatility_ratio"
        )
    )

    volatility_level = live.get(
        "volatility_level"
    )

    garch_1d = safe_float(
        garch.get(
            "forecast_vol_1d"
        )
    )

    garch_5d = safe_float(
        garch.get(
            "forecast_vol_5d"
        )
    )

    garch_21d = safe_float(
        garch.get(
            "forecast_vol_21d"
        )
    )

    har_1d = safe_float(
        har.get(
            "forecast_vol_1d"
        )
    )

    # ============================================================
    # LOAD IV SURFACE
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING OPTIONS IV SURFACE")
    print("=" * 70)

    with open(
        IV_PATH,
        "r"
    ) as f:

        iv_data = json.load(
            f
        )

    options_generated_at = (
        iv_data.get(
            "generated_at"
        )
    )

    options_spot = safe_float(
        iv_data.get(
            "spot"
        )
    )

    atm_strike = safe_float(
        iv_data.get(
            "atm_strike"
        )
    )

    expiries = iv_data.get(
        "expiries",
        []
    )

    surface = iv_data.get(
        "surface",
        []
    )

    if not expiries:
        raise ValueError(
            "No option expiries found."
        )

    nearest_expiry = (
        expiries[0]
    )

    next_expiry = (
        expiries[1]
        if len(expiries) > 1
        else None
    )

    atm_iv_nearest = find_atm_iv(
        surface,
        nearest_expiry,
        atm_strike
    )

    atm_iv_next = (
        find_atm_iv(
            surface,
            next_expiry,
            atm_strike
        )
        if next_expiry
        else np.nan
    )

    moneyness_skew = (
        calculate_skew(
            surface,
            nearest_expiry,
            atm_strike
        )
    )

    term_structure = (
        calculate_term_structure(
            surface,
            expiries,
            atm_strike
        )
    )

    # ============================================================
    # IV / REALIZED VOL COMPARISON
    # ============================================================

    if (
        np.isfinite(atm_iv_nearest)
        and
        np.isfinite(realized_21d)
        and
        realized_21d > 0
    ):

        iv_rv_ratio = (
            atm_iv_nearest
            /
            realized_21d
        )

        iv_minus_rv = (
            atm_iv_nearest
            -
            realized_21d
        )

    else:

        iv_rv_ratio = np.nan
        iv_minus_rv = np.nan

    # ============================================================
    # LOAD SABR
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING SABR SURFACE")
    print("=" * 70)

    with open(
        SABR_PATH,
        "r"
    ) as f:

        sabr_data = json.load(
            f
        )

    sabr_values = (
        find_sabr_atm_values(
            sabr_data,
            nearest_expiry,
            atm_strike
        )
    )

    # ============================================================
    # DERIVED TRANSPARENT STATES
    # ============================================================

    # ------------------------------------------------------------
    # Realized volatility state
    # ------------------------------------------------------------

    if (
        np.isfinite(
            volatility_ratio
        )
    ):

        if volatility_ratio < 0.75:

            realized_vol_state = (
                "LOW"
            )

        elif volatility_ratio < 1.25:

            realized_vol_state = (
                "NORMAL"
            )

        else:

            realized_vol_state = (
                "ELEVATED"
            )

    else:

        realized_vol_state = (
            "UNAVAILABLE"
        )

    # ------------------------------------------------------------
    # IV versus realized volatility
    #
    # This is descriptive, not a trading signal.
    # ------------------------------------------------------------

    if np.isfinite(
        iv_rv_ratio
    ):

        if iv_rv_ratio < 0.90:

            implied_vs_realized = (
                "IV_BELOW_REALIZED"
            )

        elif iv_rv_ratio > 1.10:

            implied_vs_realized = (
                "IV_ABOVE_REALIZED"
            )

        else:

            implied_vs_realized = (
                "IV_NEAR_REALIZED"
            )

    else:

        implied_vs_realized = (
            "UNAVAILABLE"
        )

    # ------------------------------------------------------------
    # Term structure
    # ------------------------------------------------------------

    if np.isfinite(
        term_structure
    ):

        if term_structure > 0.005:

            term_structure_state = (
                "UPWARD"
            )

        elif term_structure < -0.005:

            term_structure_state = (
                "DOWNWARD"
            )

        else:

            term_structure_state = (
                "FLAT"
            )

    else:

        term_structure_state = (
            "UNAVAILABLE"
        )

    # ============================================================
    # BUILD UNIFIED STATE
    # ============================================================

    unified = pd.DataFrame(
        [
            {

                # ------------------------------------------------
                # Identity
                # ------------------------------------------------

                "asset":
                    "NIFTY_50",

                # ------------------------------------------------
                # HMM
                # ------------------------------------------------

                "hmm_date":
                    hmm_date,

                "hmm_regime":
                    hmm_regime,

                "hmm_p_bull":
                    p_bull,

                "hmm_p_side":
                    p_side,

                "hmm_p_bear":
                    p_bear,

                "hmm_current_vol_5d":
                    current_vol_5d,

                "hmm_current_vol_20d":
                    current_vol_20d,

                "hmm_current_vol_60d":
                    current_vol_60d,

                "hmm_regime_conditioned_21d_vol":
                    regime_conditioned_21d_vol,

                # ------------------------------------------------
                # GARCH / HAR
                # ------------------------------------------------

                "volatility_model_date":
                    volatility_date,

                "latest_close":
                    latest_close,

                "realized_vol_21d":
                    realized_21d,

                "long_run_vol_21d":
                    long_run_21d,

                "realized_vol_ratio":
                    volatility_ratio,

                "realized_vol_level":
                    volatility_level,

                "garch_1d":
                    garch_1d,

                "garch_5d":
                    garch_5d,

                "garch_21d":
                    garch_21d,

                "har_1d":
                    har_1d,

                # ------------------------------------------------
                # OPTIONS
                # ------------------------------------------------

                "options_generated_at":
                    options_generated_at,

                "options_spot":
                    options_spot,

                "atm_strike":
                    atm_strike,

                "nearest_expiry":
                    nearest_expiry,

                "next_expiry":
                    next_expiry,

                "atm_iv_nearest":
                    atm_iv_nearest,

                "atm_iv_next":
                    atm_iv_next,

                "iv_rv_ratio":
                    iv_rv_ratio,

                "iv_minus_realized":
                    iv_minus_rv,

                "moneyness_skew":
                    moneyness_skew,

                "term_structure_difference":
                    term_structure,

                # ------------------------------------------------
                # SABR
                # ------------------------------------------------

                "sabr_iv":
                    sabr_values[
                        "sabr_iv"
                    ],

                "sabr_alpha":
                    sabr_values[
                        "sabr_alpha"
                    ],

                "sabr_beta":
                    sabr_values[
                        "sabr_beta"
                    ],

                "sabr_rho":
                    sabr_values[
                        "sabr_rho"
                    ],

                "sabr_nu":
                    sabr_values[
                        "sabr_nu"
                    ],

                # ------------------------------------------------
                # Transparent derived descriptors
                # ------------------------------------------------

                "realized_vol_state":
                    realized_vol_state,

                "implied_vs_realized_state":
                    implied_vs_realized,

                "term_structure_state":
                    term_structure_state,
            }
        ]
    )

    # ============================================================
    # SAVE
    # ============================================================

    unified.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ============================================================
    # PRINT
    # ============================================================

    print()
    print("=" * 70)
    print("UNIFIED QUANTOS MARKET STATE")
    print("=" * 70)

    print(
        f"Asset                 : NIFTY 50"
    )

    print(
        f"HMM date              : "
        f"{hmm_date.date()}"
    )

    print(
        f"HMM regime            : "
        f"{hmm_regime}"
    )

    print(
        f"P(BULL)               : "
        f"{p_bull:.2%}"
    )

    print(
        f"P(SIDE)               : "
        f"{p_side:.2%}"
    )

    print(
        f"P(BEAR)               : "
        f"{p_bear:.2%}"
    )

    print()
    print(
        f"Realized 21D Vol      : "
        f"{realized_21d:.2%}"
    )

    print(
        f"Long-run 21D Vol      : "
        f"{long_run_21d:.2%}"
    )

    print(
        f"Volatility Ratio      : "
        f"{volatility_ratio:.3f}"
    )

    print(
        f"Volatility Level      : "
        f"{volatility_level}"
    )

    print()
    print(
        f"GARCH 1D              : "
        f"{garch_1d:.2%}"
    )

    print(
        f"GARCH 5D              : "
        f"{garch_5d:.2%}"
    )

    print(
        f"GARCH 21D             : "
        f"{garch_21d:.2%}"
    )

    print(
        f"HAR 1D                : "
        f"{har_1d:.2%}"
    )

    print()
    print(
        f"Options date          : "
        f"{options_generated_at}"
    )

    print(
        f"Options spot          : "
        f"{options_spot:.2f}"
    )

    print(
        f"ATM strike            : "
        f"{atm_strike:.2f}"
    )

    print(
        f"Nearest expiry        : "
        f"{nearest_expiry}"
    )

    print(
        f"ATM IV                : "
        f"{atm_iv_nearest:.2%}"
    )

    print(
        f"Next expiry ATM IV    : "
        f"{atm_iv_next:.2%}"
    )

    print(
        f"IV / Realized Vol     : "
        f"{iv_rv_ratio:.3f}"
    )

    print(
        f"Moneyness skew        : "
        f"{moneyness_skew:.2%}"
    )

    print(
        f"Term structure        : "
        f"{term_structure:.2%}"
    )

    print()
    print(
        f"Regime-conditioned "
        f"21D Vol              : "
        f"{regime_conditioned_21d_vol:.2%}"
    )

    print(
        f"Realized Vol State    : "
        f"{realized_vol_state}"
    )

    print(
        f"IV vs Realized        : "
        f"{implied_vs_realized}"
    )

    print(
        f"Term Structure State  : "
        f"{term_structure_state}"
    )

    print()
    print("=" * 70)
    print("FILE SAVED")
    print("=" * 70)

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("UNIFIED MARKET STATE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()