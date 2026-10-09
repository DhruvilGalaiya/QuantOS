from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRICE_PATH = ROOT / "data/regime/daily/nifty_50.parquet"
VOL_PATH = ROOT / "data/regime/volatility/nifty_50.parquet"
GARCH_PATH = ROOT / "data/regime/volatility_predictions/nifty_50.json"
HMM_PATH = ROOT / "data/regime/live_predictions/nifty_50.json"

IV_SURFACE_PATH = (
    ROOT / "data/regime/volatility_surface/nifty_50_iv_surface.json"
)

OPTION_DB_PATH = (
    ROOT / "data/regime/options_live/options_live.sqlite"
)

OUTPUT_DIR = ROOT / "data/portfolio/monte_carlo_inputs"
OUTPUT_PATH = OUTPUT_DIR / "nifty_50_monte_carlo_inputs.json"


def load_price_data() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_PATH).copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp", "close"]
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
    )

    return df


def load_volatility_data() -> pd.DataFrame:
    if not VOL_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(VOL_PATH).copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.sort_values("timestamp")

    return df


def load_garch() -> dict:
    if not GARCH_PATH.exists():
        return {}

    with open(GARCH_PATH, "r") as f:
        data = json.load(f)

    return data.get("live", data)


def load_hmm() -> dict:
    if not HMM_PATH.exists():
        return {}

    with open(HMM_PATH, "r") as f:
        data = json.load(f)

    return data.get("summary", data)


def get_realized_vol(
    vol_df: pd.DataFrame,
) -> dict:

    if vol_df.empty:
        return {}

    row = vol_df.iloc[-1]

    result = {}

    mapping = {
        "realized_vol_1d": [
            "realized_vol_1d",
            "rv_1d",
        ],
        "realized_vol_5d": [
            "realized_vol_5d",
            "rv_5d",
        ],
        "realized_vol_21d": [
            "realized_vol_21d",
            "rv_21d",
        ],
        "forward_vol_5d": [
            "forward_vol_5d",
        ],
        "forward_vol_21d": [
            "forward_vol_21d",
        ],
    }

    for output_name, candidates in mapping.items():

        for column in candidates:

            if column not in vol_df.columns:
                continue

            value = row[column]

            if pd.notna(value):
                result[output_name] = float(value)

            break

    return result


def load_atm_iv_from_surface() -> Optional[float]:
    """
    Load the nearest-expiry ATM IV from the validated
    NIFTY 50 IV surface.

    The IV surface engine explicitly identifies ATM rows,
    so this is preferred over taking a generic median of
    option IV observations from SQLite.
    """

    if not IV_SURFACE_PATH.exists():
        return None

    try:

        with open(IV_SURFACE_PATH, "r") as f:
            data = json.load(f)

        surface = data.get("surface", [])

        atm_rows = [
            row
            for row in surface
            if row.get("is_atm") is True
        ]

        if not atm_rows:
            return None

        atm_rows.sort(
            key=lambda row: (
                int(
                    row.get(
                        "days_to_expiry",
                        10**9,
                    )
                ),
                str(
                    row.get(
                        "expiry",
                        "",
                    )
                ),
            )
        )

        atm_iv = atm_rows[0].get(
            "implied_volatility"
        )

        if atm_iv is None:
            return None

        atm_iv = float(atm_iv)

        if 0 < atm_iv < 5:
            return atm_iv

    except Exception:
        return None

    return None


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices = load_price_data()
    volatility = load_volatility_data()
    garch = load_garch()
    hmm = load_hmm()

    if prices.empty:
        raise RuntimeError(
            "NIFTY price dataset is empty."
        )

    latest = prices.iloc[-1]

    latest_date = pd.Timestamp(
        latest["timestamp"]
    )

    latest_close = float(
        latest["close"]
    )

    realized = get_realized_vol(
        volatility
    )

    garch_date = garch.get(
        "latest_date"
    )

    garch_1d = (
        garch
        .get("garch", {})
        .get("forecast_vol_1d")
    )

    garch_5d = (
        garch
        .get("garch", {})
        .get("forecast_vol_5d")
    )

    garch_21d = (
        garch
        .get("garch", {})
        .get("forecast_vol_21d")
    )

    har_1d = (
        garch
        .get("har", {})
        .get("forecast_vol_1d")
    )

    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    atm_iv = load_atm_iv_from_surface()

    option_database = (
        str(OPTION_DB_PATH)
        if OPTION_DB_PATH.exists()
        else None
    )

    # --------------------------------------------------------
    # Current realized volatility
    # --------------------------------------------------------

    current_realized_21d = garch.get(
        "current_realized_vol_21d"
    )

    if current_realized_21d is None:
        current_realized_21d = realized.get(
            "realized_vol_21d"
        )

    # --------------------------------------------------------
    # Volatility anchor
    #
    # GARCH 21D and nearest-expiry ATM IV are averaged
    # when both are available.
    # --------------------------------------------------------

    if (
        garch_21d is not None
        and atm_iv is not None
    ):

        anchor = float(
            np.mean(
                [
                    float(garch_21d),
                    float(atm_iv),
                ]
            )
        )

        anchor_method = (
            "GARCH 21D + ATM IV average"
        )

    elif garch_21d is not None:

        anchor = float(
            garch_21d
        )

        anchor_method = "GARCH 21D"

    elif atm_iv is not None:

        anchor = float(
            atm_iv
        )

        anchor_method = "ATM IV"

    elif current_realized_21d is not None:

        anchor = float(
            current_realized_21d
        )

        anchor_method = (
            "21D realized volatility"
        )

    else:

        anchor = None
        anchor_method = "Unavailable"

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    probabilities = hmm.get(
        "probabilities",
        {},
    )

    transition_matrix = hmm.get(
        "transition_matrix",
        {},
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "asset": "NIFTY_50",

        "market": {
            "latest_date":
                latest_date.strftime(
                    "%Y-%m-%d"
                ),

            "next_trading_session":
                "2026-10-05",

            "latest_close":
                latest_close,
        },

        "data_freshness": {

            "market_date":
                latest_date.strftime(
                    "%Y-%m-%d"
                ),

            "volatility_date":
                garch_date,

            "hmm_date":
                hmm.get(
                    "latest_date"
                ),

            "option_database":
                option_database,

            "iv_surface":
                (
                    str(IV_SURFACE_PATH)
                    if IV_SURFACE_PATH.exists()
                    else None
                ),

        },

        "realized_volatility":
            realized,

        "garch": {

            "model":
                garch
                .get("garch", {})
                .get("model"),

            "forecast_vol_1d":
                garch_1d,

            "forecast_vol_5d":
                garch_5d,

            "forecast_vol_21d":
                garch_21d,

        },

        "har": {

            "forecast_vol_1d":
                har_1d,

        },

        "options": {

            "atm_iv":
                atm_iv,

            "available":
                atm_iv is not None,

            "source":
                (
                    str(IV_SURFACE_PATH)
                    if atm_iv is not None
                    else None
                ),

            "definition":
                (
                    "Nearest-expiry ATM IV "
                    "from validated NIFTY 50 "
                    "IV surface"
                ),

        },

        "volatility_anchor": {

            "value":
                anchor,

            "method":
                anchor_method,

        },

        "hmm": {

            "current_regime":
                hmm.get(
                    "current_regime"
                ),

            "current_confidence":
                hmm.get(
                    "current_raw_hmm_confidence"
                ),

            "next_regime":
                hmm.get(
                    "next_regime"
                ),

            "next_day_probability":
                hmm.get(
                    "next_day_probability"
                ),

            "probabilities":
                probabilities,

            "transition_matrix":
                transition_matrix,

            "stickiness_score":
                hmm.get(
                    "stickiness_score"
                ),

            "stickiness_level":
                hmm.get(
                    "stickiness_level"
                ),

            "current_regime_streak":
                hmm.get(
                    "current_regime_streak"
                ),

        },

    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("=" * 70)
    print(
        "QuantOS MONTE CARLO INPUT BUILDER"
    )
    print("=" * 70)

    print(
        f"Latest market date : "
        f"{latest_date.date()}"
    )

    print(
        f"Latest NIFTY close : "
        f"₹{latest_close:,.2f}"
    )

    print()

    print(
        "21D realized vol   :",
        (
            f"{current_realized_21d:.2%}"
            if current_realized_21d is not None
            else "unavailable"
        ),
    )

    print(
        "GARCH 1D/5D/21D   :",
        (
            f"{float(garch_1d):.2%} / "
            f"{float(garch_5d):.2%} / "
            f"{float(garch_21d):.2%}"
            if all(
                x is not None
                for x in [
                    garch_1d,
                    garch_5d,
                    garch_21d,
                ]
            )
            else "unavailable"
        ),
    )

    print(
        "HAR 1D             :",
        (
            f"{float(har_1d):.2%}"
            if har_1d is not None
            else "unavailable"
        ),
    )

    print(
        "Volatility date    :",
        garch_date,
    )

    print(
        "HMM date           :",
        hmm.get("latest_date"),
    )

    print(
        "Current regime     :",
        hmm.get("current_regime"),
    )

    print(
        "Next regime        :",
        hmm.get("next_regime"),
    )

    print(
        "Next-day probability:",
        (
            f"{float(hmm.get('next_day_probability')):.2%}"
            if hmm.get(
                "next_day_probability"
            ) is not None
            else "unavailable"
        ),
    )

    print(
        "Option database    :",
        (
            option_database
            if option_database is not None
            else "NONE"
        ),
    )

    print(
        "IV surface         :",
        (
            str(IV_SURFACE_PATH)
            if IV_SURFACE_PATH.exists()
            else "NONE"
        ),
    )

    print(
        "ATM IV             :",
        (
            f"{atm_iv:.2%}"
            if atm_iv is not None
            else "unavailable"
        ),
    )

    print(
        "Anchor             :",
        (
            f"{anchor:.2%}"
            if anchor is not None
            else "unavailable"
        ),
    )

    print(
        "Method             :",
        anchor_method,
    )

    print()

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()