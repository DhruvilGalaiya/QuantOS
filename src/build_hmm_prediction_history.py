"""
QuantOS HMM Historical Prediction History

Builds an auditable record of the V10 HMM's historical
next-trading-day regime predictions.

For each prediction date T:

    Data through T
         ↓
    V10 prediction
         ↓
    Target = next available trading session
         ↓
    Actual = observed V10 regime on target session

The latest prediction remains PENDING until the next
trading session's data becomes available.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("data/regime/live_predictions")

LATEST_FILE = (
    OUTPUT_DIR / "latest_predictions.json"
)

# Show two years of history in the dashboard.
HISTORY_YEARS = 2


# ============================================================
# LOAD V10 OUTPUT
# ============================================================

def load_predictions():

    if not LATEST_FILE.exists():
        raise FileNotFoundError(
            f"\nV10 prediction file not found:\n"
            f"{LATEST_FILE}\n\n"
            f"Run this first:\n"
            f"python src/run_live_hmm.py"
        )

    with LATEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# BUILD ONE SYMBOL'S HISTORY
# ============================================================

def prepare_symbol_history(
    symbol,
    payload,
):

    daily_output = payload.get(
        "daily_output",
        [],
    )

    if not daily_output:
        raise ValueError(
            f"{symbol}: daily_output is missing or empty."
        )

    df = pd.DataFrame(
        daily_output
    )

    required_columns = [
        "date",
        "current_regime",
        "current_raw_hmm_confidence",
        "next_regime",
        "next_day_probability",
        "bull_probability",
        "side_probability",
        "bear_probability",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: missing V10 columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean dates
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Keep last two years
    # --------------------------------------------------------

    latest_date = df["date"].max()

    cutoff_date = (
        latest_date
        - pd.DateOffset(
            years=HISTORY_YEARS
        )
    )

    df = df[
        df["date"] >= cutoff_date
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Prediction made on T is for the NEXT AVAILABLE
    # TRADING SESSION.
    #
    # Example:
    #
    # Sep 28 → prediction for Sep 29
    #
    # Friday → Monday also works automatically.
    # --------------------------------------------------------

    # Next available trading date for historical rows.
    # --------------------------------------------------------
    # Target trading date
    # --------------------------------------------------------

    df["target_trading_date"] = (
        df["date"].shift(-1)
    )

    # Latest prediction targets the next calendar day
    # when the next trading observation is not available.
    if len(df) > 0 and pd.isna(
        df.iloc[-1]["target_trading_date"]
    ):

        latest_prediction_date = (
            df.iloc[-1]["date"]
        )

        next_calendar_day = (
            latest_prediction_date
            + pd.Timedelta(days=1)
        )

        df.loc[
            df.index[-1],
            "target_trading_date"
        ] = next_calendar_day

    # --------------------------------------------------------
    # Actual regime on the next trading session
    # --------------------------------------------------------

    df["actual_next_regime"] = (
        df["current_regime"].shift(-1)
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    df["agreement"] = pd.NA

    evaluated = (
        df["actual_next_regime"].notna()
    )

    df.loc[
        evaluated,
        "agreement",
    ] = (
        df.loc[
            evaluated,
            "next_regime",
        ]
        ==
        df.loc[
            evaluated,
            "actual_next_regime",
        ]
    )

    # --------------------------------------------------------
    # Add symbol
    # --------------------------------------------------------

    df.insert(
        0,
        "symbol",
        symbol,
    )

    # --------------------------------------------------------
    # Rename prediction date
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "date": "prediction_date"
        }
    )

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    columns = [
        "symbol",
        "prediction_date",
        "target_trading_date",
        "current_regime",
        "current_raw_hmm_confidence",
        "bull_probability",
        "side_probability",
        "bear_probability",
        "next_regime",
        "next_day_probability",
        "actual_next_regime",
        "agreement",
        "stay_probability",
        "recent_consistency",
        "current_regime_streak",
        "stickiness_score",
    ]

    columns = [
        col
        for col in columns
        if col in df.columns
    ]

    return df[
        columns
    ]


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    symbol,
    df,
):

    evaluated = df[
        df["actual_next_regime"].notna()
    ].copy()

    pending = df[
        df["actual_next_regime"].isna()
    ].copy()

    evaluated_count = len(
        evaluated
    )

    agreement_count = int(
        evaluated["agreement"]
        .fillna(False)
        .sum()
    )

    if evaluated_count > 0:

        overall_agreement = (
            agreement_count
            / evaluated_count
        )

    else:

        overall_agreement = None

    # --------------------------------------------------------
    # Agreement by predicted regime
    # --------------------------------------------------------

    regime_stats = {}

    for regime in [
        "BULL",
        "SIDE",
        "BEAR",
    ]:

        subset = evaluated[
            evaluated["next_regime"]
            == regime
        ]

        count = len(subset)

        if count == 0:

            agreement = None

        else:

            agreement = float(
                subset["agreement"]
                .fillna(False)
                .mean()
            )

        regime_stats[
            f"{regime.lower()}_predictions"
        ] = count

        regime_stats[
            f"{regime.lower()}_agreement"
        ] = agreement

    return {
        "symbol": symbol,

        "history_years": HISTORY_YEARS,

        "start_date": (
            df["prediction_date"]
            .min()
            .date()
            .isoformat()
            if not df.empty
            else None
        ),

        "latest_prediction_date": (
            df["prediction_date"]
            .max()
            .date()
            .isoformat()
            if not df.empty
            else None
        ),

        "latest_target_date": (
            df["target_trading_date"]
            .max()
            .date()
            .isoformat()
            if not df.empty
            else None
        ),

        "evaluated_predictions":
            evaluated_count,

        "pending_predictions":
            len(pending),

        "agreement_count":
            agreement_count,

        "overall_agreement":
            overall_agreement,

        **regime_stats,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("QUANTOS HMM HISTORICAL PREDICTION HISTORY")
    print("=" * 78)

    print()
    print(
        f"Source : {LATEST_FILE}"
    )

    print(
        f"Window : Last {HISTORY_YEARS} years"
    )

    print()

    predictions = load_predictions()

    if not isinstance(
        predictions,
        dict,
    ):

        raise ValueError(
            "latest_predictions.json has an "
            "unexpected structure."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_history = []

    summaries = []

    # ========================================================
    # PROCESS ALL SYMBOLS
    # ========================================================

    for symbol, payload in predictions.items():

        print("-" * 78)
        print(symbol)
        print("-" * 78)

        try:

            history = prepare_symbol_history(
                symbol,
                payload,
            )

            summary = build_summary(
                symbol,
                history,
            )

            # ------------------------------------------------
            # Save individual history
            # ------------------------------------------------

            symbol_file = (
                OUTPUT_DIR
                / (
                    "prediction_history_"
                    f"{symbol.lower()}.csv"
                )
            )

            history.to_csv(
                symbol_file,
                index=False,
            )

            print(
                f"History rows       : "
                f"{len(history):,}"
            )

            print(
                f"Prediction start   : "
                f"{history['prediction_date'].min().date()}"
            )

            print(
                f"Latest prediction  : "
                f"{history['prediction_date'].max().date()}"
            )

            latest_target = (
                history[
                    "target_trading_date"
                ]
                .max()
            )

            print(
                f"Latest target      : "
                f"{latest_target.date()}"
            )

            print(
                f"Evaluated          : "
                f"{summary['evaluated_predictions']}"
            )

            print(
                f"Pending            : "
                f"{summary['pending_predictions']}"
            )

            if (
                summary["overall_agreement"]
                is not None
            ):

                print(
                    f"Agreement          : "
                    f"{summary['overall_agreement'] * 100:.2f}%"
                )

            else:

                print(
                    "Agreement          : N/A"
                )

            print(
                f"Saved              : "
                f"{symbol_file}"
            )

            all_history.append(
                history
            )

            summaries.append(
                summary
            )

        except Exception as exc:

            print(
                f"[ERROR] {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    # ========================================================
    # COMBINED HISTORY
    # ========================================================

    if all_history:

        combined_history = pd.concat(
            all_history,
            ignore_index=True,
        )

        combined_history = (
            combined_history
            .sort_values(
                [
                    "prediction_date",
                    "symbol",
                ]
            )
            .reset_index(drop=True)
        )

        combined_file = (
            OUTPUT_DIR
            / "prediction_history.csv"
        )

        combined_history.to_csv(
            combined_file,
            index=False,
        )

        print()
        print("=" * 78)
        print("COMBINED HISTORY")
        print("=" * 78)

        print(
            f"Rows               : "
            f"{len(combined_history):,}"
        )

        print(
            f"Symbols             : "
            f"{combined_history['symbol'].nunique()}"
        )

        print(
            f"Saved               : "
            f"{combined_file}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    if summaries:

        summary_df = pd.DataFrame(
            summaries
        )

        summary_file = (
            OUTPUT_DIR
            / "prediction_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False,
        )

        print()
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)

        print(
            summary_df[
                [
                    "symbol",
                    "evaluated_predictions",
                    "pending_predictions",
                    "overall_agreement",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            f"Saved               : "
            f"{summary_file}"
        )

    print()
    print("=" * 78)
    print("HISTORICAL PREDICTION BUILD COMPLETE")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()