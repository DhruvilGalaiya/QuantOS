"""
QuantOS | Volatility Engine (3 underlyings)
--------------------------------------------

GARCH(1,1)-Student-t + HAR-RV baseline.

Input:
    data/regime/volatility/*.parquet

Output:
    data/regime/volatility_predictions/*.json

Important design:

    Historical OOS evaluation
        -> uses only observations with known forward targets

    Current live forecast
        -> uses the latest available feature row
        -> does NOT require future realized-volatility targets

Current volatility universe:
    NIFTY 50
    NIFTY BANK
    SENSEX
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "regime" / "volatility"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "regime"
    / "volatility_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VOLATILITY UNIVERSE
# ============================================================

FILES = {
    "NIFTY_50": "nifty_50.parquet",
    "NIFTY_BANK": "nifty_bank.parquet",
    "SENSEX": "sensex.parquet",
}


# ============================================================
# MODEL SETTINGS
# ============================================================

TRADING_DAYS = 252

MIN_TRAIN = 500

GARCH_REFIT_STEP = 21

FORECAST_HORIZON = 21


# ============================================================
# METRICS
# ============================================================

def mse(y_true, y_pred):
    """
    Mean Squared Error.

    NaN observations are ignored.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            (
                y_true[mask]
                - y_pred[mask]
            ) ** 2
        )
    )


def qlike(y_true, y_pred):
    """
    QLIKE volatility forecast loss.

    Lower is better.

    NaN observations are ignored.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.maximum(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        1e-12,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
    )

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            np.log(y_pred[mask])
            + y_true[mask]
            / y_pred[mask]
        )
    )


# ============================================================
# HAR-RV
# ============================================================

def har_forecast(
    rv_daily,
    rv_weekly,
    rv_monthly,
    train_end,
):
    """
    Fit HAR-RV using observations available before train_end.

    train_end represents the number of observations available
    for model fitting.

    The final row can therefore be used for the current forecast
    even though its future realized-volatility target does not exist.
    """

    if train_end < MIN_TRAIN:
        raise ValueError(
            "Not enough observations for HAR"
        )

    # Predict RV_t using information available at t-1.
    #
    # y:
    #   RV_1 ... RV_(train_end-1)
    #
    # X:
    #   RV_0 ... RV_(train_end-2)
    y = (
        rv_daily
        .iloc[1:train_end]
        .to_numpy(float)
    )

    X = np.column_stack([
        rv_daily
        .iloc[:train_end - 1]
        .to_numpy(float),

        rv_weekly
        .iloc[:train_end - 1]
        .to_numpy(float),

        rv_monthly
        .iloc[:train_end - 1]
        .to_numpy(float),
    ])

    mask = (
        np.isfinite(y)
        & np.isfinite(X).all(axis=1)
        & (y >= 0)
    )

    if mask.sum() < MIN_TRAIN - 10:
        raise ValueError(
            "Not enough valid observations for HAR"
        )

    model = sm.OLS(
        y[mask],
        sm.add_constant(
            X[mask],
            has_constant="add",
        ),
    ).fit()

    # Current information set.
    current = np.array([
        rv_daily.iloc[train_end - 1],
        rv_weekly.iloc[train_end - 1],
        rv_monthly.iloc[train_end - 1],
    ], dtype=float)

    if not np.isfinite(current).all():
        raise ValueError(
            "Current HAR features contain NaN/inf"
        )

    forecast = float(
        model.predict(
            sm.add_constant(
                current.reshape(1, -1),
                has_constant="add",
            )
        )[0]
    )

    return max(
        forecast,
        1e-12,
    ), model


# ============================================================
# GARCH
# ============================================================

def fit_garch_forecast(
    returns,
    train_end,
    horizon,
):
    """
    Fit GARCH(1,1)-Student-t using returns available up to
    train_end-1 and forecast the requested horizon.

    No future returns are used.
    """

    train = (
        returns
        .iloc[:train_end]
        .dropna()
        .astype(float)
    )

    if len(train) < MIN_TRAIN:
        raise ValueError(
            "Not enough observations for GARCH"
        )

    model = arch_model(
        train * 100.0,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    fitted = model.fit(
        disp="off",
        show_warning=False,
    )

    forecast = fitted.forecast(
        horizon=horizon,
        reindex=False,
    )

    variance = (
        forecast
        .variance
        .iloc[-1]
        .to_numpy(float)
        / 10000.0
    )

    return (
        np.maximum(
            variance,
            1e-12,
        ),
        fitted,
    )


# ============================================================
# OOS EVALUATION
# ============================================================

def evaluate_symbol(df):
    """
    Perform rolling out-of-sample evaluation.

    Only observations for which forward realized-volatility
    targets are actually known are used here.

    The final 21 rows have no known future targets and are
    intentionally excluded from OOS scoring.

    They remain available for live forecasting.
    """

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    n = len(df)

    records = []

    # The final 21 observations do not have known 21-day
    # forward targets, so stop before them.
    evaluation_end = (
        n - FORECAST_HORIZON
    )

    for t in range(
        MIN_TRAIN,
        evaluation_end,
        GARCH_REFIT_STEP,
    ):

        # ----------------------------------------------------
        # HAR
        # ----------------------------------------------------

        try:

            har_1d, _ = har_forecast(
                df["rv_daily"],
                df["rv_weekly"],
                df["rv_monthly"],
                t,
            )

        except Exception:

            har_1d = np.nan

        # ----------------------------------------------------
        # GARCH
        # ----------------------------------------------------

        try:

            gvars, _ = fit_garch_forecast(
                df["log_return_1d"],
                t,
                FORECAST_HORIZON,
            )

            g1 = float(
                gvars[0]
            )

            g5 = float(
                np.sum(
                    gvars[:5]
                )
            )

            g21 = float(
                np.sum(
                    gvars[:21]
                )
            )

        except Exception:

            g1 = np.nan
            g5 = np.nan
            g21 = np.nan

        # ----------------------------------------------------
        # Store OOS prediction
        # ----------------------------------------------------

        records.append({
            "timestamp": df["timestamp"].iloc[t],

            "garch_1d": g1,
            "garch_5d": g5,
            "garch_21d": g21,

            "har_1d": har_1d,

            "actual_1d":
                df["forward_rv_1d"].iloc[t],

            "actual_5d":
                df["forward_rv_5d"].iloc[t],

            "actual_21d":
                df["forward_rv_21d"].iloc[t],
        })

    fc = pd.DataFrame(
        records
    )

    metrics = {}

    if not fc.empty:

        # ----------------------------------------------------
        # GARCH metrics
        # ----------------------------------------------------

        for horizon in (
            1,
            5,
            21,
        ):

            metrics[
                f"garch_{horizon}d_mse"
            ] = mse(
                fc[
                    f"actual_{horizon}d"
                ],
                fc[
                    f"garch_{horizon}d"
                ],
            )

            metrics[
                f"garch_{horizon}d_qlike"
            ] = qlike(
                fc[
                    f"actual_{horizon}d"
                ],
                fc[
                    f"garch_{horizon}d"
                ],
            )

        # ----------------------------------------------------
        # HAR metrics
        # ----------------------------------------------------

        metrics[
            "har_1d_mse"
        ] = mse(
            fc["actual_1d"],
            fc["har_1d"],
        )

        metrics[
            "har_1d_qlike"
        ] = qlike(
            fc["actual_1d"],
            fc["har_1d"],
        )

    return metrics


# ============================================================
# VOLATILITY LEVEL
# ============================================================

def volatility_level(
    current_vol,
    long_run_vol,
):
    """
    Relative volatility classification.

    Ratio:

        current 21D vol
        ----------------
        long-run 21D vol

    Thresholds:

        < 0.75       LOW
        0.75-1.25    MEDIUM
        > 1.25       HIGH
    """

    ratio = (
        current_vol
        / max(
            long_run_vol,
            1e-12,
        )
    )

    if ratio < 0.75:

        level = "LOW"

    elif ratio <= 1.25:

        level = "MEDIUM"

    else:

        level = "HIGH"

    return level, ratio


# ============================================================
# CURRENT / LIVE FORECAST
# ============================================================

def live_forecast(df):
    """
    Produce the current volatility forecast using the latest
    available feature row.

    IMPORTANT:

    The latest row does NOT need forward targets.

    Example:

        2026-09-04
             ↓
        latest known features
             ↓
        GARCH/HAR forecast

    This is the reason the dataset builder preserves the
    final rows even though their future realized volatility
    is unknown.
    """

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    n = len(df)

    if n < MIN_TRAIN:
        raise ValueError(
            "Not enough observations for live forecast"
        )

    # --------------------------------------------------------
    # GARCH current forecast
    # --------------------------------------------------------

    gvars, _ = fit_garch_forecast(
        df["log_return_1d"],
        n,
        FORECAST_HORIZON,
    )

    # --------------------------------------------------------
    # HAR current forecast
    # --------------------------------------------------------

    har_1d, _ = har_forecast(
        df["rv_daily"],
        df["rv_weekly"],
        df["rv_monthly"],
        n,
    )

    # --------------------------------------------------------
    # Current realized volatility
    # --------------------------------------------------------

    current_rv = float(
        df["rv_21d"].iloc[-1]
    )

    if not np.isfinite(current_rv):
        raise ValueError(
            "Latest 21D realized variance is invalid"
        )

    current_vol = float(
        np.sqrt(
            current_rv
            * TRADING_DAYS
            / 21
        )
    )

    # --------------------------------------------------------
    # Long-run volatility reference
    # --------------------------------------------------------

    long_run_rv = float(
        df["rv_21d"].median()
    )

    long_run_vol = float(
        np.sqrt(
            max(
                long_run_rv,
                1e-12,
            )
            * TRADING_DAYS
            / 21
        )
    )

    # --------------------------------------------------------
    # Volatility level
    # --------------------------------------------------------

    level, ratio = volatility_level(
        current_vol,
        long_run_vol,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "latest_date":
            df["timestamp"]
            .iloc[-1]
            .isoformat(),

        "latest_close":
            float(
                df["close"].iloc[-1]
            ),

        "current_realized_vol_21d":
            current_vol,

        "long_run_vol_21d":
            long_run_vol,

        "volatility_ratio":
            ratio,

        "volatility_level":
            level,

        "garch": {
            "model":
                "GARCH(1,1)-Student-t",

            "forecast_vol_1d":
                float(
                    np.sqrt(
                        gvars[0]
                        * TRADING_DAYS
                    )
                ),

            "forecast_vol_5d":
                float(
                    np.sqrt(
                        np.sum(
                            gvars[:5]
                        )
                        * TRADING_DAYS
                        / 5
                    )
                ),

            "forecast_vol_21d":
                float(
                    np.sqrt(
                        np.sum(
                            gvars[:21]
                        )
                        * TRADING_DAYS
                        / 21
                    )
                ),
        },

        "har": {
            "model":
                "HAR-RV",

            "forecast_vol_1d":
                float(
                    np.sqrt(
                        har_1d
                        * TRADING_DAYS
                    )
                ),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "QuantOS VOLATILITY ENGINE | "
        "NIFTY / BANK NIFTY / SENSEX"
    )

    print("=" * 70)

    print(
        f"\nInput directory:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    results = {}

    success = 0

    # ========================================================
    # PROCESS EACH SYMBOL
    # ========================================================

    for symbol, filename in FILES.items():

        print("\n" + "-" * 70)
        print(symbol)
        print("-" * 70)

        try:

            input_path = (
                INPUT_DIR / filename
            )

            if not input_path.exists():

                raise FileNotFoundError(
                    f"Input file not found: "
                    f"{input_path}"
                )

            # ------------------------------------------------
            # Load dataset
            # ------------------------------------------------

            df = pd.read_parquet(
                input_path
            )

            # ------------------------------------------------
            # Validate required columns
            # ------------------------------------------------

            required_columns = [
                "timestamp",
                "close",
                "log_return_1d",
                "rv_1d",
                "rv_5d",
                "rv_21d",
                "rv_daily",
                "rv_weekly",
                "rv_monthly",
                "forward_rv_1d",
                "forward_rv_5d",
                "forward_rv_21d",
            ]

            missing = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing:

                raise ValueError(
                    f"Missing required columns: "
                    f"{missing}"
                )

            # ------------------------------------------------
            # Sort chronologically
            # ------------------------------------------------

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

            df = (
                df
                .dropna(
                    subset=[
                        "timestamp",
                        "close",
                    ]
                )
                .sort_values("timestamp")
                .drop_duplicates(
                    "timestamp",
                    keep="last",
                )
                .reset_index(drop=True)
            )

            if df.empty:

                raise ValueError(
                    "Dataset is empty"
                )

            # ------------------------------------------------
            # Print dataset boundary
            # ------------------------------------------------

            print(
                f"Dataset period: "
                f"{df['timestamp'].min()} -> "
                f"{df['timestamp'].max()}"
            )

            print(
                f"Dataset rows: "
                f"{len(df):,}"
            )

            # ------------------------------------------------
            # OOS evaluation
            # ------------------------------------------------

            metrics = evaluate_symbol(
                df
            )

            # ------------------------------------------------
            # Current forecast
            # ------------------------------------------------

            live = live_forecast(
                df
            )

            # ------------------------------------------------
            # Construct output
            # ------------------------------------------------

            payload = {
                "symbol":
                    symbol,

                "model_version":
                    "volatility_v3_3_underlyings",

                "data": {
                    "input_file":
                        str(input_path),

                    "dataset_start":
                        df["timestamp"]
                        .iloc[0]
                        .isoformat(),

                    "dataset_end":
                        df["timestamp"]
                        .iloc[-1]
                        .isoformat(),

                    "rows":
                        int(len(df)),
                },

                "evaluation":
                    metrics,

                "live":
                    live,
            }

            # ------------------------------------------------
            # Save individual JSON
            # ------------------------------------------------

            output = (
                OUTPUT_DIR
                / f"{symbol.lower()}.json"
            )

            output.write_text(
                json.dumps(
                    payload,
                    indent=2,
                ),
                encoding="utf-8",
            )

            results[symbol] = payload

            success += 1

            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            print(
                f"Latest date: "
                f"{live['latest_date']}"
            )

            print(
                f"Latest close: "
                f"{live['latest_close']:,.2f}"
            )

            print(
                f"Current 21D realized vol: "
                f"{live['current_realized_vol_21d']:.2%}"
            )

            print(
                f"Long-run 21D vol: "
                f"{live['long_run_vol_21d']:.2%}"
            )

            print(
                f"Volatility ratio: "
                f"{live['volatility_ratio']:.3f}"
            )

            print(
                f"Volatility level: "
                f"{live['volatility_level']}"
            )

            print(
                f"GARCH 1D: "
                f"{live['garch']['forecast_vol_1d']:.2%}"
            )

            print(
                f"GARCH 5D: "
                f"{live['garch']['forecast_vol_5d']:.2%}"
            )

            print(
                f"GARCH 21D: "
                f"{live['garch']['forecast_vol_21d']:.2%}"
            )

            print(
                f"HAR 1D: "
                f"{live['har']['forecast_vol_1d']:.2%}"
            )

            # ------------------------------------------------
            # OOS metrics
            # ------------------------------------------------

            print("\nOOS metrics:")

            for key, value in metrics.items():

                if pd.isna(value):

                    print(
                        f"  {key:<24} NaN"
                    )

                else:

                    print(
                        f"  {key:<24} "
                        f"{value:.8g}"
                    )

            print(
                f"\nSaved: {output}"
            )

        except Exception as exc:

            print(
                f"ERROR: {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    # ========================================================
    # COMBINED OUTPUT
    # ========================================================

    combined = (
        OUTPUT_DIR
        / "latest_volatility.json"
    )

    combined.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "VOLATILITY ENGINE COMPLETE"
    )

    print("=" * 70)

    print(
        f"Successful symbols: "
        f"{success}/{len(FILES)}"
    )

    print(
        f"Saved: {combined}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()