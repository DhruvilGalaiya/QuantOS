
"""
QuantOS | LIVE HMM V10 - FINAL WORKING RUNNER

Runs the existing production V10 HMM and writes a complete JSON artifact
for Streamlit.

Important:
- Does NOT change the V10 model.
- Adds the V10-required reference `rule_regime` before inference.
- Correctly persists:
    * current regime
    * next-day probabilities
    * current streak
    * stickiness
    * 3x3 transition matrix
    * long-run/stationary distribution
    * daily history
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import quantos_hmm_v10 as v10


# ============================================================
# CONFIG
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

FEATURE_DIR = PROJECT_ROOT / "data" / "regime" / "features"
OUTPUT_DIR = PROJECT_ROOT / "data" / "regime" / "live_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE = 1.85

REFERENCE_LOOKBACK = 20
REFERENCE_THRESHOLD = 0.05

SYMBOL_FILES = {
    "NIFTY_50": "nifty_50.parquet",
    "NIFTY_BANK": "nifty_bank.parquet",
    "NIFTY_IT": "nifty_it.parquet",
    "NIFTY_PHARMA": "nifty_pharma.parquet",
    "NIFTY_AUTO": "nifty_auto.parquet",
    "NIFTY_FIN_SERVICE": "nifty_fin_service.parquet",
    "INDIA_VIX": "india_vix.parquet",
    "SENSEX": "sensex.parquet",
}


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(value: Any):
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return {
            "index": [json_safe(x) for x in value.index.tolist()],
            "columns": [str(x) for x in value.columns.tolist()],
            "data": [
                [json_safe(x) for x in row]
                for row in value.to_numpy(dtype=object).tolist()
            ],
        }

    if isinstance(value, pd.Series):
        return [json_safe(x) for x in value.tolist()]

    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def dataframe_records(df: pd.DataFrame | None):
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    if df.empty:
        return []

    clean = df.replace([np.inf, -np.inf], np.nan)
    clean = clean.astype(object).where(pd.notna(clean), None)

    return json_safe(clean.to_dict(orient="records"))


# ============================================================
# REFERENCE REGIME
# ============================================================

def add_rule_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = ["timestamp", "close"]
    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for rule_regime: {missing}"
        )

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
    )

    out = (
        out.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    rolling_log_return = np.log(
        out["close"] / out["close"].shift(REFERENCE_LOOKBACK)
    )

    # V10 convention:
    # 0 = BULL, 1 = SIDE, 2 = BEAR
    out["rule_regime"] = 1

    out.loc[
        rolling_log_return > REFERENCE_THRESHOLD,
        "rule_regime",
    ] = 0

    out.loc[
        rolling_log_return < -REFERENCE_THRESHOLD,
        "rule_regime",
    ] = 2

    return out


# ============================================================
# EXTRACTION HELPERS
# ============================================================

def find_column(df: pd.DataFrame, candidates: list[str]):
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def probability_dict(df: pd.DataFrame | None):
    if df is None or df.empty:
        return {}

    regime_col = find_column(
        df,
        ["regime", "state", "next_regime"],
    )

    probability_col = find_column(
        df,
        [
            "next_day_probability",
            "next_probability",
            "probability",
            "forecast_probability",
        ],
    )

    if regime_col is None or probability_col is None:
        return {}

    result = {}

    for _, row in df.iterrows():
        regime = str(row[regime_col]).upper()

        try:
            value = float(row[probability_col])
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            result[regime] = max(0.0, value)

    total = sum(result.values())

    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result


def long_matrix_to_wide(matrix_df: pd.DataFrame | None):
    """
    V10 returns matrix_df in long format:

        current_regime | next_regime | probability

    Convert it to the actual 3x3 matrix used by the dashboard.
    """

    if matrix_df is None or matrix_df.empty:
        return None

    current_col = find_column(
        matrix_df,
        ["current_regime", "current_state"],
    )

    next_col = find_column(
        matrix_df,
        ["next_regime", "next_state"],
    )

    probability_col = find_column(
        matrix_df,
        ["probability", "transition_probability"],
    )

    if (
        current_col is None
        or next_col is None
        or probability_col is None
    ):
        return None

    states = ["BULL", "SIDE", "BEAR"]

    wide = pd.DataFrame(
        0.0,
        index=states,
        columns=states,
    )

    for _, row in matrix_df.iterrows():
        current = str(row[current_col]).upper()
        nxt = str(row[next_col]).upper()

        try:
            probability = float(row[probability_col])
        except (TypeError, ValueError):
            continue

        if (
            current in states
            and nxt in states
            and np.isfinite(probability)
        ):
            wide.loc[current, nxt] = probability

    return wide


def stationary_dict(stationary_df: pd.DataFrame | None):
    if stationary_df is None or stationary_df.empty:
        return {}

    regime_col = find_column(
        stationary_df,
        ["regime", "state", "label"],
    )

    probability_col = find_column(
        stationary_df,
        [
            "long_run_probability",
            "stationary_probability",
            "probability",
            "share",
        ],
    )

    if regime_col is None or probability_col is None:
        return {}

    result = {}

    for _, row in stationary_df.iterrows():
        regime = str(row[regime_col]).upper()

        try:
            value = float(row[probability_col])
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            result[regime] = max(0.0, value)

    total = sum(result.values())

    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result


# ============================================================
# BUILD ARTIFACT
# ============================================================

def build_payload(
    summary,
    probability_df,
    matrix_df,
    stationary_df,
    daily_df,
    P_hmm,
    P_reference,
    restarts,
):
    payload = {}

    # V10 summary is authoritative for scalar values.
    if isinstance(summary, dict):
        payload["summary"] = json_safe(summary)
        payload.update(json_safe(summary))
    else:
        payload["summary"] = json_safe(summary)

    # Probabilities.
    next_probs = probability_dict(probability_df)

    if next_probs:
        payload["probabilities"] = next_probs
        payload["next_day_probability"] = max(next_probs.values())

    # Transition matrix.
    wide_matrix = long_matrix_to_wide(matrix_df)

    if wide_matrix is not None:
        payload["transition_matrix"] = json_safe(wide_matrix)
        payload["matrix_output"] = dataframe_records(matrix_df)

    # Long-run distribution.
    stationary = stationary_dict(stationary_df)

    if stationary:
        payload["stationary_distribution"] = stationary
        payload["stationary_output"] = dataframe_records(stationary_df)

    # Daily history.
    if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
        payload["daily_output"] = dataframe_records(
            daily_df.tail(500)
        )

        latest = daily_df.iloc[-1]

        # V10 uses current_regime_streak, not current_streak.
        streak_col = find_column(
            daily_df,
            [
                "current_regime_streak",
                "current_streak",
                "current_streak_days",
                "streak",
                "regime_streak",
            ],
        )

        if streak_col is not None:
            try:
                streak = int(round(float(latest[streak_col])))
                payload["current_regime_streak"] = streak
                payload["current_streak"] = streak
            except (TypeError, ValueError):
                pass

        # Scalar values from the daily row, as a second source.
        for field, candidates in {
            "current_regime": [
                "current_regime",
                "regime",
                "state_regime",
            ],
            "next_regime": [
                "next_regime",
                "forecast_regime",
                "next_state",
            ],
            "stickiness_score": [
                "stickiness_score",
                "stickiness",
                "score",
            ],
            "recent_consistency": [
                "recent_consistency",
                "consistency",
            ],
            "stay_probability": [
                "stay_probability",
            ],
            "next_day_probability": [
                "next_day_probability",
            ],
            "date": [
                "date",
                "timestamp",
            ],
        }.items():
            column = find_column(
                daily_df,
                candidates,
            )

            if column is not None:
                value = latest[column]

                if field in [
                    "stickiness_score",
                    "recent_consistency",
                    "stay_probability",
                    "next_day_probability",
                ]:
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue

                    if not np.isfinite(value):
                        continue

                payload[field] = json_safe(value)

    # Preserve diagnostic information.
    payload["P_hmm"] = json_safe(P_hmm)
    payload["P_reference"] = json_safe(P_reference)
    payload["restarts"] = json_safe(restarts)

    # Explicit aliases used by dashboard.
    if "current_regime_streak" in payload:
        payload["current_streak"] = payload[
            "current_regime_streak"
        ]

    if "stickiness_score" in payload:
        payload["stickiness"] = payload[
            "stickiness_score"
        ]

    return payload


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("QuantOS LIVE HMM V10")
    print("=" * 70)
    print(
        f"Reference regime: {REFERENCE_LOOKBACK}-session "
        f"log return, threshold ±{REFERENCE_THRESHOLD:.0%}"
    )

    combined = {}
    success = 0

    for symbol, filename in SYMBOL_FILES.items():
        print("\n" + "-" * 70)
        print(symbol)
        print("-" * 70)

        feature_file = FEATURE_DIR / filename

        try:
            if not feature_file.exists():
                raise FileNotFoundError(
                    f"Feature file not found: {feature_file}"
                )

            df = pd.read_parquet(feature_file)
            df = add_rule_regime(df)

            required = [
                "timestamp",
                "close",
                "rule_regime",
                "log_return_1d",
                "volatility_20",
                "rsi_14",
                "volume_ratio",
                "atr_14",
                "close_vs_sma20",
                "close_vs_sma50",
            ]

            missing = [
                c for c in required
                if c not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing required V10 columns: {missing}"
                )

            (
                summary,
                probability_df,
                matrix_df,
                stationary_df,
                daily_df,
                P_hmm,
                P_reference,
                restarts,
            ) = v10.production_output(
                df,
                TEMPERATURE,
            )

            payload = build_payload(
                summary,
                probability_df,
                matrix_df,
                stationary_df,
                daily_df,
                P_hmm,
                P_reference,
                restarts,
            )

            # Metadata fallback.
            payload.setdefault(
                "latest_date",
                df["timestamp"].iloc[-1].date().isoformat(),
            )

            payload.setdefault(
                "latest_close",
                float(df["close"].iloc[-1]),
            )

            # Save individual.
            individual_file = (
                OUTPUT_DIR
                / f"{symbol.lower()}.json"
            )

            with individual_file.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            combined[symbol] = payload
            success += 1

            current = payload.get(
                "current_regime",
                "N/A",
            )

            nxt = payload.get(
                "next_regime",
                "N/A",
            )

            next_p = payload.get(
                "next_day_probability",
                np.nan,
            )

            stick = payload.get(
                "stickiness_score",
                np.nan,
            )

            streak = payload.get(
                "current_regime_streak",
                0,
            )

            print(
                f"Rows:        {len(df):,}"
            )
            print(
                f"Latest date: {payload.get('latest_date', 'N/A')}"
            )
            print(
                f"Latest close: {payload['latest_close']:,.2f}"
            )
            print(
                f"\nCurrent regime: {current}"
            )
            print(
                f"Next regime:    {nxt}"
            )

            try:
                print(
                    f"Next probability: "
                    f"{float(next_p) * 100:.2f}%"
                )
            except (TypeError, ValueError):
                pass

            print("\nNext-day probabilities:")

            for regime, value in payload.get(
                "probabilities",
                {},
            ).items():
                print(
                    f"  {regime:<10} "
                    f"{float(value) * 100:6.2f}%"
                )

            try:
                print(
                    f"\nStickiness: {float(stick):.1f}/100 "
                    f"({payload.get('stickiness_level', 'N/A')})"
                )
            except (TypeError, ValueError):
                pass

            print(
                f"Current streak: {streak}"
            )

            print(
                "Transition matrix: "
                f"{'SAVED' if 'transition_matrix' in payload else 'MISSING'}"
            )

            print(
                "Long-run mix:     "
                f"{'SAVED' if 'stationary_distribution' in payload else 'MISSING'}"
            )

            print(
                f"\nSaved: {individual_file}"
            )

        except Exception as exc:
            print(
                f"ERROR: {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    combined_file = (
        OUTPUT_DIR
        / "latest_predictions.json"
    )

    with combined_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            combined,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("LIVE HMM COMPLETE")
    print("=" * 70)
    print(
        f"Successful symbols: "
        f"{success}/{len(SYMBOL_FILES)}"
    )
    print(
        f"Saved: {combined_file}"
    )


if __name__ == "__main__":
    main()
