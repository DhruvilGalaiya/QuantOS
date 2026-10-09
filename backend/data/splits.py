"""
QuantOS temporal dataset split builder.

TRAIN: 2000-2022 historical train + 2025 recent regime
VALID: 2000-2022 historical validation + 2023-2024 recent validation
TEST:  2000-2022 historical test + 2026 final out-of-sample test

This is intentionally a mixed-regime split. 2025 training is newer than
2023-2024 validation, so target leakage is checked only in the forward
direction: source timestamp < destination timestamp <= target endpoint.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from app.database import engine

HIST_START, HIST_END = 2000, 2022
RECENT_VALIDATION_START, RECENT_VALIDATION_END = 2023, 2024
RECENT_TRAIN_YEAR, FINAL_TEST_YEAR = 2025, 2026
HIST_TRAIN_FRACTION = 0.80
HIST_VALIDATION_FRACTION = 0.10
PURGE_ROWS = 5


def load_targets():
    q = """
        SELECT timestamp, symbol, target_return_1d, target_return_5d,
               target_direction_1d, target_direction_5d,
               target_end_timestamp_1d, target_end_timestamp_5d
        FROM market_targets
        WHERE target_return_5d IS NOT NULL
        ORDER BY symbol, timestamp;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)
    for c in ["timestamp", "target_end_timestamp_1d",
              "target_end_timestamp_5d"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def create_historical_split(df):
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    n = len(df)
    if n < 100:
        raise ValueError(
            f"{df['symbol'].iloc[0]} has only {n} historical rows."
        )
    train_end = int(n * HIST_TRAIN_FRACTION)
    val_end = int(n * (HIST_TRAIN_FRACTION + HIST_VALIDATION_FRACTION))
    return (
        df.iloc[:max(0, train_end - PURGE_ROWS)].copy(),
        df.iloc[min(n, train_end + PURGE_ROWS):
                max(min(n, train_end + PURGE_ROWS), val_end - PURGE_ROWS)].copy(),
        df.iloc[min(n, val_end + PURGE_ROWS):].copy(),
    )


def add_split(df, name):
    x = df.copy()
    x["dataset_split"] = name
    return x


def purge_between(source, destination):
    """
    Remove source rows whose 1d or 5d target reaches a later destination
    observation. This is directional and therefore works with the mixed
    2025-TRAIN / 2023-2024-VALIDATION design.
    """
    if source.empty or destination.empty:
        return source.copy(), 0, 0

    dest_times = pd.Series(
        pd.to_datetime(
            destination["timestamp"].dropna().sort_values().unique()
        )
    )
    if dest_times.empty:
        return source.copy(), 0, 0

    source = source.copy()
    unsafe = set()
    c1 = c5 = 0

    for idx, row in source.iterrows():
        t = row["timestamp"]
        if pd.isna(t):
            continue
        future = dest_times > t
        if not future.any():
            continue

        for label, col in [
            ("1d", "target_end_timestamp_1d"),
            ("5d", "target_end_timestamp_5d"),
        ]:
            end = row[col]
            if pd.isna(end):
                continue
            crosses = (future & (dest_times <= end)).any()
            if crosses:
                unsafe.add(idx)
                if label == "1d":
                    c1 += 1
                else:
                    c5 += 1

    if unsafe:
        source = source.drop(index=list(unsafe)).copy()
    return source, c1, c5


def apply_purges(symbol, train, validation, test):
    print(f"\nTarget-horizon purge checks for {symbol}...")

    train, a1, a5 = purge_between(train, validation)
    print(f"  TRAIN -> VALIDATION: 1D={a1}, 5D={a5}")

    train, b1, b5 = purge_between(train, test)
    print(f"  TRAIN -> TEST:       1D={b1}, 5D={b5}")

    validation, c1, c5 = purge_between(validation, test)
    print(f"  VALIDATION -> TEST:  1D={c1}, 5D={c5}")

    return train, validation, test


def keys(df):
    return set(zip(df["symbol"], df["timestamp"]))


def build_splits():
    print("=" * 70)
    print("BUILDING LEAKAGE-AWARE TEMPORAL DATA SPLITS")
    print("=" * 70)

    df = load_targets()
    print(f"\nTotal usable rows: {len(df):,}")
    print(f"Symbols: {df['symbol'].nunique()}")

    all_train, all_validation, all_test = [], [], []

    for symbol in sorted(df["symbol"].unique()):
        print(f"\nProcessing {symbol}...")
        s = df[df["symbol"] == symbol].copy()
        s["year"] = s["timestamp"].dt.year

        historical = s[s["year"].between(HIST_START, HIST_END)].copy()
        ht, hv, htest = create_historical_split(historical)

        rv = s[s["year"].between(
            RECENT_VALIDATION_START, RECENT_VALIDATION_END
        )].copy()
        rt = s[s["year"] == RECENT_TRAIN_YEAR].copy()
        ft = s[s["year"] == FINAL_TEST_YEAR].copy()

        train = pd.concat([
            add_split(ht, "historical_train"),
            add_split(rt, "recent_train"),
        ], ignore_index=True)

        validation = pd.concat([
            add_split(hv, "historical_validation"),
            add_split(rv, "recent_validation"),
        ], ignore_index=True)

        test = pd.concat([
            add_split(htest, "historical_test"),
            add_split(ft, "final_test"),
        ], ignore_index=True)

        subset = ["symbol", "timestamp"]
        train = train.drop_duplicates(subset=subset)
        validation = validation.drop_duplicates(subset=subset)
        test = test.drop_duplicates(subset=subset)

        train, validation, test = apply_purges(
            symbol, train, validation, test
        )

        tk, vk, xk = keys(train), keys(validation), keys(test)
        if tk & vk:
            raise RuntimeError(f"{symbol}: TRAIN/VALIDATION overlap.")
        if tk & xk:
            raise RuntimeError(f"{symbol}: TRAIN/TEST overlap.")
        if vk & xk:
            raise RuntimeError(f"{symbol}: VALIDATION/TEST overlap.")

        all_train.append(train)
        all_validation.append(validation)
        all_test.append(test)

    train = pd.concat(all_train, ignore_index=True)
    validation = pd.concat(all_validation, ignore_index=True)
    test = pd.concat(all_test, ignore_index=True)

    train = train.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    validation = validation.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)
    test = test.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    print("\nWriting splits to PostgreSQL...")
    with engine.begin() as conn:
        train.to_sql("dataset_train", conn, if_exists="replace", index=False)
        validation.to_sql(
            "dataset_validation", conn, if_exists="replace", index=False
        )
        test.to_sql("dataset_test", conn, if_exists="replace", index=False)

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)
    print(f"\nTraining rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")
    return train, validation, test


def inspect_splits():
    print("\n" + "=" * 70)
    print("SPLIT DISTRIBUTION")
    print("=" * 70)

    queries = {
        "TRAIN": "dataset_train",
        "VALIDATION": "dataset_validation",
        "TEST": "dataset_test",
    }
    with engine.connect() as conn:
        for name, table in queries.items():
            q = f"""
                SELECT EXTRACT(YEAR FROM timestamp)::INT AS data_year,
                       dataset_split, COUNT(*) AS rows
                FROM {table}
                GROUP BY EXTRACT(YEAR FROM timestamp), dataset_split
                ORDER BY EXTRACT(YEAR FROM timestamp), dataset_split;
            """
            print(f"\n--- {name} ---")
            print(pd.read_sql(text(q), conn).to_string(index=False))

        q = """
            SELECT symbol, COUNT(*) AS rows
            FROM dataset_train
            GROUP BY symbol ORDER BY symbol;
        """
        print("\n--- SYMBOL_TOTALS (TRAIN) ---")
        print(pd.read_sql(text(q), conn).to_string(index=False))


def verify_no_overlap():
    print("\n" + "=" * 70)
    print("VERIFYING SPLIT ISOLATION")
    print("=" * 70)

    queries = {
        "Train / Validation": """
            SELECT COUNT(*) AS n FROM dataset_train a
            JOIN dataset_validation b
            ON a.symbol=b.symbol AND a.timestamp=b.timestamp;
        """,
        "Train / Test": """
            SELECT COUNT(*) AS n FROM dataset_train a
            JOIN dataset_test b
            ON a.symbol=b.symbol AND a.timestamp=b.timestamp;
        """,
        "Validation / Test": """
            SELECT COUNT(*) AS n FROM dataset_validation a
            JOIN dataset_test b
            ON a.symbol=b.symbol AND a.timestamp=b.timestamp;
        """,
    }

    passed = True
    with engine.connect() as conn:
        for name, q in queries.items():
            n = int(pd.read_sql(text(q), conn).iloc[0]["n"])
            print(f"{name} overlap: {n}")
            passed &= n == 0

    if not passed:
        raise RuntimeError("SPLIT ISOLATION FAILED.")
    print("\nSPLIT ISOLATION PASSED.")


def verify_target_horizon_isolation():
    print("\n" + "=" * 70)
    print("VERIFYING TARGET-HORIZON ISOLATION")
    print("=" * 70)

    with engine.connect() as conn:
        train = pd.read_sql(text("""
            SELECT symbol, timestamp, target_end_timestamp_1d,
                   target_end_timestamp_5d
            FROM dataset_train;
        """), conn)
        validation = pd.read_sql(text("""
            SELECT symbol, timestamp FROM dataset_validation;
        """), conn)
        test = pd.read_sql(text("""
            SELECT symbol, timestamp FROM dataset_test;
        """), conn)

    for d in [train, validation, test]:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    for c in ["target_end_timestamp_1d", "target_end_timestamp_5d"]:
        train[c] = pd.to_datetime(train[c], errors="coerce")

    def crossings(destination, end_col):
        total = 0
        for symbol, src in train.groupby("symbol"):
            dst = destination.loc[
                destination["symbol"] == symbol, "timestamp"
            ].dropna().sort_values()
            if dst.empty:
                continue
            for _, row in src.iterrows():
                end = row[end_col]
                if pd.isna(end):
                    continue
                if ((dst > row["timestamp"]) & (dst <= end)).any():
                    total += 1
        return total

    checks = {
        "TRAIN -> VALIDATION 1D": crossings(
            validation, "target_end_timestamp_1d"
        ),
        "TRAIN -> VALIDATION 5D": crossings(
            validation, "target_end_timestamp_5d"
        ),
        "TRAIN -> TEST 1D": crossings(
            test, "target_end_timestamp_1d"
        ),
        "TRAIN -> TEST 5D": crossings(
            test, "target_end_timestamp_5d"
        ),
    }

    for name, n in checks.items():
        print(f"{name}: {n}")

    if any(n != 0 for n in checks.values()):
        raise RuntimeError("TARGET-HORIZON LEAKAGE REMAINS.")

    print("\nTARGET-HORIZON ISOLATION PASSED.")


if __name__ == "__main__":
    build_splits()
    inspect_splits()
    verify_no_overlap()
    verify_target_horizon_isolation()