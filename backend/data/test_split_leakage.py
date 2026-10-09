import sys
from pathlib import Path

# Add backend/ to Python's import path
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text
from app.database import engine

def run_split_leakage_checks():

    print("=" * 70)
    print("SPLIT TARGET LEAKAGE CHECK")
    print("=" * 70)

    checks = {

        "TRAIN -> VALIDATION 1D": """
            SELECT COUNT(*)
            FROM dataset_train tr
            JOIN dataset_validation va
              ON tr.symbol = va.symbol
             AND va.timestamp > tr.timestamp
             AND va.timestamp <= tr.target_end_timestamp_1d;
        """,

        "TRAIN -> VALIDATION 5D": """
            SELECT COUNT(*)
            FROM dataset_train tr
            JOIN dataset_validation va
              ON tr.symbol = va.symbol
             AND va.timestamp > tr.timestamp
             AND va.timestamp <= tr.target_end_timestamp_5d;
        """,

        "TRAIN -> TEST 1D": """
            SELECT COUNT(*)
            FROM dataset_train tr
            JOIN dataset_test te
              ON tr.symbol = te.symbol
             AND te.timestamp > tr.timestamp
             AND te.timestamp <= tr.target_end_timestamp_1d;
        """,

        "TRAIN -> TEST 5D": """
            SELECT COUNT(*)
            FROM dataset_train tr
            JOIN dataset_test te
              ON tr.symbol = te.symbol
             AND te.timestamp > tr.timestamp
             AND te.timestamp <= tr.target_end_timestamp_5d;
        """,

        "VALIDATION -> TEST 1D": """
            SELECT COUNT(*)
            FROM dataset_validation va
            JOIN dataset_test te
              ON va.symbol = te.symbol
             AND te.timestamp > va.timestamp
             AND te.timestamp <= va.target_end_timestamp_1d;
        """,

        "VALIDATION -> TEST 5D": """
            SELECT COUNT(*)
            FROM dataset_validation va
            JOIN dataset_test te
              ON va.symbol = te.symbol
             AND te.timestamp > va.timestamp
             AND te.timestamp <= va.target_end_timestamp_5d;
        """
    }

    all_passed = True

    with engine.connect() as conn:

        for name, query in checks.items():

            result = conn.execute(text(query)).scalar()

            print(f"\n{name}")
            print(f"Leakage rows: {result}")

            if result != 0:
                all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("TARGET LEAKAGE CHECK PASSED")
        print("No target horizon crosses dataset boundaries.")
    else:
        print("TARGET LEAKAGE DETECTED")
        print("Dataset boundaries require correction.")

    print("=" * 70)


if __name__ == "__main__":
    run_split_leakage_checks()