
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mlops.experiments.run_hmm_experiment import (
    FEATURES,
    load_and_validate,
)


class TestHMMInputValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "features.parquet"

        n = 30
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
            "close": np.linspace(100, 110, n),
        })

        for feature in FEATURES:
            self.df[feature] = np.linspace(0.1, 1.0, n)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_and_validate(self, df):
        df.to_parquet(self.path, index=False)
        return load_and_validate(self.path)

    def test_valid_data_passes(self):
        result = self.write_and_validate(self.df)
        self.assertEqual(len(result), 30)
        self.assertIn("rule_regime", result.columns)

    def test_missing_feature_is_rejected(self):
        df = self.df.drop(columns=[FEATURES[0]])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            self.write_and_validate(df)

    def test_non_finite_feature_is_rejected(self):
        df = self.df.copy()
        df.loc[5, FEATURES[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.write_and_validate(df)

    def test_invalid_close_is_rejected(self):
        df = self.df.copy()
        df.loc[5, "close"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            self.write_and_validate(df)

    def test_duplicate_timestamps_are_removed(self):
        df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        result = self.write_and_validate(df)
        self.assertEqual(len(result), 30)
        self.assertFalse(result["timestamp"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
