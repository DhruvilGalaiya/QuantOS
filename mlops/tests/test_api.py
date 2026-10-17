import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from mlops.serving import app as api


TEST_SNAPSHOT = {
    "model_version": "HMM_V10",
    "generated_at": "2026-10-17T12:00:00+00:00",
    "data_through": "2026-10-14",
    "symbol": "NIFTY_50",
    "summary": {"current_regime": "BEAR"},
    "next_day_probabilities": {
        "BULL": 0.12,
        "SIDE": 0.29,
        "BEAR": 0.59,
    },
    "transition_matrix": [],
}


class TestRegimeAPI(unittest.TestCase):
    def setUp(self):
        self.snapshot_patch = patch(
            "mlops.serving.app.build_snapshot",
            return_value=TEST_SNAPSHOT,
        )
        self.snapshot_patch.start()
        self.addCleanup(self.snapshot_patch.stop)

        self.client_context = TestClient(api.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["model_version"], "HMM_V10")

    def test_regime_endpoint(self):
        response = self.client.get("/v1/regime")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["symbol"], "NIFTY_50")
        self.assertEqual(body["model_version"], "HMM_V10")
        self.assertEqual(
            set(body["next_day_probabilities"]),
            {"BULL", "SIDE", "BEAR"},
        )
        self.assertAlmostEqual(
            sum(body["next_day_probabilities"].values()),
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
