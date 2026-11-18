import unittest
from datetime import date, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from mlops.serving import app as api


TEST_SNAPSHOT = {
    "model_version": "HMM_V10",
    "generated_at": "2026-10-17T12:00:00+00:00",
    "data_through": date.today().isoformat(),
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
            return_value=TEST_SNAPSHOT.copy(),
        )
        self.snapshot_patch.start()
        self.addCleanup(self.snapshot_patch.stop)

        self.client_context = TestClient(api.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_health_endpoint_reports_fresh_data(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["data_fresh"])
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

    def test_stale_data_degrades_health(self):
        stale_snapshot = TEST_SNAPSHOT.copy()
        stale_snapshot["data_through"] = (
            date.today() - timedelta(days=10)
        ).isoformat()

        with patch("mlops.serving.app._snapshot", stale_snapshot):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["data_fresh"])

    def test_stale_data_is_not_served(self):
        stale_snapshot = TEST_SNAPSHOT.copy()
        stale_snapshot["data_through"] = (
            date.today() - timedelta(days=10)
        ).isoformat()

        with patch("mlops.serving.app._snapshot", stale_snapshot):
            response = self.client.get("/v1/regime")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["error"], "stale_market_data")


if __name__ == "__main__":
    unittest.main()
