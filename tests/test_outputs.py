import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineOutputTests(unittest.TestCase):
    def test_metrics_are_valid(self):
        metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
        self.assertIn(metrics["selected_model"], {"logistic_regression", "xgboost"})
        self.assertGreaterEqual(metrics["logistic_regression_auc"], 0.5)
        self.assertGreaterEqual(metrics["xgboost_auc"], 0.5)

    def test_model_and_chart_exist(self):
        self.assertGreater((ROOT / "outputs" / "model.pkl").stat().st_size, 0)
        self.assertGreater((ROOT / "outputs" / "roc_curve.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
