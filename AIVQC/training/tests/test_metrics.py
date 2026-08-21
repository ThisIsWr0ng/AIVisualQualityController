from __future__ import annotations

import unittest

from aivqc_training.metrics import evaluate_detection_metrics


class DetectionMetricsTests(unittest.TestCase):
    def test_perfect_detection_has_perfect_metrics(self) -> None:
        predictions = [
            {
                "boxes": [[10.0, 10.0, 30.0, 30.0]],
                "labels": [1],
                "scores": [0.95],
            }
        ]
        targets = [{"boxes": [[10.0, 10.0, 30.0, 30.0]], "labels": [1]}]

        result = evaluate_detection_metrics(predictions, targets, ["defect"], 0.25)

        self.assertAlmostEqual(1.0, result["map50"])
        self.assertAlmostEqual(1.0, result["map50_95"])
        self.assertAlmostEqual(1.0, result["precision"])
        self.assertAlmostEqual(1.0, result["recall"])
        self.assertAlmostEqual(1.0, result["f1"])

    def test_missing_detection_counts_false_negative(self) -> None:
        predictions = [{"boxes": [], "labels": [], "scores": []}]
        targets = [{"boxes": [[10.0, 10.0, 30.0, 30.0]], "labels": [1]}]

        result = evaluate_detection_metrics(predictions, targets, ["defect"], 0.25)

        class_result = result["per_class"]["defect"]
        self.assertEqual(0, class_result["tp"])
        self.assertEqual(1, class_result["fn"])
        self.assertAlmostEqual(0.0, result["recall"])

    def test_high_scoring_false_positive_reduces_precision(self) -> None:
        predictions = [
            {
                "boxes": [
                    [50.0, 50.0, 70.0, 70.0],
                    [10.0, 10.0, 30.0, 30.0],
                ],
                "labels": [1, 1],
                "scores": [0.99, 0.90],
            }
        ]
        targets = [{"boxes": [[10.0, 10.0, 30.0, 30.0]], "labels": [1]}]

        result = evaluate_detection_metrics(predictions, targets, ["defect"], 0.25)

        self.assertAlmostEqual(0.5, result["precision"])
        self.assertAlmostEqual(1.0, result["recall"])
        self.assertLess(result["map50"], 1.0)


if __name__ == "__main__":
    unittest.main()
