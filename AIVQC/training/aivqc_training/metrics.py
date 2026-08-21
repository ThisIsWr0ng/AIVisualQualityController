"""Dependency-light object-detection metrics used by the training backend."""

from __future__ import annotations

from collections import defaultdict
from math import isnan
from typing import Any


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0

    recall_curve = [0.0, *recalls, 1.0]
    precision_curve = [1.0, *precisions, 0.0]
    for index in range(len(precision_curve) - 2, -1, -1):
        precision_curve[index] = max(precision_curve[index], precision_curve[index + 1])

    return sum(
        (recall_curve[index] - recall_curve[index - 1]) * precision_curve[index]
        for index in range(1, len(recall_curve))
        if recall_curve[index] != recall_curve[index - 1]
    )


def _evaluate_class(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    label: int,
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, float | int]:
    ground_truth: dict[int, list[list[float]]] = defaultdict(list)
    detections: list[tuple[float, int, list[float]]] = []

    for image_index, target in enumerate(targets):
        for box, target_label in zip(target["boxes"], target["labels"]):
            if target_label == label:
                ground_truth[image_index].append(box)

    for image_index, prediction in enumerate(predictions):
        for box, predicted_label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if predicted_label == label and score >= score_threshold:
                detections.append((score, image_index, box))

    detections.sort(key=lambda item: item[0], reverse=True)
    matched = {index: [False] * len(boxes) for index, boxes in ground_truth.items()}
    true_positives: list[int] = []
    false_positives: list[int] = []

    for _, image_index, detected_box in detections:
        candidates = ground_truth.get(image_index, [])
        overlaps = [_iou(detected_box, box) for box in candidates]
        best_index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1

        if (
            best_index >= 0
            and overlaps[best_index] >= iou_threshold
            and not matched[image_index][best_index]
        ):
            matched[image_index][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    positive_count = sum(len(boxes) for boxes in ground_truth.values())
    cumulative_true = 0
    cumulative_false = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for true_positive, false_positive in zip(true_positives, false_positives):
        cumulative_true += true_positive
        cumulative_false += false_positive
        recalls.append(cumulative_true / positive_count if positive_count else 0.0)
        denominator = cumulative_true + cumulative_false
        precisions.append(cumulative_true / denominator if denominator else 0.0)

    false_negative_count = max(0, positive_count - cumulative_true)
    precision = precisions[-1] if precisions else 0.0
    recall = recalls[-1] if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    average_precision = (
        _average_precision(recalls, precisions) if positive_count else float("nan")
    )
    return {
        "ap": average_precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": cumulative_true,
        "fp": cumulative_false,
        "fn": false_negative_count,
        "ground_truth_count": positive_count,
    }


def evaluate_detection_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    class_names: list[str],
    score_threshold: float,
) -> dict[str, Any]:
    """Calculate mAP50, mAP50-95 and per-class precision/recall/F1."""

    iou_thresholds = [value / 100 for value in range(50, 100, 5)]
    per_class: dict[str, dict[str, float | int]] = {}
    all_average_precisions: list[float] = []
    map50_values: list[float] = []

    for label, class_name in enumerate(class_names, start=1):
        threshold_results = [
            _evaluate_class(
                predictions,
                targets,
                label,
                iou_threshold,
                score_threshold,
            )
            for iou_threshold in iou_thresholds
        ]
        primary = threshold_results[0]
        valid_average_precisions = [
            float(result["ap"])
            for result in threshold_results
            if not isnan(float(result["ap"]))
        ]
        if valid_average_precisions:
            all_average_precisions.extend(valid_average_precisions)
            map50_values.append(float(primary["ap"]))

        per_class[class_name] = {
            "precision": primary["precision"],
            "recall": primary["recall"],
            "f1": primary["f1"],
            "ap50": 0.0 if isnan(float(primary["ap"])) else primary["ap"],
            "map50_95": (
                sum(valid_average_precisions) / len(valid_average_precisions)
                if valid_average_precisions
                else 0.0
            ),
            "tp": primary["tp"],
            "fp": primary["fp"],
            "fn": primary["fn"],
            "ground_truth_count": primary["ground_truth_count"],
        }

    precision_values = [float(item["precision"]) for item in per_class.values()]
    recall_values = [float(item["recall"]) for item in per_class.values()]
    f1_values = [float(item["f1"]) for item in per_class.values()]
    return {
        "map50": sum(map50_values) / len(map50_values) if map50_values else 0.0,
        "map50_95": (
            sum(all_average_precisions) / len(all_average_precisions)
            if all_average_precisions
            else 0.0
        ),
        "precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "score_threshold": score_threshold,
        "per_class": per_class,
    }
