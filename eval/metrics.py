"""Evaluation Metrics Computation for OmniWatch.

Responsibility:
- Computes Hit@1, Hit@5 with temporal tolerance tau (tau=5.0s and tau=15.0s).
- Computes Mean Reciprocal Rank (MRR).
- Computes existence accuracy and false-positive rate on negative queries.
- Computes median and p90 temporal localization error (seconds).
- Computes warm-process p50 and p95 query latencies.
- Complies strictly with docs/02_RESEARCH_AND_EVAL_PLAN.md section 5.
"""

from __future__ import annotations

import numpy as np


def compute_interval_distance(
    p_start: float,
    p_end: float,
    g_start: float,
    g_end: float,
) -> float:
    """Computes distance in seconds between intervals. 0.0 if overlapping."""
    if not (p_end < g_start or p_start > g_end):
        return 0.0
    return float(min(abs(p_start - g_end), abs(g_start - p_end)))


def is_prediction_correct(
    pred: dict,
    gt: dict,
    tau: float = 5.0,
) -> bool:
    """Checks if single prediction matches ground truth camera and temporal range within tau."""
    pred_cam = str(pred.get("camera_id", "")).strip()
    gt_cam = str(gt.get("camera_id", "")).strip()
    if pred_cam != gt_cam:
        return False

    p_start = float(pred.get("t_start", 0.0))
    p_end = float(pred.get("t_end", p_start))
    g_start = float(gt.get("t_start", 0.0))
    g_end = float(gt.get("t_end", g_start))

    dist = compute_interval_distance(p_start, p_end, g_start, g_end)
    return dist <= tau


def compute_hit_at_k(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 1,
    tau: float = 5.0,
) -> float:
    """Computes Hit@k metric requiring exact camera match and temporal overlap within tau."""
    if not ground_truths:
        # Negative query: Hit@k is 0.0
        return 0.0

    top_k = predictions[:k]
    for pred in top_k:
        for gt in ground_truths:
            if is_prediction_correct(pred, gt, tau=tau):
                return 1.0
    return 0.0


def compute_mrr(
    predictions: list[dict],
    ground_truths: list[dict],
    tau: float = 5.0,
) -> float:
    """Computes Mean Reciprocal Rank over first correct match."""
    if not ground_truths:
        return 0.0

    for rank, pred in enumerate(predictions, start=1):
        for gt in ground_truths:
            if is_prediction_correct(pred, gt, tau=tau):
                return 1.0 / float(rank)
    return 0.0


def compute_existence_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    score_threshold: float = 0.45,
) -> dict[str, float]:
    """Computes existence accuracy and false positive rate on negatives."""
    is_positive_gt = len(ground_truths) > 0
    passed_preds = [p for p in predictions if float(p.get("score", 1.0)) >= score_threshold]
    has_positive_pred = len(passed_preds) > 0

    if is_positive_gt:
        correct = 1.0 if has_positive_pred else 0.0
        is_fp = 0.0
    else:
        # Negative query
        correct = 1.0 if not has_positive_pred else 0.0
        is_fp = 1.0 if has_positive_pred else 0.0

    return {
        "correct": correct,
        "is_negative": 0.0 if is_positive_gt else 1.0,
        "is_false_positive": is_fp,
    }


def compute_localization_error(
    predictions: list[dict],
    ground_truths: list[dict],
) -> float | None:
    """Computes minimum temporal localization error (seconds) for first correct-camera hit."""
    if not ground_truths or not predictions:
        return None

    gt_cams = {str(gt.get("camera_id", "")).strip() for gt in ground_truths}
    for pred in predictions:
        pred_cam = str(pred.get("camera_id", "")).strip()
        if pred_cam in gt_cams:
            p_start = float(pred.get("t_start", 0.0))
            p_end = float(pred.get("t_end", p_start))
            # Find closest matching camera GT interval
            distances = [
                compute_interval_distance(
                    p_start,
                    p_end,
                    float(gt.get("t_start", 0.0)),
                    float(gt.get("t_end", float(gt.get("t_start", 0.0)))),
                )
                for gt in ground_truths
                if str(gt.get("camera_id", "")).strip() == pred_cam
            ]
            if distances:
                return float(min(distances))
    return None


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Computes p50, p90, p95, and mean latency percentiles from measured values."""
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
