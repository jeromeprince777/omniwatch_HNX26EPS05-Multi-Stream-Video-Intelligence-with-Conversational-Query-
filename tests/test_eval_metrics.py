"""Unit tests for Evaluation Metrics (eval/metrics.py)."""

import pytest
from eval.metrics import (
    compute_hit_at_k,
    compute_mrr,
    compute_existence_metrics,
    compute_localization_error,
    compute_latency_stats,
    compute_interval_distance,
)


def test_interval_distance():
    # Overlapping intervals
    assert compute_interval_distance(10.0, 20.0, 15.0, 25.0) == 0.0
    # Contained interval
    assert compute_interval_distance(12.0, 18.0, 10.0, 25.0) == 0.0
    # Disjoint intervals
    assert compute_interval_distance(10.0, 20.0, 25.0, 30.0) == 5.0
    assert compute_interval_distance(30.0, 35.0, 10.0, 20.0) == 10.0


def test_hit_at_k():
    gt = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 20.0}]
    
    # Exact camera and overlap
    preds_exact = [{"camera_id": "cam1", "t_start": 12.0, "t_end": 18.0, "score": 0.8}]
    assert compute_hit_at_k(preds_exact, gt, k=1, tau=5.0) == 1.0

    # Wrong camera
    preds_wrong_cam = [{"camera_id": "cam2", "t_start": 12.0, "t_end": 18.0, "score": 0.8}]
    assert compute_hit_at_k(preds_wrong_cam, gt, k=1, tau=5.0) == 0.0

    # Distance within tau=5
    preds_within_tau = [{"camera_id": "cam1", "t_start": 23.0, "t_end": 25.0, "score": 0.8}]
    assert compute_hit_at_k(preds_within_tau, gt, k=1, tau=5.0) == 1.0

    # Distance beyond tau=5 but within tau=15
    preds_beyond_tau5 = [{"camera_id": "cam1", "t_start": 30.0, "t_end": 32.0, "score": 0.8}]
    assert compute_hit_at_k(preds_beyond_tau5, gt, k=1, tau=5.0) == 0.0
    assert compute_hit_at_k(preds_beyond_tau5, gt, k=1, tau=15.0) == 1.0

    # Hit@5 when match is in rank 3
    preds_rank3 = [
        {"camera_id": "cam2", "t_start": 0.0, "t_end": 1.0, "score": 0.9},
        {"camera_id": "cam3", "t_start": 0.0, "t_end": 1.0, "score": 0.85},
        {"camera_id": "cam1", "t_start": 12.0, "t_end": 15.0, "score": 0.8},
    ]
    assert compute_hit_at_k(preds_rank3, gt, k=1, tau=5.0) == 0.0
    assert compute_hit_at_k(preds_rank3, gt, k=5, tau=5.0) == 1.0

    # Negative query (empty GT)
    assert compute_hit_at_k(preds_exact, [], k=1, tau=5.0) == 0.0


def test_mrr():
    gt = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 20.0}]
    
    # Rank 1 match -> MRR = 1.0
    preds_r1 = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 15.0}]
    assert compute_mrr(preds_r1, gt, tau=5.0) == 1.0

    # Rank 2 match -> MRR = 0.5
    preds_r2 = [
        {"camera_id": "cam2", "t_start": 0.0, "t_end": 5.0},
        {"camera_id": "cam1", "t_start": 10.0, "t_end": 15.0},
    ]
    assert compute_mrr(preds_r2, gt, tau=5.0) == 0.5

    # No match -> MRR = 0.0
    preds_none = [{"camera_id": "cam2", "t_start": 0.0, "t_end": 5.0}]
    assert compute_mrr(preds_none, gt, tau=5.0) == 0.0


def test_existence_metrics():
    # Positive query with matching prediction
    gt_pos = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 20.0}]
    preds = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 15.0, "score": 0.55}]
    m_pos = compute_existence_metrics(preds, gt_pos, score_threshold=0.45)
    assert m_pos["correct"] == 1.0
    assert m_pos["is_false_positive"] == 0.0

    # Negative query with zero predictions -> correct
    m_neg_correct = compute_existence_metrics([], [], score_threshold=0.45)
    assert m_neg_correct["correct"] == 1.0
    assert m_neg_correct["is_false_positive"] == 0.0

    # Negative query with false positive prediction
    preds_fp = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 15.0, "score": 0.55}]
    m_neg_fp = compute_existence_metrics(preds_fp, [], score_threshold=0.45)
    assert m_neg_fp["correct"] == 0.0
    assert m_neg_fp["is_false_positive"] == 1.0


def test_localization_error():
    gt = [{"camera_id": "cam1", "t_start": 10.0, "t_end": 20.0}]
    preds = [{"camera_id": "cam1", "t_start": 25.0, "t_end": 30.0}]
    err = compute_localization_error(preds, gt)
    assert err == 5.0


def test_latency_stats():
    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_stats(lats)
    assert stats["p50"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["mean"] == 30.0
