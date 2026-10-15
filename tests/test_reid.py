"""Tests for cross-camera re-identification and timeline linking (Prompt 10)."""

import numpy as np
import pytest

from omniwatch.reid.reid import compute_appearance_similarity, link_cross_camera_tracklets


def test_compute_appearance_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    assert pytest.approx(compute_appearance_similarity(v1, v2)) == 1.0
    assert pytest.approx(compute_appearance_similarity(v1, v3)) == 0.0


def test_link_cross_camera_tracklets_temporal_and_appearance():
    # 3 tracklets:
    # 1: cam1 at t=10..20, person, embedding [1, 0]
    # 2: cam2 at t=25..35, person, embedding [0.99, 0.01] (should link!)
    # 3: cam1 at t=40..50, person, embedding [1, 0] (same cam as 1, but after 2, should link to 2)
    # 4: cam2 at t=5..15, overlapping time, wrong order (should not link before 1)
    tracklets = [
        {"tracklet_id": 1, "camera_id": "cam1", "cls": "person", "t_start": 10.0, "t_end": 20.0},
        {"tracklet_id": 2, "camera_id": "cam2", "cls": "person", "t_start": 25.0, "t_end": 35.0},
        {"tracklet_id": 3, "camera_id": "cam1", "cls": "person", "t_start": 40.0, "t_end": 50.0},
    ]

    embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.98, 0.05]),
        3: np.array([0.97, 0.08]),
    }

    paths = link_cross_camera_tracklets(
        tracklets,
        embeddings,
        max_transit_time_s=60.0,
        similarity_thresh=0.70,
    )

    # Should form a single connected path of 3 hops across cam1 -> cam2 -> cam1
    assert len(paths) == 1
    hop_ids = [t["tracklet_id"] for t in paths[0]]
    assert hop_ids == [1, 2, 3]


def test_link_cross_camera_tracklets_different_class_not_linked():
    tracklets = [
        {"tracklet_id": 1, "camera_id": "cam1", "cls": "person", "t_start": 10.0, "t_end": 20.0},
        {"tracklet_id": 2, "camera_id": "cam2", "cls": "car", "t_start": 25.0, "t_end": 35.0},
    ]
    embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
    }

    paths = link_cross_camera_tracklets(tracklets, embeddings, similarity_thresh=0.5)
    # Different classes must not be linked
    assert len(paths) == 2
