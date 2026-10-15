"""Unit Tests for Multi-Object Tracking, Tracklet Assembly, and FAISS Vector Indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omniwatch.detect.detector import Box
from omniwatch.index.vector import VectorIndex
from omniwatch.track.tracker import CameraTracker
from omniwatch.track.tracklets import assemble_tracklet


def test_tracker_isolated_across_cameras():
    """Verifies that tracking state and track IDs never leak across cameras."""
    tracker_cam1 = CameraTracker(camera_id="cam1", track_thresh=0.3, sample_fps=2.0)
    tracker_cam2 = CameraTracker(camera_id="cam2", track_thresh=0.3, sample_fps=2.0)

    # Box on cam1
    b1 = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.3, cls="person", conf=0.9, quality=0.8)
    tracks_cam1 = tracker_cam1.update([b1], timestamp=10.0, frame_idx=0)
    assert len(tracks_cam1) == 1
    assert tracks_cam1[0].track_id == 1
    assert tracks_cam1[0].camera_id == "cam1"

    # Box on cam2 (same position, should still start at track_id 1 on cam2)
    b2 = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.3, cls="car", conf=0.9, quality=0.85)
    tracks_cam2 = tracker_cam2.update([b2], timestamp=10.0, frame_idx=0)
    assert len(tracks_cam2) == 1
    assert tracks_cam2[0].track_id == 1
    assert tracks_cam2[0].camera_id == "cam2"

    # Advance cam1: box moves slightly
    b1_moved = Box(x1=0.12, y1=0.12, x2=0.32, y2=0.32, cls="person", conf=0.88, quality=0.8)
    tracks_cam1_step2 = tracker_cam1.update([b1_moved], timestamp=10.5, frame_idx=1)
    assert len(tracks_cam1_step2) == 1
    assert tracks_cam1_step2[0].track_id == 1
    assert len(tracks_cam1_step2[0].history) == 2

    # Check that cam2 state was completely unaffected
    assert len(tracker_cam2.active_tracks) == 1
    assert tracker_cam2.active_tracks[0].track_id == 1
    assert len(tracker_cam2.active_tracks[0].history) == 1


def test_tracklet_assembly():
    """Verifies majority vote classification, color aggregation, and quality-weighted embedding."""
    tracker = CameraTracker(camera_id="cam1")

    # Manually build a single track with 3 observations
    d1 = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.3, cls="car", conf=0.9, quality=0.8)
    d2 = Box(x1=0.11, y1=0.11, x2=0.31, y2=0.31, cls="car", conf=0.95, quality=0.9)
    d3 = Box(x1=0.12, y1=0.12, x2=0.32, y2=0.32, cls="truck", conf=0.8, quality=0.5)

    v1 = np.random.randn(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(512).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    track = tracker.update([d1], timestamp=100.0, frame_idx=0)[0]
    track.history[0].primary_color = "red"
    track.history[0].secondary_color = "black"
    track.history[0].embedding = v1
    track.history[0].obs_id = 1001

    _ = tracker.update([d2], timestamp=100.5, frame_idx=1)
    track.history[1].primary_color = "red"
    track.history[1].secondary_color = "black"
    track.history[1].embedding = v2
    track.history[1].obs_id = 1002

    _ = tracker.update([d3], timestamp=101.0, frame_idx=2)
    track.history[2].primary_color = "blue"
    track.history[2].secondary_color = "white"
    track.history[2].obs_id = 1003

    finished_tracks = tracker.finish_all()
    assert len(finished_tracks) == 1

    tracklet = assemble_tracklet(finished_tracks[0], site_id="site_alpha", top_k=2)

    # 1. Class by majority vote (car vs truck -> car)
    assert tracklet.cls == "car"
    assert tracklet.t_start == 100.0
    assert tracklet.t_end == 101.0
    assert tracklet.n_obs == 3

    # 2. Dominant color (red appears twice, blue once -> red)
    assert tracklet.attrs["color_primary"] == "red"

    # 3. Best observation (d2 had quality 0.9)
    assert tracklet.best_obs_id == 1002
    assert len(tracklet.topk_obs_ids) == 2

    # 4. Normalized aggregate embedding
    assert tracklet.aggregate_embedding is not None
    assert tracklet.aggregate_embedding.shape == (512,)
    assert pytest.approx(np.linalg.norm(tracklet.aggregate_embedding), abs=1e-4) == 1.0


def test_vector_index_add_search_filter(tmp_path: Path):
    """Verifies FAISS IndexFlatIP exact retrieval and candidate pre-filtering."""
    idx = VectorIndex(dimension=128, index_name="test_index")

    # Generate 4 random normalized vectors
    np.random.seed(42)
    raw_vecs = np.random.randn(4, 128).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norm_vecs = raw_vecs / norms
    entity_ids = [10, 20, 30, 40]

    rows = idx.add(norm_vecs, entity_ids)
    assert rows == [0, 1, 2, 3]
    assert idx.ntotal == 4

    # 1. Unfiltered search: query with vector 1 (entity_id 20)
    res_unfiltered = idx.search(norm_vecs[1], top_k=2)
    assert len(res_unfiltered) == 2
    top_id, top_score = res_unfiltered[0]
    assert top_id == 20
    assert pytest.approx(top_score, abs=1e-4) == 1.0

    # 2. Filtered search: candidate pre-filtering (allowed_ids = {10, 30})
    res_filtered = idx.search(norm_vecs[1], top_k=2, allowed_ids={10, 30})
    assert len(res_filtered) == 2
    for eid, _score in res_filtered:
        assert eid in {10, 30}


def test_vector_index_save_load_rebuild(tmp_path: Path):
    """Verifies vector index persistence and rebuildability from shards."""
    idx = VectorIndex(dimension=64, index_name="shard_test")
    vecs = np.random.randn(3, 64).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx.add(vecs, [1, 2, 3])

    idx.save(tmp_path)
    assert (tmp_path / "shard_test.index").exists()
    assert (tmp_path / "shard_test_embs.npy").exists()
    assert (tmp_path / "shard_test_id_map.json").exists()

    # Load into fresh instance
    idx_loaded = VectorIndex(dimension=64, index_name="shard_test")
    idx_loaded.load(tmp_path)
    assert idx_loaded.ntotal == 3

    # Rebuild from shards
    (tmp_path / "shard_test.index").unlink()  # delete index
    assert not (tmp_path / "shard_test.index").exists()

    idx_rebuilt = VectorIndex(dimension=64, index_name="shard_test")
    idx_rebuilt.rebuild_from_shards(tmp_path)
    assert (tmp_path / "shard_test.index").exists()
    assert idx_rebuilt.ntotal == 3

    res = idx_rebuilt.search(vecs[0], top_k=1)
    assert res[0][0] == 1
    assert pytest.approx(res[0][1], abs=1e-4) == 1.0
