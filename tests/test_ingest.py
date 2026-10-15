"""Acceptance Tests for Multi-Camera Video Ingestion, Decoding, and Idempotency."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from omniwatch.config import load_config
from omniwatch.db import get_connection
from omniwatch.ingest.decode import decode_video_stream, get_video_metadata
from omniwatch.ingest.manifest import load_manifest
from omniwatch.ingest.pipeline import run_ingestion


def create_synthetic_video(file_path: Path, num_frames: int = 60, fps: int = 30, width: int = 320, height: int = 240) -> Path:
    """Generates a small valid MP4 video using OpenCV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create patterned frame with moving element
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw moving circle
        cx = int((i * 5) % width)
        cy = int((i * 3) % height)
        cv2.circle(frame, (cx, cy), 20, (0, 255, 120), -1)
        writer.write(frame)

    writer.release()
    return file_path


def test_decode_timestamps(tmp_path: Path):
    """Verifies that decoded frames yield exact UTC timestamps matching start_time + t."""
    video_path = tmp_path / "test_timing.mp4"
    fps = 30
    duration_frames = 60  # 2.0 seconds
    create_synthetic_video(video_path, num_frames=duration_frames, fps=fps)

    t0 = 1700000000.0  # reference epoch
    sample_fps = 2.0  # should sample at t = 0.0s, 0.5s, 1.0s, 1.5s (approx 4 frames)

    records = list(decode_video_stream(video_path, start_time_epoch=t0, sample_fps=sample_fps))

    assert len(records) >= 4, f"Expected at least 4 sampled frames, got {len(records)}"
    for idx, rec in enumerate(records):
        expected_time = t0 + (idx * 0.5)
        # Allow +/- 0.05s tolerance for discrete frame quantization
        assert pytest.approx(rec.timestamp, abs=0.05) == expected_time
        assert rec.frame.shape == (240, 320, 3)
        assert rec.is_motion is True


def test_corrupt_file_skipped(tmp_path: Path):
    """Verifies that corrupt or truncated video files are logged and skipped without crashing."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"NOT_A_VALID_MP4_HEADER_CORRUPTED_DATA")

    # Should safely return an empty iterator
    records = list(decode_video_stream(corrupt_file, start_time_epoch=1000.0))
    assert len(records) == 0

    meta = get_video_metadata(corrupt_file)
    assert meta["native_fps"] == 0.0
    assert meta["duration"] == 0.0


def test_manifest_missing_start_time_error(tmp_path: Path):
    """Verifies missing start_time without --assume-synced fails with clear message."""
    manifest_file = tmp_path / "bad_manifest.yaml"
    data = {
        "site_id": "test_site",
        "cameras": [
            {"camera_id": "cam_unsynced", "video": "data/fake.mp4"}
        ],
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    with pytest.raises(ValueError, match="Missing required 'start_time' for camera 'cam_unsynced'"):
        load_manifest(manifest_file, assume_synced=False)

    # When assume_synced=True, it should successfully parse with default t0
    manifest = load_manifest(manifest_file, assume_synced=True)
    assert "cam_unsynced" in manifest.cameras
    assert manifest.cameras["cam_unsynced"].start_time == 0.0


def test_ingest_idempotency_and_force(tmp_path: Path):
    """Verifies camera ingestion writes rows, skips duplicate runs, and overwrites with force."""
    video1 = tmp_path / "cam1.mp4"
    video2 = tmp_path / "cam2.mp4"
    create_synthetic_video(video1, num_frames=60, fps=30)
    create_synthetic_video(video2, num_frames=60, fps=30)

    db_file = tmp_path / "test_ingest.db"
    manifest_file = tmp_path / "manifest.yaml"

    manifest_data = {
        "site_id": "site_campus",
        "cameras": [
            {
                "camera_id": "cam1",
                "label": "North Gate",
                "video": str(video1),
                "start_time": "2026-10-08T09:00:00+00:00",
            },
            {
                "camera_id": "cam2",
                "label": "Main Lobby",
                "video": str(video2),
                "start_time": "2026-10-08T09:00:10+00:00",
            },
        ],
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f)

    cfg = load_config()

    # Run 1: First ingestion
    status1 = run_ingestion(manifest_file, config=cfg, db_path=db_file, force=False)
    assert status1.completed_cameras == 2
    assert status1.camera_progress["cam1"].status == "completed"
    assert status1.camera_progress["cam2"].status == "completed"

    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id, label, native_fps, duration, thumb_path FROM cameras WHERE site_id = 'site_campus'")
    rows = cursor.fetchall()
    assert len(rows) == 2
    for r in rows:
        assert r["native_fps"] == 30.0
        assert r["duration"] > 1.9
        assert r["thumb_path"] is not None
        assert Path(r["thumb_path"]).exists()

    # Verify site table row
    site_row = conn.execute("SELECT * FROM sites WHERE site_id = 'site_campus'").fetchone()
    assert site_row is not None
    assert site_row["reference_now"] is not None
    conn.close()

    # Run 2: Re-ingest without force (Idempotency check)
    status2 = run_ingestion(manifest_file, config=cfg, db_path=db_file, force=False)
    assert status2.camera_progress["cam1"].status == "skipped"
    assert status2.camera_progress["cam2"].status == "skipped"

    # Verify no duplicate camera rows were created
    conn = get_connection(db_file)
    count = conn.execute("SELECT count(*) FROM cameras WHERE site_id = 'site_campus'").fetchone()[0]
    assert count == 2
    conn.close()

    # Run 3: Re-ingest with force=True
    status3 = run_ingestion(manifest_file, config=cfg, db_path=db_file, force=True)
    assert status3.camera_progress["cam1"].status == "completed"
    assert status3.camera_progress["cam2"].status == "completed"


def test_decode_throughput_measurement(tmp_path: Path):
    """Measures and logs exact frame decode throughput on a 90-frame video."""
    video = tmp_path / "throughput_test.mp4"
    total_frames = 90
    create_synthetic_video(video, num_frames=total_frames, fps=30, width=640, height=480)

    t_start = time.perf_counter()
    decoded_count = 0
    # Decode at high sample rate to measure pure read throughput
    for _ in decode_video_stream(video, start_time_epoch=0.0, sample_fps=30.0):
        decoded_count += 1
    t_end = time.perf_counter()

    elapsed = max(t_end - t_start, 1e-6)
    throughput_fps = decoded_count / elapsed
    latency_ms = (elapsed / max(decoded_count, 1)) * 1000.0

    print(f"\n[MEASURED TEST THROUGHPUT] Decoded {decoded_count} frames in {elapsed:.4f}s: {throughput_fps:.2f} FPS ({latency_ms:.3f} ms/frame)")
    assert decoded_count > 0
    assert throughput_fps > 50.0  # Should easily exceed 50 FPS on modern CPUs
