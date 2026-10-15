"""Ingestion Pipeline Orchestrator for Multi-Camera Video Streams.

Orchestrates multi-camera decoding, open-vocab detection, per-camera tracking,
explicit attribute extraction, SQLite observations/tracklets population,
and FAISS vector indexing.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omniwatch.config import OmniWatchConfig, load_config
from omniwatch.db import get_connection, init_db
from omniwatch.detect.detector import BaseDetector, Box, get_detector
from omniwatch.embed.attrs import associate_carries, extract_colors
from omniwatch.embed.encoder import VLEncoder
from omniwatch.index.vector import VectorIndex
from omniwatch.ingest.decode import (
    decode_video_stream,
    extract_representative_thumbnail,
    get_video_metadata,
)
from omniwatch.ingest.manifest import IngestManifest, ResolvedCamera, load_manifest
from omniwatch.track.tracker import CameraTracker
from omniwatch.track.tracklets import assemble_tracklet

logger = logging.getLogger("omniwatch.ingest.pipeline")


@dataclass
class CameraIngestProgress:
    camera_id: str
    status: str = "pending"  # "pending", "in_progress", "completed", "skipped", "failed"
    total_frames: int = 0
    processed_frames: int = 0
    detected_objects: int = 0
    tracklets_count: int = 0
    message: str | None = None


@dataclass
class IngestStatus:
    site_id: str
    total_cameras: int = 0
    completed_cameras: int = 0
    is_running: bool = False
    camera_progress: dict[str, CameraIngestProgress] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CURRENT_STATUS: IngestStatus | None = None


def get_current_ingest_status(status_file: str | Path = "data/ingest_status.json") -> dict[str, Any]:
    """Returns current ingest progress from memory or status file."""
    global _CURRENT_STATUS
    if _CURRENT_STATUS is not None:
        return _CURRENT_STATUS.to_dict()

    path = Path(status_file)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "idle"}


def _save_status(status: IngestStatus, status_file: str | Path = "data/ingest_status.json") -> None:
    path = Path(status_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status.to_dict(), f, indent=2)


def process_single_camera(
    site_id: str,
    camera: ResolvedCamera,
    config: OmniWatchConfig,
    db_path: str | Path,
    detector: BaseDetector | None = None,
    encoder: VLEncoder | None = None,
    force: bool = False,
    status_tracker: IngestStatus | None = None,
) -> dict[str, Any]:
    """Ingests a single camera stream with isolated tracker, attributes, and SQLite insertion."""
    cam_id = camera.camera_id
    logger.info("Processing camera '%s' (site '%s')", cam_id, site_id)

    if status_tracker:
        status_tracker.camera_progress[cam_id].status = "in_progress"
        _save_status(status_tracker)

    conn = get_connection(db_path)
    try:
        # Check idempotency
        row = conn.execute(
            "SELECT 1 FROM cameras WHERE site_id = ? AND camera_id = ?",
            (site_id, cam_id),
        ).fetchone()

        if row and not force:
            logger.info("Camera '%s' already ingested. Skipping (use --force to re-ingest).", cam_id)
            if status_tracker:
                status_tracker.camera_progress[cam_id].status = "skipped"
                status_tracker.camera_progress[cam_id].message = "Already exists in database"
                _save_status(status_tracker)
            return {"camera_id": cam_id, "status": "skipped", "tracklets": 0, "observations": 0}

        primary_video = camera.primary_video_path
        if not primary_video or not Path(primary_video).exists():
            logger.warning("Primary video file not found for camera '%s': %s", cam_id, primary_video)
            if status_tracker:
                status_tracker.camera_progress[cam_id].status = "failed"
                status_tracker.camera_progress[cam_id].message = f"File not found: {primary_video}"
                _save_status(status_tracker)
            return {"camera_id": cam_id, "status": "failed", "error": "file_not_found"}

        # Extract video metadata
        meta = get_video_metadata(primary_video)
        duration = meta["duration"]
        native_fps = meta["native_fps"]
        width = meta["width"]
        height = meta["height"]

        # Generate thumbnail
        thumb_dir = Path(config.storage.evidence_cache_dir) / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{site_id}_{cam_id}.jpg"
        extract_representative_thumbnail(primary_video, thumb_path)

        # Clear existing observations and tracklets for this camera if force
        if force:
            conn.execute("DELETE FROM observations WHERE site_id = ? AND camera_id = ?", (site_id, cam_id))
            conn.execute("DELETE FROM tracklets WHERE site_id = ? AND camera_id = ?", (site_id, cam_id))
            conn.commit()

        # Write camera row
        conn.execute(
            """
            INSERT OR REPLACE INTO cameras (
                site_id, camera_id, label, video_path, start_time,
                native_fps, width, height, duration, thumb_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                site_id,
                cam_id,
                camera.label,
                str(primary_video),
                camera.start_time,
                native_fps,
                width,
                height,
                duration,
                str(thumb_path) if thumb_path.exists() else None,
            ),
        )
        conn.commit()

        # Initialize detector and encoder if not provided
        active_detector = detector or get_detector(config)
        active_encoder = encoder or VLEncoder(
            model_name=config.current.encoder_name,
            pretrained=config.current.encoder_pretrained,
            device=config.resolve_device(),
        )

        sample_fps = config.current.sample_fps
        motion_gate = config.current.motion_gate
        motion_thresh = config.current.motion_threshold

        # Initialize per-camera isolated tracker
        max_gap_s = config.tracker.track_buffer / max(sample_fps, 0.1)
        tracker = CameraTracker(
            camera_id=cam_id,
            track_thresh=config.tracker.track_thresh,
            match_thresh=config.tracker.match_thresh,
            max_gap_s=max_gap_s,
            sample_fps=sample_fps,
        )

        processed_frames = 0
        total_detections = 0

        # Process frames
        for segment in camera.segments:
            seg_start = segment.start_time or camera.start_time
            for record in decode_video_stream(
                video_path=segment.video_path,
                start_time_epoch=seg_start,
                sample_fps=sample_fps,
                motion_gate=motion_gate,
                motion_threshold=motion_thresh,
            ):
                processed_frames += 1

                frame_boxes: list[Box] = []
                if record.is_motion:
                    detect_results = active_detector.detect([record.frame])
                    if detect_results:
                        frame_boxes = detect_results[0]

                total_detections += len(frame_boxes)

                    # Separate bags for carries association
                    from omniwatch.embed.attrs import associate_person_bag
                    bag_candidates = [
                        (b.bbox, b.cls, b.conf)
                        for b in frame_boxes
                        if b.cls in ("backpack", "handbag", "suitcase", "umbrella", "box")
                    ]

                    # Extract attributes and crops
                    crops_to_encode = []
                    valid_box_indices = []

                    for idx, b in enumerate(frame_boxes):
                        p_col, s_col = extract_colors(b.crop, cls=b.cls) if b.crop is not None else ("gray", "gray")
                        carries_items = associate_carries(b.bbox, [(bc[0], bc[1]) for bc in bag_candidates]) if b.cls == "person" else []
                        if b.cls == "person":
                            has_bag, b_type, b_conf = associate_person_bag(b.bbox, bag_candidates)
                            b.carrying_bag = has_bag
                            b.bag_type = b_type
                            b.bag_conf = b_conf

                        # Stash on Box
                        b.primary_color = p_col
                        b.secondary_color = s_col
                        b.carries = carries_items

                        if b.crop is not None:
                            crops_to_encode.append(b.crop)
                            valid_box_indices.append(idx)

                    # Batch encode crops
                    if crops_to_encode:
                        crop_embeddings = active_encoder.encode_images(crops_to_encode)
                        for idx_in_valid, box_idx in enumerate(valid_box_indices):
                            frame_boxes[box_idx].embedding = crop_embeddings[idx_in_valid]

                # Update camera tracker
                tracker.update(frame_boxes, record.timestamp, record.frame_idx)

                # Transfer attributes and frame reference to latest detection history in active tracks
                for track in tracker.active_tracks:
                    if track.history and track.last_seen_frame == record.frame_idx:
                        last_det = track.history[-1]
                        box = last_det.box
                        last_det.primary_color = getattr(box, "primary_color", "gray")
                        last_det.secondary_color = getattr(box, "secondary_color", "gray")
                        last_det.carries = getattr(box, "carries", [])
                        last_det.carrying_bag = getattr(box, "carrying_bag", False)
                        last_det.bag_type = getattr(box, "bag_type", None)
                        last_det.bag_conf = getattr(box, "bag_conf", 0.0)
                        last_det.embedding = getattr(box, "embedding", None)
                        last_det.frame_ref = record.frame.copy()

                if status_tracker and processed_frames % 20 == 0:
                    status_tracker.camera_progress[cam_id].processed_frames = processed_frames
                    status_tracker.camera_progress[cam_id].detected_objects = total_detections
                    _save_status(status_tracker)

        # Flush all remaining tracks
        finished_tracks = tracker.finish_all()
        logger.info("Camera '%s' tracking complete: %d tracks assembled", cam_id, len(finished_tracks))

        # Assemble and persist tracklets + observations
        vector_dir = Path(config.storage.vector_dir)
        vector_dir.mkdir(parents=True, exist_ok=True)

        obs_embeddings: list[np.ndarray] = []
        obs_ids: list[int] = []

        trk_embeddings: list[np.ndarray] = []
        trk_ids: list[int] = []

        for track in finished_tracks:
            if not track.history:
                continue

            tracklet = assemble_tracklet(track, site_id=site_id)

            # Insert tracklet first
            cursor = conn.execute(
                """
                INSERT INTO tracklets (
                    site_id, camera_id, local_track_id, cls,
                    t_start, t_end, n_obs, best_obs_id, topk_obs_json,
                    attrs_json, emb_row
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    site_id,
                    cam_id,
                    tracklet.local_track_id,
                    tracklet.cls,
                    tracklet.t_start,
                    tracklet.t_end,
                    tracklet.n_obs,
                    None,  # will update after observations
                    tracklet.topk_obs_json,
                    tracklet.attrs_json,
                    None,
                ),
            )
            tracklet_id = cursor.lastrowid
            tracklet.tracklet_id = tracklet_id

            # Insert observations for this tracklet
            best_quality = -1.0
            best_obs_id = None
            topk_obs_collected = []

            for det in track.history:
                b = det.box
                thumb_file_obs = thumb_dir / f"obs_{tracklet_id}_{det.frame_idx}.jpg"
                px1 = getattr(b, "px1", int(round(b.x1 * width)))
                py1 = getattr(b, "py1", int(round(b.y1 * height)))
                px2 = getattr(b, "px2", int(round(b.x2 * width)))
                py2 = getattr(b, "py2", int(round(b.y2 * height)))
                f_w = getattr(b, "frame_w", width)
                f_h = getattr(b, "frame_h", height)
                c_bag = 1 if getattr(det, "carrying_bag", False) or getattr(b, "carrying_bag", False) else 0
                b_type = getattr(det, "bag_type", getattr(b, "bag_type", None))
                b_conf = getattr(det, "bag_conf", getattr(b, "bag_conf", 0.0))

                # Pre-render keyframe thumbnail at index time directly on frame
                frame_data = getattr(det, "frame_ref", None)
                if frame_data is not None:
                    from omniwatch.evidence.thumbs import render_detection_thumbnail
                    render_detection_thumbnail(
                        frame_data,
                        (b.x1, b.y1, b.x2, b.y2),
                        thumb_file_obs,
                        label=f"#{tracklet_id} {b.cls} ({b.conf:.2f})",
                    )

                cursor_obs = conn.execute(
                    """
                    INSERT INTO observations (
                        site_id, camera_id, tracklet_id, frame_idx, ts,
                        x1, y1, x2, y2, px1, py1, px2, py2, frame_w, frame_h,
                        cls, det_conf, quality,
                        color_primary, color_secondary, carries,
                        carrying_bag, bag_type, bag_conf, thumb_path, emb_row
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        site_id,
                        cam_id,
                        tracklet_id,
                        det.frame_idx,
                        det.timestamp,
                        b.x1,
                        b.y1,
                        b.x2,
                        b.y2,
                        px1,
                        py1,
                        px2,
                        py2,
                        f_w,
                        f_h,
                        b.cls,
                        b.conf,
                        b.quality,
                        det.primary_color,
                        det.secondary_color,
                        json.dumps(det.carries),
                        c_bag,
                        b_type,
                        b_conf,
                        str(thumb_file_obs) if thumb_file_obs.exists() else None,
                        None,
                    ),
                )
                obs_id = cursor_obs.lastrowid
                det.obs_id = obs_id

                if b.quality > best_quality:
                    best_quality = b.quality
                    best_obs_id = obs_id
                    # Pre-render tracklet evidence thumbnail trk_{tracklet_id}.jpg
                    if thumb_file_obs.exists():
                        import shutil
                        trk_thumb_file = thumb_dir / f"trk_{tracklet_id}.jpg"
                        shutil.copyfile(str(thumb_file_obs), str(trk_thumb_file))

                topk_obs_collected.append((b.quality, obs_id))

                if det.embedding is not None:
                    obs_embeddings.append(det.embedding)
                    obs_ids.append(obs_id)

            # Update best_obs_id and topk_obs_json on tracklet
            topk_ids = [
                obs_id
                for _, obs_id in sorted(topk_obs_collected, key=lambda x: x[0], reverse=True)[:3]
            ]
            conn.execute(
                "UPDATE tracklets SET best_obs_id = ?, topk_obs_json = ? WHERE tracklet_id = ?",
                (best_obs_id, json.dumps(topk_ids), tracklet_id),
            )

            if tracklet.aggregate_embedding is not None:
                trk_embeddings.append(tracklet.aggregate_embedding)
                trk_ids.append(tracklet_id)

        conn.commit()

        # Update Vector Indexes
        obs_index = VectorIndex(dimension=512, index_name="obs")
        trk_index = VectorIndex(dimension=512, index_name="trk")

        if (vector_dir / "obs.index").exists():
            obs_index.load(vector_dir)
        if (vector_dir / "trk.index").exists():
            trk_index.load(vector_dir)

        if obs_embeddings and obs_ids:
            obs_matrix = np.vstack(obs_embeddings)
            obs_rows = obs_index.add(obs_matrix, obs_ids)
            for obs_id, emb_row in zip(obs_ids, obs_rows, strict=False):
                conn.execute("UPDATE observations SET emb_row = ? WHERE obs_id = ?", (emb_row, obs_id))
            obs_index.save(vector_dir)

        if trk_embeddings and trk_ids:
            trk_matrix = np.vstack(trk_embeddings)
            trk_rows = trk_index.add(trk_matrix, trk_ids)
            for trk_id, emb_row in zip(trk_ids, trk_rows, strict=False):
                conn.execute("UPDATE tracklets SET emb_row = ? WHERE tracklet_id = ?", (emb_row, trk_id))
            trk_index.save(vector_dir)

        conn.commit()

        if status_tracker:
            status_tracker.camera_progress[cam_id].status = "completed"
            status_tracker.camera_progress[cam_id].processed_frames = processed_frames
            status_tracker.camera_progress[cam_id].detected_objects = total_detections
            status_tracker.camera_progress[cam_id].tracklets_count = len(finished_tracks)
            _save_status(status_tracker)

        logger.info(
            "Camera '%s' ingested: %d frames, %d tracklets, %d observations",
            cam_id,
            processed_frames,
            len(finished_tracks),
            len(obs_ids),
        )
        return {
            "camera_id": cam_id,
            "status": "completed",
            "sampled_frames": processed_frames,
            "tracklets": len(finished_tracks),
            "observations": len(obs_ids),
        }

    except Exception as exc:
        logger.exception("Error processing camera '%s': %s", cam_id, exc)
        if status_tracker:
            status_tracker.camera_progress[cam_id].status = "failed"
            status_tracker.camera_progress[cam_id].message = str(exc)
            _save_status(status_tracker)
        return {"camera_id": cam_id, "status": "failed", "error": str(exc)}
    finally:
        conn.close()


def run_ingestion(
    manifest_path: str | Path,
    config: OmniWatchConfig | None = None,
    db_path: str | Path | None = None,
    assume_synced: bool = False,
    force: bool = False,
    max_workers: int = 1,  # Sequential by default for deterministic GPU inference
    detector: BaseDetector | None = None,
    encoder: VLEncoder | None = None,
) -> IngestStatus:
    """Orchestrates end-to-end ingestion pipeline from manifest.yaml."""
    global _CURRENT_STATUS

    cfg = config or load_config()
    target_db = db_path or cfg.storage.db_path

    # Initialize SQLite database
    init_db(target_db)

    # Load and validate manifest
    manifest: IngestManifest = load_manifest(manifest_path, assume_synced=assume_synced)
    site_id = manifest.site_id

    # Initialize status tracker
    status = IngestStatus(
        site_id=site_id,
        total_cameras=len(manifest.cameras),
        is_running=True,
    )
    for cam_id in manifest.cameras:
        status.camera_progress[cam_id] = CameraIngestProgress(camera_id=cam_id)
    _CURRENT_STATUS = status
    _save_status(status)

    # Initialize site row
    conn = get_connection(target_db)
    try:
        ref_now = manifest.reference_now
        now_ts = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO sites (site_id, reference_now, created_at)
            VALUES (?, ?, ?)
            """,
            (site_id, ref_now, now_ts),
        )
        conn.commit()
    finally:
        conn.close()

    # Pre-instantiate shared detector and encoder
    active_detector = detector or get_detector(cfg)
    active_encoder = encoder or VLEncoder(
        model_name=cfg.current.encoder_name,
        pretrained=cfg.current.encoder_pretrained,
        device=cfg.resolve_device(),
    )

    results = []
    if max_workers <= 1 or len(manifest.cameras) <= 1:
        for camera in manifest.cameras.values():
            res = process_single_camera(
                site_id=site_id,
                camera=camera,
                config=cfg,
                db_path=target_db,
                detector=active_detector,
                encoder=active_encoder,
                force=force,
                status_tracker=status,
            )
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cam = {
                executor.submit(
                    process_single_camera,
                    site_id,
                    cam,
                    cfg,
                    target_db,
                    active_detector,
                    active_encoder,
                    force,
                    status,
                ): cam.camera_id
                for cam in manifest.cameras.values()
            }
            for future in as_completed(future_to_cam):
                results.append(future.result())

    completed_count = sum(1 for r in results if r["status"] in ("completed", "skipped"))
    status.completed_cameras = completed_count
    status.is_running = False
    status.end_time = time.time()
    _save_status(status)

    # Update reference_now if it was not given initially
    conn = get_connection(target_db)
    try:
        cursor = conn.execute(
            "SELECT MAX(start_time + duration) AS latest_end FROM cameras WHERE site_id = ?",
            (site_id,),
        )
        row = cursor.fetchone()
        if row and row["latest_end"] is not None:
            latest_end = float(row["latest_end"])
            conn.execute(
                "UPDATE sites SET reference_now = COALESCE(reference_now, ?) WHERE site_id = ?",
                (latest_end, site_id),
            )
            conn.commit()
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print(f" OmniWatch Ingest Completed for Site '{site_id}'")
    print("=" * 60)
    for r in results:
        cam_id = r["camera_id"]
        status_str = r["status"]
        trks = r.get("tracklets", 0)
        obss = r.get("observations", 0)
        print(f" Camera '{cam_id}': status={status_str}, tracklets={trks}, observations={obss}")
    print("=" * 60)

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniWatch Video Ingest Orchestrator")
    parser.add_argument("manifest", type=str, help="Path to manifest.yaml")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Pipeline processing preset",
    )
    parser.add_argument(
        "--assume-synced",
        action="store_true",
        help="Assume unsynchronized cameras share t0 start time",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest of cameras already present in database",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel camera ingestion worker count",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Custom SQLite database path",
    )
    args = parser.parse_args()

    cfg = load_config(preset=args.preset)
    logging.basicConfig(level=getattr(logging, cfg.system.log_level, logging.INFO))

    run_ingestion(
        manifest_path=args.manifest,
        config=cfg,
        db_path=args.db,
        assume_synced=args.assume_synced,
        force=args.force,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
