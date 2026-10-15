"""Debug script to inspect bounding box and frame reference for tracklet #338.

Loads stored tracklet #338 metadata and observation from SQLite,
fetches the original full-resolution frame via stored frame reference,
draws the bounding box using plain cv2.rectangle, and saves to debug/bbox_338.jpg.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omniwatch.scripts.debug_bbox")


def run_debug_bbox(db_path: str = "data/omniwatch.db", target_tracklet_id: int = 338, out_path: str = "debug/bbox_338.jpg") -> dict:
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # Query tracklet 338 and associated best observation
        cur.execute(
            """
            SELECT t.tracklet_id, t.site_id, t.camera_id, t.cls, t.t_start, t.t_end,
                   t.n_obs, t.best_obs_id, t.attrs_json,
                   c.video_path, c.native_fps, c.width, c.height,
                   o.obs_id, o.frame_idx, o.ts, o.x1, o.y1, o.x2, o.y2, o.det_conf, o.quality
            FROM tracklets t
            JOIN cameras c ON t.camera_id = c.camera_id
            LEFT JOIN observations o ON t.best_obs_id = o.obs_id
            WHERE t.tracklet_id = ?
            """,
            (target_tracklet_id,),
        )
        row = cur.fetchone()
        if not row:
            # If 338 doesn't exist, find all tracklets count and the highest or sample one
            cur.execute("SELECT tracklet_id, camera_id, cls FROM tracklets ORDER BY tracklet_id ASC LIMIT 10")
            samples = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) FROM tracklets")
            total_count = cur.fetchone()[0]
            return {
                "error": f"Tracklet #{target_tracklet_id} not found in database",
                "total_tracklets": total_count,
                "sample_tracklets": samples,
            }

        data = dict(row)
        video_path = data["video_path"]
        frame_idx = data["frame_idx"]
        native_fps = data["native_fps"] or 30.0
        x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]

        logger.info(
            "Tracklet #%d: camera=%s, cls=%s, best_obs_id=%s, stored frame_idx=%s, coords=(%.3f, %.3f, %.3f, %.3f)",
            target_tracklet_id, data["camera_id"], data["cls"], data["best_obs_id"], frame_idx, x1, y1, x2, y2
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video file {video_path}", "data": data}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # 1. Read frame using stored frame_idx via CAP_PROP_POS_FRAMES
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {
                "error": f"Failed to read frame at index {frame_idx} from {video_path}",
                "total_frames": total_frames,
                "data": data,
            }

        h, w = frame.shape[:2]
        px1 = int(round(x1 * w))
        py1 = int(round(y1 * h))
        px2 = int(round(x2 * w))
        py2 = int(round(y2 * h))

        annotated = frame.copy()
        # Draw plain cv2.rectangle as requested
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)  # Red box in BGR
        cv2.putText(
            annotated,
            f"Trk #{target_tracklet_id} {data['cls']} (stored frame_idx={frame_idx})",
            (px1, max(30, py1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(out_file), annotated)
        logger.info("Saved annotated frame to %s", out_file)

        # Also let's compute what the native frame index SHOULD have been if frame_idx was sampled at 2 FPS!
        expected_native_frame = int(round((data["ts"] - row["start_time"] if "start_time" in row.keys() else 0) * native_fps))
        
        return {
            "tracklet_id": target_tracklet_id,
            "camera_id": data["camera_id"],
            "video_path": video_path,
            "total_frames_in_video": total_frames,
            "orig_dimensions": (orig_w, orig_h),
            "stored_frame_idx": frame_idx,
            "stored_timestamp": data["ts"],
            "normalized_bbox": (x1, y1, x2, y2),
            "pixel_bbox": (px1, py1, px2, py2),
            "output_image": str(out_file),
            "output_exists": out_file.exists(),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    result = run_debug_bbox()
    print(json.dumps(result, indent=2))
