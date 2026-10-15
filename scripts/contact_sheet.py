"""Contact Sheet Generator for OmniWatch Observation Verification.

Samples 30 random observations from the database, renders full-resolution frames
with exact bounding boxes drawn, and tiles them into a 5x6 contact sheet image saved
to debug/contact_sheet.jpg to visually confirm boxes sit on real people/objects.
"""

from __future__ import annotations

import logging
import math
import random
import sqlite3
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omniwatch.scripts.contact_sheet")


def generate_contact_sheet(
    db_path: str = "data/omniwatch.db",
    out_path: str = "debug/contact_sheet.jpg",
    n_samples: int = 30,
    seed: int = 42,
    cols: int = 5,
    tile_w: int = 360,
    tile_h: int = 202,
) -> dict:
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # Query observations with camera info
        cur.execute(
            """
            SELECT o.obs_id, o.tracklet_id, o.camera_id, o.frame_idx, o.ts,
                   o.x1, o.y1, o.x2, o.y2, o.cls, o.det_conf, o.quality,
                   c.video_path, c.label
            FROM observations o
            JOIN cameras c ON o.camera_id = c.camera_id
            WHERE o.x2 > o.x1 AND o.y2 > o.y1
            ORDER BY o.obs_id ASC
            """
        )
        all_rows = cur.fetchall()
        if not all_rows:
            return {"error": "No observations found in database"}

        total_obs = len(all_rows)
        sample_count = min(n_samples, total_obs)
        selected_rows = random.sample(all_rows, sample_count)
        selected_rows.sort(key=lambda r: (r["camera_id"], r["frame_idx"]))

        logger.info("Sampled %d observations out of %d total for contact sheet", sample_count, total_obs)

        # Video capture cache to avoid reopening same video 30 times
        cap_cache: dict[str, cv2.VideoCapture] = {}

        tiles = []
        for row in selected_rows:
            v_path = row["video_path"]
            frame_idx = row["frame_idx"]
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            cls_name = row["cls"]
            conf = row["det_conf"] or 0.0
            trk_id = row["tracklet_id"]

            if v_path not in cap_cache:
                cap_cache[v_path] = cv2.VideoCapture(v_path)
            cap = cap_cache[v_path]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            tile_img = None
            if ret and frame is not None:
                h, w = frame.shape[:2]
                px1 = int(round(x1 * w))
                py1 = int(round(y1 * h))
                px2 = int(round(x2 * w))
                py2 = int(round(y2 * h))

                annotated = frame.copy()
                # Draw box
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)  # Yellow
                label_txt = f"#{trk_id} {cls_name} {conf:.2f}"
                cv2.putText(
                    annotated,
                    label_txt,
                    (px1, max(20, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tile_img = cv2.resize(annotated, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            else:
                # Black placeholder with warning
                tile_img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(
                    tile_img,
                    f"Frame {frame_idx} read error",
                    (20, tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            tiles.append(tile_img)

        for cap in cap_cache.values():
            cap.release()

        # Build contact sheet grid
        rows = math.ceil(len(tiles) / cols)
        sheet_w = cols * tile_w
        sheet_h = rows * tile_h
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

        for idx, tile in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            y_start = r * tile_h
            x_start = c * tile_w
            sheet[y_start : y_start + tile_h, x_start : x_start + tile_w] = tile
            # Draw subtle grid boundary
            cv2.rectangle(sheet, (x_start, y_start), (x_start + tile_w, y_start + tile_h), (50, 50, 50), 1)

        cv2.imwrite(str(out_file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("Saved contact sheet (%dx%d) to %s", sheet_w, sheet_h, out_file)

        return {
            "status": "success",
            "samples": len(tiles),
            "output_file": str(out_file),
            "dimensions": (sheet_w, sheet_h),
            "total_observations_available": total_obs,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    res = generate_contact_sheet()
    import json
    print(json.dumps(res, indent=2))
