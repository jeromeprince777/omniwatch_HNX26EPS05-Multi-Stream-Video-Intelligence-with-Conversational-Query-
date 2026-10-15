"""Thumbnail Generation and Bounding Box Rendering Module.

Responsibility:
- Renders bounding boxes on full camera frames for evidence cards.
- Crops object bounding boxes with optional padding for gallery view and VLM verification.
- Saves generated thumbnails with deterministic filenames in evidence cache.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def crop_detection(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad: float = 0.05,
) -> np.ndarray:
    """Extracts object crop with optional margin padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_w = (x2 - x1) * pad
    pad_h = (y2 - y1) * pad
    cx1 = max(0, int((x1 - pad_w) * w))
    cy1 = max(0, int((y1 - pad_h) * h))
    cx2 = min(w, int((x2 + pad_w) * w))
    cy2 = min(h, int((y2 + pad_h) * h))

    if cx2 <= cx1 or cy2 <= cy1:
        return frame
    return frame[cy1:cy2, cx1:cx2]


def render_detection_thumbnail(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    out_path: str | Path,
    label: str | None = None,
) -> Path:
    """Draws normalized bounding box on frame with tactical HUD styling and saves."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    x1, y1, x2, y2 = bbox
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)

    # Cyber Cyan box in BGR: (255, 240, 0)
    box_color = (255, 240, 0)
    cv2.rectangle(annotated, (px1, py1), (px2, py2), box_color, 2)

    # Corner brackets (tactical HUD style)
    corner_len = max(8, int(min(px2 - px1, py2 - py1) * 0.2))
    cv2.line(annotated, (px1, py1), (px1 + corner_len, py1), (0, 255, 255), 3)
    cv2.line(annotated, (px1, py1), (px1, py1 + corner_len), (0, 255, 255), 3)
    cv2.line(annotated, (px2, py1), (px2 - corner_len, py1), (0, 255, 255), 3)
    cv2.line(annotated, (px2, py1), (px2, py1 + corner_len), (0, 255, 255), 3)
    cv2.line(annotated, (px1, py2), (px1 + corner_len, py2), (0, 255, 255), 3)
    cv2.line(annotated, (px1, py2), (px1, py2 - corner_len), (0, 255, 255), 3)
    cv2.line(annotated, (px2, py2), (px2 - corner_len, py2), (0, 255, 255), 3)
    cv2.line(annotated, (px2, py2), (px2, py2 - corner_len), (0, 255, 255), 3)

    if label:
        cv2.putText(
            annotated,
            label,
            (px1, max(20, py1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_file), annotated)
    return out_file
