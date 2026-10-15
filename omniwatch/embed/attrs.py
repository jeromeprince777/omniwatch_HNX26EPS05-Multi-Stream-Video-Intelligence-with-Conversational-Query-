"""Explicit Attribute Extraction Module (Color Naming and Carries Association).

Fixes CLIP color-binding and compound phrase weaknesses by:
1. Converting object crops to perceptual CIE Lab space.
2. Clustering dominant colors across class-specific subregions (torso for persons, body for vehicles).
3. Mapping centroids to a calibrated named palette (red, orange, yellow, green, blue, purple,
   pink, brown, black, white, gray, silver, beige).
4. Associating carried items (backpack, handbag, suitcase, box) by spatial overlap and proximity.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Calibrated standard CIE Lab reference values for named palette
NAMED_PALETTE_LAB: dict[str, tuple[float, float, float]] = {
    "red": (53.2, 80.1, 67.2),
    "orange": (67.0, 43.0, 74.0),
    "yellow": (87.7, -8.6, 86.8),
    "green": (55.0, -50.0, 45.0),
    "blue": (32.3, 79.2, -107.9),
    "purple": (40.0, 50.0, -40.0),
    "pink": (70.0, 55.0, 15.0),
    "brown": (38.0, 20.0, 30.0),
    "black": (5.0, 0.0, 0.0),
    "white": (98.0, 0.0, 0.0),
    "gray": (50.0, 0.0, 0.0),
    "silver": (75.0, 0.0, 0.0),
    "beige": (85.0, 5.0, 20.0),
}


def _delta_e(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """Euclidean color difference in CIE Lab space."""
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )


def match_lab_to_palette(lab: tuple[float, float, float]) -> str:
    """Finds closest named color in palette by minimal delta_e."""
    best_name = "gray"
    min_dist = float("inf")
    for name, ref_lab in NAMED_PALETTE_LAB.items():
        dist = _delta_e(lab, ref_lab)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


def extract_colors(crop: np.ndarray, cls: str | None = None) -> tuple[str, str]:
    """Extracts dominant primary and secondary color names from an image crop.

    Args:
        crop: BGR image crop as np.ndarray.
        cls: Object class name to select relevant region (e.g. torso for person).

    Returns:
        tuple[str, str]: (primary_color_name, secondary_color_name).
    """
    if crop is None or crop.size == 0:
        return "gray", "gray"

    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return "gray", "gray"

    # 1. Select region of interest depending on class
    if cls == "person":
        # Torso: vertical 20% to 65%, horizontal 20% to 80%
        y1, y2 = int(h * 0.20), int(h * 0.65)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        roi = crop[y1:y2, x1:x2]
    elif cls in ("car", "truck", "bus", "motorcycle", "vehicle"):
        # Central vehicle body: vertical 25% to 75%, horizontal 15% to 85%
        y1, y2 = int(h * 0.25), int(h * 0.75)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        roi = crop[y1:y2, x1:x2]
    else:
        # Generic central 70%
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        roi = crop[y1:y2, x1:x2]

    if roi.size == 0:
        roi = crop

    # 2. Convert to RGB and OpenCV Lab
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    lab_cv = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2Lab)

    # Flatten pixels to shape [N, 3]
    pixels_lab = lab_cv.reshape(-1, 3).astype(np.float32)

    # Convert OpenCV Lab [0..255] to standard CIE Lab:
    # L_std = L_cv * 100 / 255; a_std = a_cv - 128; b_std = b_cv - 128
    l_std = pixels_lab[:, 0] * (100.0 / 255.0)
    a_std = pixels_lab[:, 1] - 128.0
    b_std = pixels_lab[:, 2] - 128.0
    std_lab_pixels = np.column_stack([l_std, a_std, b_std])

    # 3. K-means clustering (k=3)
    k = min(3, len(std_lab_pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        std_lab_pixels.astype(np.float32),
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    # Count pixels per cluster
    counts = np.bincount(labels.flatten(), minlength=k)
    sorted_cluster_indices = np.argsort(-counts)

    primary_lab = tuple(centers[sorted_cluster_indices[0]])
    primary_color = match_lab_to_palette(primary_lab)  # type: ignore[arg-type]

    secondary_color = primary_color
    for idx in sorted_cluster_indices[1:]:
        cand_lab = tuple(centers[idx])
        cand_color = match_lab_to_palette(cand_lab)  # type: ignore[arg-type]
        if cand_color != primary_color:
            secondary_color = cand_color
            break

    return primary_color, secondary_color


def associate_carries(
    person_bbox: tuple[float, float, float, float],
    candidate_bboxes: list[tuple[tuple[float, float, float, float], str]],
    overlap_threshold: float = 0.05,
    proximity_threshold: float = 0.08,
) -> list[str]:
    """Identifies carried objects (bags, backpacks, boxes) associated with a person.

    Args:
        person_bbox: (px1, py1, px2, py2) in normalized [0, 1] coords.
        candidate_bboxes: list of ((bx1, by1, bx2, by2), class_name).
        overlap_threshold: intersection / bag_area threshold for containment.
        proximity_threshold: max normalized distance between centers or borders.

    Returns:
        list[str]: Sorted, deduplicated list of carried object classes.
    """
    px1, py1, px2, py2 = person_bbox
    p_w = px2 - px1
    p_h = py2 - py1
    if p_w <= 0 or p_h <= 0:
        return []

    carried: set[str] = set()

    for (bx1, by1, bx2, by2), cls_name in candidate_bboxes:
        b_w = bx2 - bx1
        b_h = by2 - by1
        bag_area = max(b_w * b_h, 1e-6)

        # Intersection
        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection_area = iw * ih
        overlap_ratio = intersection_area / bag_area

        if overlap_ratio >= overlap_threshold:
            carried.add(cls_name)
            continue

        # Proximity check: boundary distance between boxes
        dx = max(0.0, max(px1 - bx2, bx1 - px2))
        dy = max(0.0, max(py1 - by2, by1 - py2))
        dist = math.hypot(dx, dy)
        if dist <= proximity_threshold:
            carried.add(cls_name)

    return sorted(carried)


def associate_person_bag(
    person_bbox: tuple[float, float, float, float],
    bag_candidates: list[tuple[tuple[float, float, float, float], str, float]],
    expand_ratio: float = 0.10,
    min_bag_overlap: float = 0.50,
) -> tuple[bool, str | None, float]:
    """Associates bag boxes (backpack, handbag, suitcase) with a person box.

    Checks if bag box overlaps person box (intersection / bag_area >= 0.5,
    with person box expanded by 10% on each side).

    Returns:
        (carrying_bag: bool, bag_type: str | None, bag_conf: float)
    """
    px1, py1, px2, py2 = person_bbox
    pw = px2 - px1
    ph = py2 - py1
    if pw <= 0 or ph <= 0:
        return (False, None, 0.0)

    # Expand person box by 10%
    exp_px1 = max(0.0, px1 - expand_ratio * pw)
    exp_py1 = max(0.0, py1 - expand_ratio * ph)
    exp_px2 = min(1.0, px2 + expand_ratio * pw)
    exp_py2 = min(1.0, py2 + expand_ratio * ph)

    best_match = (False, None, 0.0)
    best_overlap = -1.0

    for (bx1, by1, bx2, by2), bag_cls, bag_conf in bag_candidates:
        bw = bx2 - bx1
        bh = by2 - by1
        bag_area = max(bw * bh, 1e-6)

        ix1 = max(exp_px1, bx1)
        iy1 = max(exp_py1, by1)
        ix2 = min(exp_px2, bx2)
        iy2 = min(exp_py2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection_area = iw * ih
        overlap_ratio = intersection_area / bag_area

        if overlap_ratio >= min_bag_overlap and overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_match = (True, bag_cls, float(bag_conf))

    return best_match


def compute_quality_score(
    crop: np.ndarray,
    bbox: tuple[float, float, float, float],
    conf: float,
) -> float:
    """Computes detection crop quality score in [0, 1].

    Combines box area, Laplacian image sharpness, and detector confidence.
    """
    if crop is None or crop.size == 0:
        return 0.0

    # 1. Normalized bbox area
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1) * (y2 - y1))
    # Area saturation: 0.005 is small, 0.08+ is full score
    area_score = float(np.clip((area - 0.005) / 0.075, 0.0, 1.0))

    # 2. Sharpness via Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize with exponential saturation
    sharpness_score = float(1.0 - math.exp(-max(0.0, lap_var) / 120.0))

    # 3. Detector confidence
    conf_score = float(np.clip(conf, 0.0, 1.0))

    # Weighted quality
    quality = 0.40 * conf_score + 0.35 * sharpness_score + 0.25 * area_score
    return round(float(np.clip(quality, 0.0, 1.0)), 4)
