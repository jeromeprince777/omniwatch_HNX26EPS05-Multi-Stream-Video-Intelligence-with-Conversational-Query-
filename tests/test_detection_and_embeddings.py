"""Unit and Smoke Tests for Object Detection, Attribute Extraction, and Embeddings."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from omniwatch.detect.detector import YOLOWorldDetector
from omniwatch.embed.attrs import associate_carries, compute_quality_score, extract_colors
from omniwatch.embed.encoder import VLEncoder


def test_color_naming_synthetic():
    """Verifies that synthetic pure-color rectangles are correctly classified."""
    # Red rectangle (BGR: 0, 0, 255)
    red_crop = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    primary, _ = extract_colors(red_crop)
    assert primary == "red", f"Expected 'red', got '{primary}'"

    # Blue rectangle (BGR: 255, 0, 0)
    blue_crop = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    primary, _ = extract_colors(blue_crop)
    assert primary == "blue", f"Expected 'blue', got '{primary}'"

    # Green rectangle (BGR: 0, 255, 0)
    green_crop = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)
    primary, _ = extract_colors(green_crop)
    assert primary == "green", f"Expected 'green', got '{primary}'"

    # White rectangle (BGR: 255, 255, 255)
    white_crop = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)
    primary, _ = extract_colors(white_crop)
    assert primary == "white", f"Expected 'white', got '{primary}'"

    # Black rectangle (BGR: 10, 10, 10)
    black_crop = np.full((100, 100, 3), (10, 10, 10), dtype=np.uint8)
    primary, _ = extract_colors(black_crop)
    assert primary == "black", f"Expected 'black', got '{primary}'"

    # Yellow rectangle (BGR: 0, 255, 255)
    yellow_crop = np.full((100, 100, 3), (0, 255, 255), dtype=np.uint8)
    primary, _ = extract_colors(yellow_crop)
    assert primary == "yellow", f"Expected 'yellow', got '{primary}'"


def test_carries_association():
    """Verifies spatial containment and proximity rules for carried items."""
    person_box = (0.2, 0.2, 0.4, 0.8)

    # Overlapping backpack
    overlapping_bag = ((0.25, 0.3, 0.35, 0.5), "backpack")
    # Proximity adjacent box
    adjacent_box = ((0.41, 0.4, 0.46, 0.55), "box")
    # Distant handbag (not carried)
    distant_bag = ((0.75, 0.75, 0.85, 0.85), "handbag")

    candidates = [overlapping_bag, adjacent_box, distant_bag]
    carried = associate_carries(person_box, candidates)

    assert "backpack" in carried
    assert "box" in carried
    assert "handbag" not in carried


def test_quality_scoring():
    """Verifies that sharpness, confidence, and size properly impact crop quality."""
    sharp_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(sharp_crop, (50, 50), 30, (255, 255, 255), 2)  # High gradient edges
    high_q = compute_quality_score(sharp_crop, (0.1, 0.1, 0.5, 0.5), conf=0.95)

    blurred_crop = cv2.GaussianBlur(sharp_crop, (25, 25), 0)  # Very blurry
    low_q = compute_quality_score(blurred_crop, (0.1, 0.1, 0.15, 0.15), conf=0.20)

    assert high_q > low_q
    assert 0.0 <= high_q <= 1.0
    assert 0.0 <= low_q <= 1.0


def test_encoder_smoke_and_shapes():
    """Smoke test for VLEncoder verifying normalized embedding shapes."""
    encoder = VLEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="auto")
    assert encoder.dimension == 512

    # Image encoding
    crop1 = np.full((120, 120, 3), (0, 0, 255), dtype=np.uint8)
    crop2 = np.full((80, 80, 3), (255, 0, 0), dtype=np.uint8)
    img_embs = encoder.encode_images([crop1, crop2])

    assert img_embs.shape == (2, 512)
    # Check L2-normalization: ||v|| == 1
    norms = np.linalg.norm(img_embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)

    # Text encoding
    text_embs = encoder.encode_text(["a red car", "a person in black"])
    assert text_embs.shape == (2, 512)
    norms_text = np.linalg.norm(text_embs, axis=1)
    np.testing.assert_allclose(norms_text, [1.0, 1.0], atol=1e-4)

    # Prompt ensemble
    ens_emb = encoder.encode_prompt_ensemble("red car")
    assert ens_emb.shape == (512,)
    assert pytest.approx(np.linalg.norm(ens_emb), abs=1e-4) == 1.0


def test_detector_smoke_and_shapes():
    """Smoke test verifying detector adapter runs and returns normalized boxes."""
    # Synthetic frame with a white vehicle-like rectangle and a blue circle
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(frame, (100, 150), (350, 350), (255, 255, 255), -1)
    cv2.circle(frame, (450, 250), 60, (255, 100, 0), -1)

    detector = YOLOWorldDetector(
        model_name="models/yolov8s-worldv2.pt",
        vocabulary=["car", "person", "box"],
        conf_thresh=0.10,
        device="auto",
    )
    results = detector.detect([frame])
    assert len(results) == 1
    boxes = results[0]

    for b in boxes:
        assert 0.0 <= b.x1 <= 1.0
        assert 0.0 <= b.y1 <= 1.0
        assert 0.0 <= b.x2 <= 1.0
        assert 0.0 <= b.y2 <= 1.0
        assert 0.0 <= b.conf <= 1.0
        assert 0.0 <= b.quality <= 1.0
        assert isinstance(b.cls, str)
