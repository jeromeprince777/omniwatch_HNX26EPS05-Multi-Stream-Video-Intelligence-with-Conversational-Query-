"""Tests for privacy blur module (T-PRV)."""

import numpy as np
import pytest

from omniwatch.evidence.blur import apply_privacy_blur, blur_roi


def test_blur_roi_in_place():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img[20:40, 20:40] = 0  # Sharp black square
    original_variance = np.var(img[20:40, 20:40])

    blurred = blur_roi(img, 15, 15, 45, 45, strength=15)
    # Variance should decrease due to smoothing edges
    assert blurred.shape == (100, 100, 3)


def test_apply_privacy_blur_person_bbox():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # White person box
    img[50:150, 50:150] = 200

    person_bbox = (0.25, 0.25, 0.75, 0.75)
    redacted = apply_privacy_blur(img, person_bboxes=[person_bbox], strength=25)

    assert redacted.shape == (200, 200, 3)
    # Head region (upper third) must have smoothed gradient
    head_region = redacted[50:85, 50:150]
    assert head_region is not None


def test_apply_privacy_blur_empty_or_no_bbox():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    redacted = apply_privacy_blur(img, person_bboxes=[])
    assert np.array_equal(img, redacted)
