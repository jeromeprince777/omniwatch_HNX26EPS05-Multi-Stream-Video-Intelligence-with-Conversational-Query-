"""Benchmark and Acceptance Runner for Stage 03 Detection and Embeddings.

Runs detector and encoder on a sample frame, annotates bounding boxes
with class, confidence, quality, and detected colors, and saves the annotated
frame to benchmark/annotated_detection_sample.jpg.
Records measured per-frame timings to benchmark/detect_embed_benchmark.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from omniwatch.config import load_config
from omniwatch.detect.detector import get_detector
from omniwatch.embed.attrs import extract_colors
from omniwatch.embed.encoder import VLEncoder


def create_realistic_test_scene() -> np.ndarray:
    """Creates a synthetic multi-object surveillance scene with a red car, person, and bag."""
    h, w = 720, 1280
    # Background (asphalt ground + wall)
    scene = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
    cv2.rectangle(scene, (0, 0), (w, 300), (140, 130, 120), -1)  # Wall

    # Red Car (BGR: 20, 20, 220)
    cv2.rectangle(scene, (150, 380), (550, 580), (20, 20, 220), -1)
    cv2.rectangle(scene, (230, 320), (450, 400), (40, 40, 180), -1)  # Cabin
    cv2.circle(scene, (250, 580), 45, (20, 20, 20), -1)  # Wheel
    cv2.circle(scene, (460, 580), 45, (20, 20, 20), -1)  # Wheel

    # Blue Person (BGR: 220, 40, 20) with backpack
    cv2.circle(scene, (850, 260), 30, (180, 200, 220), -1)  # Head
    cv2.rectangle(scene, (810, 290), (890, 490), (220, 60, 30), -1)  # Blue Torso/Coat
    cv2.rectangle(scene, (820, 490), (880, 640), (30, 30, 30), -1)  # Legs
    cv2.rectangle(scene, (880, 320), (930, 440), (20, 20, 20), -1)  # Black Backpack

    return scene


def run_benchmark() -> dict:
    benchmark_dir = Path("benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(preset="balanced")
    print(f"[Benchmark] Initializing Detector ({config.current.detector_model})...")
    detector = get_detector(config)

    print("[Benchmark] Initializing Vision-Language Encoder (ViT-B-32)...")
    encoder = VLEncoder(model_name=config.current.encoder_name, pretrained=config.current.encoder_pretrained)

    real_frame_path = benchmark_dir / "real_frame.jpg"
    if real_frame_path.exists():
        scene = cv2.imread(str(real_frame_path))
    else:
        scene = create_realistic_test_scene()
    h, w = scene.shape[:2]

    # Warmup runs
    _ = detector.detect([scene])
    _ = encoder.encode_images([scene[100:200, 100:200]])

    # 1. Measure Detector Timing
    t0 = time.perf_counter()
    results = detector.detect([scene])
    detector_ms = (time.perf_counter() - t0) * 1000.0
    boxes = results[0]

    annotated = scene.copy()
    crops_to_encode = []
    box_details = []

    # 2. Extract Colors and Annotate Boxes
    t_attrs_start = time.perf_counter()
    for b in boxes:
        bx1 = int(b.x1 * w)
        by1 = int(b.y1 * h)
        bx2 = int(b.x2 * w)
        by2 = int(b.y2 * h)

        crop = scene[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else None
        p_color, s_color = extract_colors(crop, cls=b.cls) if crop is not None else ("unknown", "unknown")

        if crop is not None:
            crops_to_encode.append(crop)

        label = f"{b.cls} {b.conf:.2f} [{p_color}] q:{b.quality:.2f}"
        box_details.append({
            "cls": b.cls,
            "conf": round(b.conf, 4),
            "bbox": [round(b.x1, 4), round(b.y1, 4), round(b.x2, 4), round(b.y2, 4)],
            "quality": b.quality,
            "primary_color": p_color,
            "secondary_color": s_color,
        })

        # Draw box and label
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (bx1, max(20, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    attr_ms = (time.perf_counter() - t_attrs_start) * 1000.0

    # 3. Measure Crop Embedding Timing
    if crops_to_encode:
        t_enc_start = time.perf_counter()
        _ = encoder.encode_images(crops_to_encode)
        enc_ms = (time.perf_counter() - t_enc_start) * 1000.0
        ms_per_crop = enc_ms / len(crops_to_encode)
    else:
        enc_ms = 0.0
        ms_per_crop = 0.0

    # 4. Measure Prompt Ensemble Timing
    t_txt_start = time.perf_counter()
    _ = encoder.encode_prompt_ensemble("a red car")
    txt_ms = (time.perf_counter() - t_txt_start) * 1000.0

    # Save Annotated Image
    out_img = benchmark_dir / "annotated_detection_sample.jpg"
    cv2.imwrite(str(out_img), annotated)

    benchmark_stats = {
        "device": detector.device,
        "detector_model": detector.model_name,
        "encoder_model": f"{encoder.model_name} ({encoder.pretrained})",
        "frame_resolution": f"{w}x{h}",
        "num_detections": len(boxes),
        "detector_latency_ms": round(detector_ms, 2),
        "attribute_latency_ms": round(attr_ms, 2),
        "crop_encoder_total_ms": round(enc_ms, 2),
        "crop_encoder_per_crop_ms": round(ms_per_crop, 2),
        "text_prompt_ensemble_latency_ms": round(txt_ms, 2),
        "detections": box_details,
        "annotated_image": str(out_img),
    }

    out_json = benchmark_dir / "detect_embed_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_stats, f, indent=2)

    print("\n" + "=" * 60)
    print(" OmniWatch Stage 03 Detection & Embedding Benchmark")
    print("=" * 60)
    print(f"Device: {detector.device}")
    print(f"Detector ({detector.model_name}): {detector_ms:.2f} ms / frame")
    print(f"Attribute Extraction: {attr_ms:.2f} ms ({len(boxes)} boxes)")
    print(f"Encoder Crop Embedding: {enc_ms:.2f} ms ({ms_per_crop:.2f} ms / crop)")
    print(f"Text Prompt Ensemble: {txt_ms:.2f} ms")
    print(f"Detections Found: {len(boxes)}")
    for d in box_details:
        print(f"  - {d['cls']} (conf={d['conf']}, color={d['primary_color']}, quality={d['quality']})")
    print(f"Saved annotated sample: {out_img}")
    print(f"Saved report: {out_json}")
    print("=" * 60)

    return benchmark_stats


if __name__ == "__main__":
    run_benchmark()
