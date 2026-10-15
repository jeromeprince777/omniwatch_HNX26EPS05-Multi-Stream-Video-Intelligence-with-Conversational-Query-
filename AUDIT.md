# AUDIT.md — OmniWatch Architecture, Data Flow & Coordinate Lifecycle Audit

**Project:** HackNex 2026 Problem Statement HNX26EPS05 (Multi-Stream Video Intelligence)  
**Date:** 2026-10-09  
**Status:** Step 0 Complete (Audit & Root Cause Analysis)

---

## 1. System Inventory: Files & Functions by Responsibility

| Pipeline Stage | Responsible File | Key Classes & Functions |
|---|---|---|
| **Frame Decoding** | `omniwatch/ingest/decode.py` | `decode_video_stream()`: Decodes video at `sample_fps` (default 2 FPS) with motion gating, yields `FrameRecord`.<br>`VideoDecoder.read_frame_at_index()`: Random access seeking via `cv2.CAP_PROP_POS_FRAMES`.<br>`get_video_metadata()`: Reads FPS, width, height, frame count, duration.<br>`extract_representative_thumbnail()`: Extracts initial non-dark frame for camera gallery. |
| **Detection** | `omniwatch/detect/detector.py` | `BaseDetector`: Abstract base class.<br>`YOLOWorldDetector.detect()`: Batched inference via Ultralytics YOLO-World, outputs `list[list[Box]]`.<br>`ClosedSetDetector.detect()`: Closed-set YOLOv8 COCO inference.<br>`Box`: Dataclass holding normalized coords `(x1, y1, x2, y2)`, class, conf, quality, and BGR crop. |
| **Tracking** | `omniwatch/track/tracker.py`<br>`omniwatch/track/tracklets.py` | `CameraTracker.update()`: Per-camera isolated ByteTrack tracker.<br>`CameraTracker._match_detections()`: Two-stage Hungarian association (IoU cost matrix + appearance fallback).<br>`assemble_tracklet()`: Aggregates detection history into `TrackletAssembly` with quality-weighted embedding. |
| **Crop Extraction & Attributes** | `omniwatch/embed/attrs.py`<br>`omniwatch/detect/detector.py` | `extract_colors()`: Color naming via K-Means and CIELAB color-space mapping.<br>`associate_carries()`: Associates person boxes with nearby/overlapping bags.<br>`compute_quality_score()`: Weighted product of detector confidence, Laplacian variance (sharpness), and box area.<br>`detector.py:lines 128-132`: Pixel slice `frame_img[py1:py2, px1:px2]` extraction. |
| **Embedding** | `omniwatch/embed/encoder.py` | `VLEncoder.encode_images()`: Batched OpenCLIP / SigLIP image embedding with L2 normalization.<br>`VLEncoder.encode_text()`: Prompt ensemble text embedding with L2 normalization. |
| **FAISS Indexing** | `omniwatch/index/vector.py` | `VectorIndex.add()`: Appends L2-normalized float32 vectors to FAISS `IndexFlatIP`.<br>`VectorIndex.search()`: Dot-product inner product similarity search.<br>`VectorIndex.save() / load()`: Persists `.index` and `_ids.npy` mappings on disk. |
| **Search & Retrieval** | `omniwatch/query/retrieve.py`<br>`omniwatch/query/resolver.py`<br>`omniwatch/query/parser.py`<br>`omniwatch/query/rerank.py` | `retrieve_candidates()`: Combines SQL prefilter (camera, time, class) with vector dot-product scoring and attribute fusion.<br>`parse_query()`: NL query parser (Qwen 2.5 LLM / Ollama with `RuleQueryParser` fallback).<br>`resolve_time_range()`, `resolve_location_referents()`: Maps referents and relative temporal bounds.<br>`SmolVLM2Verifier.rerank()`: Visual question answering verification scoring. |
| **Overlay & Thumbnail Rendering** | `omniwatch/evidence/thumbs.py`<br>`omniwatch/api/main.py` | `render_detection_thumbnail()`: Draws bounding box with tactical HUD corners and label on full frame.<br>`crop_detection()`: Crops bounding box with configurable padding (default 5%).<br>`GET /api/results/{tracklet_id}/thumb`: FastAPI handler that queries SQLite, calls `VideoDecoder.read_frame_at_index()`, and generates the overlay JPEG. |

---

## 2. End-to-End Data Flow

```
Raw MP4 Videos + manifest.yaml
   │
   ├─► VideoDecoder / decode_video_stream() [omniwatch/ingest/decode.py]
   │      Decodes frames at sample_fps (2.0), applies motion gating
   │      Emits: FrameRecord(frame_idx, timestamp, frame)
   │
   ├─► Detector [omniwatch/detect/detector.py]
   │      YOLO-World / YOLOv8s inference
   │      Extracts pixel crops: frame[py1:py2, px1:px2]
   │      Emits: Box(x1, y1, x2, y2 [0..1], cls, conf, quality, crop)
   │
   ├─► Attributes & Associations [omniwatch/embed/attrs.py]
   │      extract_colors(crop) -> primary_color, secondary_color
   │      associate_carries(person_box, bag_boxes) -> carries list
   │
   ├─► Vision-Language Encoder [omniwatch/embed/encoder.py]
   │      Batched OpenCLIP ViT-B-32 -> L2-normalized 512-dim vectors
   │
   ├─► Single-Camera Tracker [omniwatch/track/tracker.py]
   │      Per-camera ByteTrack with IoU cost matrix
   │      Assembles SingleTrack -> TrackletAssembly
   │
   ├─► Storage Persistence [omniwatch/db.py & omniwatch/index/vector.py]
   │      SQLite observations: (obs_id, camera_id, tracklet_id, frame_idx, ts, x1, y1, x2, y2, cls, det_conf, quality, carries...)
   │      SQLite tracklets: (tracklet_id, camera_id, t_start, t_end, best_obs_id, attrs_json, emb_row...)
   │      FAISS IndexFlatIP: obs.index (per-crop) & trk.index (tracklet aggregates)
   │
   └─► Query & Serving [omniwatch/query/ & omniwatch/api/main.py]
          Query -> Parser -> Filter & FAISS search -> Attribute Fusion -> [SmolVLM2 Verify]
          Evidence cards: GET /api/results/{id}/thumb -> reads video frame -> draws box -> JPEG
```

---

## 3. Coordinate & Frame Index Lifecycle: Creation, Storage & Consumption

### A. Frame Index Lifecycle
1. **Creation:** In `omniwatch/ingest/decode.py:decode_video_stream()`:
   - `native_frame_idx` tracks actual OpenCV sequential frame count ($0, 1, 2, \dots, N$).
   - `sampled_frame_idx` tracks the downsampled yield count ($0, 1, 2, \dots$ at 2 FPS).
   - **CRITICAL DEFECT IDENTIFIED**: Line 186 set `FrameRecord.frame_idx = sampled_frame_idx` instead of `native_frame_idx`.
2. **Storage:** In `omniwatch/ingest/pipeline.py:lines 323-348`:
   - `INSERT INTO observations (..., frame_idx, ts, ...)` stored `det.frame_idx`, which received `sampled_frame_idx`.
3. **Consumption:** In `omniwatch/api/main.py:get_thumbnail()`:
   - Calls `VideoDecoder.read_frame_at_index(frame_idx)`.
   - `VideoDecoder` sets `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)`.
   - Because `frame_idx` was the downsampled index (e.g. 10 instead of native frame 150), OpenCV sought to native frame 10 (0.33 seconds into video, showing the opening skyline/wide view), while the bounding box was from native frame 150 (5.0 seconds into video, where the person/news anchor was).
   - Furthermore, `cv2.CAP_PROP_POS_FRAMES` has known seeking inaccuracy on non-keyframe H.264 video streams.

### B. Bounding Box Coordinate Lifecycle
1. **Creation:** In `omniwatch/detect/detector.py:detect()`:
   - YOLO returns absolute pixel coordinates: `[bx1, by1, bx2, by2]`.
   - Detector computes normalized coordinates:
     $$x_1 = \frac{\text{bx1}}{w},\quad y_1 = \frac{\text{by1}}{h},\quad x_2 = \frac{\text{bx2}}{w},\quad y_2 = \frac{\text{by2}}{h}$$
   - Pixel crop is sliced from `frame_img[py1:py2, px1:px2]`.
2. **Storage:**
   - SQLite table `observations` stores `x1, y1, x2, y2` as normalized floats ($0.0 \dots 1.0$).
   - Notice: Absolute pixel bounding boxes `(px1, py1, px2, py2)` and frame dimensions `(frame_w, frame_h)` were **not** saved in SQLite.
3. **Consumption:**
   - In `omniwatch/evidence/thumbs.py:render_detection_thumbnail()`:
     Multiplies normalized `(x1, y1, x2, y2)` by current decoded `frame.shape[:2]` ($w, h$) to obtain `(px1, py1, px2, py2)`.
   - In the web UI (`web/app.js`):
     The UI displays the server-rendered JPEG thumbnail directly, but when overlaying or rendering frame inspection, natural dimensions vs container dimensions must match.

---

## 4. Root Cause of Problems Identified

1. **Problem 1 (Tracklet #338 draws box on empty skyline instead of news anchor):**
   - **Root Cause (a):** Off-by-stride frame index. `decode_video_stream` emitted `sampled_frame_idx` (0, 1, 2...) instead of `native_frame_idx` (0, 15, 30...). At 2 FPS sampling on a 30 FPS video, sampled index 338 corresponds to native frame 338 (11.2s into video) instead of native frame 5070 (169s into video). Seeking native frame 338 lands on early studio/skyline B-roll footage.
   - **Root Cause (b):** Random seeking via `CAP_PROP_POS_FRAMES` on non-intra H.264 video can land on nearby keyframes rather than the exact target frame.
   - **Fix Required:** Decode sequentially, store `native_frame_idx`, and pre-render/cache observation keyframe thumbnails directly at index time.
2. **Problem 2 ("CONF: 38.5%" is raw CLIP similarity, not detection confidence):**
   - In `omniwatch/query/retrieve.py`, when no attributes are specified, `final_score` equals raw `emb_sim` ($\sim 0.38 - 0.39$).
   - The UI formatted this as `CONF: ${scorePercent}%`. Detection confidence (`det_conf` from YOLO) was stored in `observations` but never returned in `ResultItem` or exposed separately in the UI.
   - **Fix Required:** Return both `det_conf` and `match_score` in `CandidateResult` and `ResultItem`, calibrate CLIP similarity via negative prompt margins, and relabel the UI to "DETECTION" and "MATCH".
3. **Problem 3 (Qwen 2.5 and SmolVLM2 never used on FAST path):**
   - `main.py` defaulted query mode to `"fast"`. On `"fast"`, VLM verification is bypassed.
   - In `omniwatch/query/parser.py`, `RuleQueryParser.is_confident()` returned True for most standard queries, causing `LLMQueryParser` to be skipped.
   - Furthermore, `LLMQueryParser` was configured to call Ollama HTTP rather than the local HuggingFace/transformers Qwen integration.
   - **Fix Required:** Set default mode to ACCURATE, wire local Qwen 2.5 JSON parser with rule fallback, and run SmolVLM2 verification on top-15 candidates within the 6 GB VRAM budget.

---

## 5. Execution Roadmap (Steps 1–7)

- **Step 1:** Create `scripts/debug_bbox.py` to inspect tracklet #338 on the full frame, fix `omniwatch/ingest/decode.py` (`native_frame_idx` + `sampled_frame_idx`), persist both absolute pixel coordinates and normalized coordinates plus `frame_w, frame_h` in SQLite, cache thumbnails at index time, and build `scripts/contact_sheet.py` sampling 30 random observations.
- **Step 2:** Standardize YOLOv8s with `model.track(persist=True, tracker="bytetrack.yaml")`, filter COCO classes with `conf >= 0.40`, `imgsz=640`, drop boxes `< 64px` height or `< 0.2%` frame area, and rebuild the index.
- **Step 3:** Implement structured bag association (expanded person box by 10%, overlap $\ge 0.5$ over bag area), store `carrying_bag, bag_type, bag_conf`, and add metadata search prefiltering.
- **Step 4:** 25% padded crops for person embeddings, test/verify OpenCLIP model options via `open_clip.list_pretrained()`, rebuild FAISS IndexFlatIP, implement contrastive softmax margin calibration, and separate `DETECTION` vs `MATCH` metrics in API and UI.
- **Step 5:** Wire Qwen 2.5 structured JSON parser with rule fallback, run SmolVLM2-2.2B verification on top 15 candidates with fp16/4-bit under 6 GB VRAM budget, and report stage latencies.
- **Step 6:** Create `eval/queries.json` (15–20 queries) and `eval/run_ablation.py` computing Recall@1, Recall@5, camera accuracy, temporal IoU, and p50/p95 latencies across the 4 configurations; write `eval/results.md`.
- **Step 7:** Run `contact_sheet.py` and 3 demo queries, verify Clarify-Once SQLite persistence across restart, and produce `RUNBOOK.md`.
