# RUNBOOK — OmniWatch Multi-Stream Video Intelligence HUD

**Problem Statement:** HNX26EPS05 (Multi-Stream Video Intelligence with Conversational Query)  
**Target Hardware:** Laptop NVIDIA RTX 4050 (6 GB VRAM), 16 GB RAM, Windows 11  
**Project Version:** 1.0.0

---

## 1. System Requirements & Setup

### Environment
- **Python:** 3.11 (64-bit)
- **CUDA:** 12.1 / 12.4 compatible with NVIDIA GeForce RTX 4050 Laptop GPU (Driver >= 535)
- **Virtual Environment:** `.venv`

### Package Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support matching driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install OmniWatch dependencies
pip install ultralytics open_clip_torch faiss-cpu transformers accelerate opencv-python pydantic pyyaml fastapi uvicorn httpx scipy pillow
```

### Models & Weights Reference
All weights are stored locally in `models/` (see [`models/MANIFEST.md`](file:///c:/Users/ffgam/OneDrive/Desktop/omniwatch_pack/models/MANIFEST.md)):
- **YOLOv8s (`models/yolov8s.pt`):** ~22.5 MB (Closed-set detector for 10 COCO classes)
- **OpenCLIP ViT-B-32 (`models/open_clip_model.safetensors`):** ~605 MB (LAION-2B `laion2b_s34b_b79k`)
- **SmolVLM2-2.2B-Instruct (`models/SmolVLM2-2.2B-Instruct`):** ~4.4 GB (Local HuggingFace checkpoints for visual verification)
- **Qwen 2.5 (`models/Qwen2.5-VL-3B-Instruct` / Ollama `qwen2.5:3b`):** Query parsing and structural intent extraction.

---

## 2. Ingestion & Index Rebuilding

The ingestion pipeline sequentially decodes multi-camera footage at `sample_fps=2.0`, records native frame indices, performs YOLOv8 tracking with ByteTrack, associates bags with persons, computes OpenCLIP embeddings on 25% padded crops, and pre-renders observation thumbnails.

### Building the Index
```powershell
# Run ingestion on public CCTV dataset
.venv\Scripts\python.exe -m omniwatch.ingest.pipeline data/public_cctv/manifest.yaml --force

# Or using the convenience script
powershell -ExecutionPolicy Bypass -File scripts\ingest.ps1 data\public_cctv\manifest.yaml
```

**Artifacts Generated:**
- SQLite database: `data/omniwatch.db` (`cameras`, `observations`, `tracklets`, `referents`)
- FAISS vector indices: `data/index/obs.index` and `data/index/trk.index`
- Keyframe evidence thumbnails: `data/cache/thumbs/trk_{tracklet_id}.jpg`

---

## 3. Verification & Diagnostic Scripts

### Bounding Box & Frame Alignment Inspection
Verifies that bounding boxes are drawn on the exact native frame rather than drifting to early opening frames:
```powershell
.venv\Scripts\python.exe scripts\debug_bbox.py
```
**Output:** Saves [`debug/bbox_338.jpg`](file:///c:/Users/ffgam/OneDrive/Desktop/omniwatch_pack/debug/bbox_338.jpg).

### Contact Sheet Generator
Samples 30 random observations across the indexed footage and generates a 5x6 visual contact sheet:
```powershell
.venv\Scripts\python.exe scripts\contact_sheet.py
```
**Output:** Saves [`debug/contact_sheet.jpg`](file:///c:/Users/ffgam/OneDrive/Desktop/omniwatch_pack/debug/contact_sheet.jpg). Inspect to confirm all boxes tightly enclose people, bags, and vehicles.

### Research Ablation Study
Runs the 4 pipeline configurations against [`eval/queries.json`](file:///c:/Users/ffgam/OneDrive/Desktop/omniwatch_pack/eval/queries.json) and compiles the performance comparison table:
```powershell
.venv\Scripts\python.exe eval\run_ablation.py
```
**Output:** Generates markdown comparison table in [`eval/results.md`](file:///c:/Users/ffgam/OneDrive/Desktop/omniwatch_pack/eval/results.md).

---

## 4. Starting the Web HUD

To launch the Tactical Surveillance HUD and FastAPI REST backend:
```powershell
# Double-click or run:
.\start_omniwatch.bat

# Or directly:
.venv\Scripts\python.exe -m uvicorn omniwatch.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to: **`http://127.0.0.1:8000/`**

---

## 5. Verification Checklist

- [x] **Frame/Box Alignment:** `FrameRecord.frame_idx` set to `native_frame_idx` in `omniwatch/ingest/decode.py`.
- [x] **Coordinate Storage:** SQLite `observations` stores both normalized `(x1, y1, x2, y2)` and absolute pixel coordinates `(px1, py1, px2, py2)`, plus `frame_w` and `frame_h`.
- [x] **Pre-cached Thumbnails:** Thumbnails pre-rendered at index time so runtime display never requires OpenCV re-seeking.
- [x] **Detection Filtering:** Conf >= 0.40, imgsz 640, boxes < 64px height or < 0.2% area dropped.
- [x] **Bag Association:** Person expanded 10%, overlap >= 0.5 over bag area, stored in `carrying_bag, bag_type, bag_conf`.
- [x] **Calibrated CLIP:** Softmax margin over negative contrast prompts.
- [x] **Metrics Separation:** UI displays separate `DETECTION: X%` and `MATCH: Y%` metrics.
- [x] **Clarify-Once Durability:** Referents persist across server restarts in SQLite WAL database.

---

## 6. Minimal Debuggable Video-Search Tool (`omni_min.py`)

A self-contained, zero-overhead CLI tool for immediate debugging and ground-truth validation:
- Boxes and keyframes extracted simultaneously from YOLO results (`r.orig_img + r.boxes`), avoiding re-seeking drift.
- Keyframes (box drawn) and crops written directly to disk at detection time.
- Deterministic person-bag overlap (IoU over bag area >= 0.5 with 15% horizontal / 5% vertical expansion).
- Strict verification note flagging unverified matches when filters fall back.

### Usage
```powershell
# 1. Index footage
.\run_omni_min.bat index --cam Yokohama --video data/public_cctv/yokohama_cctv.mp4 --start "2026-10-09 09:00:00"
.\run_omni_min.bat index --cam Tours --video data/public_cctv/surveillance_tours.mp4 --start "2026-10-09 09:05:00"

# 2. Search
.\run_omni_min.bat search "person carrying a bag"
.\run_omni_min.bat search "red car"

# 3. Visual Contact Sheet (Sanity Check)
.\run_omni_min.bat sheet
# -> Generates out/sheet.jpg (30 random tracklet keyframes with labels & boxes)

# 4. Generate on-demand video clip (±2s cut with burned box & label)
.\run_omni_min.bat clip --cam Yokohama --tid 1067
# -> Generates data/clips/Yokohama_1067.mp4 (H.264 faststart transcode)
```

---

## 7. Evidence Video Clip Generation & HUD Playback

- **Dynamic Frame-by-Frame Tracking:** `omni_min.ensure_clip(cam, tid, pad_sec=2.0, force=False)` extracts the raw temporal window around `frame_idx`, executes YOLOv8 + ByteTrack across all clip frames on GPU, locks onto the target tracklet, and renders moving tactical bounding boxes, corner reticles, and motion trails on **every single frame** of the clip.
- **Threadpool Execution:** Render calls run asynchronously in a worker thread (`asyncio.to_thread`) via `GET /api/clip?cam={cam}&tid={tid}` so the main server event loop remains non-blocking. Pass `?force=true` to force a fresh re-render.
- **Fast-Start H.264 MP4:** Automatically transcoded via FFmpeg (`-c:v libx264 -preset ultrafast -movflags +faststart -an`) and cached to `data/clips/{cam}_{tid}.mp4`.
- **Static Route:** FastAPI serves cached video clips directly from `data/clips/` at `/clips/{cam}_{tid}.mp4` (`app.mount("/clips", StaticFiles(directory="data/clips"))`).
- **Interactive UX:** Clicking "▶ PLAY CLIP" activates a transient `RENDERING...` state on first click. Once ready, the modal video player plays the clip directly with native controls and speed toggles, without any synthetic browser overlays on top.
- **Keyframe Persistence:** "🎯 VIEW FRAME" serves the crisp saved keyframe with pre-drawn reticles (`/api/keyframe?path={path}`) without runtime re-seeking.

---

## 8. Real-Time Query-Driven Tracking (`omni_live.py`)

A standalone, GPU-saturating real-time query-driven tracking service:
- **Zero-Drop FIFO Pipeline:** Raw frames are streamed into a bounded FIFO queue (`_raw_q`). Every single frame is processed sequentially through YOLO + ByteTrack on GPU; no frames are discarded or skipped.
- **Direct Real-Time Encoding:** Each frame is rendered with active reticles, IDs, labels, and motion trails, encoded directly to JPEG (~3 ms), and broadcast to MJPEG clients at native 25 FPS pacing.
- **Multi-Modal Filtering:** Fast deterministic checks (class, body color vote, bag overlap) with asynchronous OpenCLIP evaluation for free-text attributes (batched tensor softmax on GPU).
- **Zero-Restart Query Updates:** POST `/api/query` updates target parameters live without resetting video playback or re-indexing.
- **Multi-Camera MJPEG Streams:** Native multi-camera grid streamed over HTTP MJPEG at `/stream/{name}` with interactive web console at `http://127.0.0.1:8100`.

### Running Real-Time Tracking
```powershell
# Quick launch with default Yokohama and Tours CCTV feeds:
.\run_omni_live.bat

# Or customize camera inputs and initial query:
.venv\Scripts\python.exe omni_live.py --src Yokohama=data/public_cctv/yokohama_cctv.mp4 --src Tours=data/public_cctv/surveillance_tours.mp4 --loop --query "person carrying a bag" --port 8100
```
Open **`http://127.0.0.1:8100`** in your browser to interactively type queries (*"person in red"*, *"white car"*, *"person with a hat"*, etc.).
