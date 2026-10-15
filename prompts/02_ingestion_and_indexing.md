# 02 — Manifest, decoding and the ingest orchestrator
**Read:** docs/00_PRD (FR-ING), docs/01_ARCHITECTURE (2, 3, 9), docs/04_DATA_PLAN (4).

**Task:**
1. `ingest/manifest.py`: load/validate `manifest.yaml` (pydantic). Require `start_time` unless `--assume-synced`. Support multiple video parts per camera ordered by start time. Convert to epoch seconds UTC.
2. `ingest/decode.py`: iterator yielding `(frame_idx, ts_epoch, frame_bgr)` at the configured sample FPS using a decoder you verify (PyAV or OpenCV; pick one, justify with a quick decode-speed comparison from the hardware probe). Optional motion gating (frame-difference threshold from config). Handle corrupt files by logging and continuing.
3. `ingest/pipeline.py`: orchestrator with stages as pluggable callables (detect, track, embed are implemented in prompts 03 and 04; use stubs that raise NotImplementedError for now but keep the interfaces). Idempotency: before processing a camera, skip it if rows exist for `(site_id, camera_id)` unless `--force`. Parallelism per camera with a worker count from config. Progress state exposed through a small JSON/DB table for `/api/ingest/status`.
4. Write `cameras` rows including duration, native fps, resolution and a camera thumbnail (first non-black frame).
5. `scripts/ingest.sh manifest.yaml [--preset fast|balanced|accurate] [--assume-synced] [--force]`.

**Acceptance:** tests with a tiny synthetic video generated in the test (ffmpeg testsrc) verify timestamps = start_time + t, idempotency, missing start_time failure message, and corrupt file skipping. Report decode throughput numbers from your run (do not estimate).
