# 04 — Per-camera tracking, tracklets, vector index
**Read:** docs/00_PRD (FR-IDX), docs/01_ARCHITECTURE (3, 5).

**Task:**
1. `track/tracker.py`: per-camera tracking with a ByteTrack-class tracker (verify the library API; Ultralytics built-in or `supervision`). Tracker state must never mix cameras. Note the sample FPS affects tracker parameters; make them config-driven and document the chosen values.
2. `track/tracklets.py`: assemble tracklets (`t_start`, `t_end`, class by majority vote, quality-weighted aggregate embedding, top-k observation ids by quality, attribute summary: color histogram, carries frequency). Close a tracklet after `max_gap_s` with no detections (config).
3. Fill `observations` and `tracklets` tables; write embeddings to `.npy` shards and build FAISS `IndexFlatIP` files `obs.index` and `trk.index` through `index/vector.py` with `add`, `search(query_vec, row_filter)`, `save`, `load`, `rebuild_from_shards`. Row filtering: implement pre-filtering by candidate row ids (from SQL) then exact scoring over that subset.
4. Wire everything into the ingest orchestrator from prompt 02 so `scripts/ingest.sh` produces a complete index.
5. Idempotency and restart: after ingest, a fresh process can load indexes and answer a raw vector query without re-running models other than the text encoder.

**Acceptance:** ingest a short multi-camera test set; print counts (observations, tracklets per camera); a script `scripts/probe_query.py "red car"` returns top tracklets with camera/time. Report real counts and timings. Verify no tracker ids leak across cameras with a test.
