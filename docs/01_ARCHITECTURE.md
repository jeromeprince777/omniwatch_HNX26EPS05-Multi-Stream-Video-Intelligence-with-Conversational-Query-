# 01 — Architecture

## 1. Data flow
```
manifest.yaml + videos
   |  decode (PyAV/OpenCV/ffmpeg), sample fps, motion gate
   v
frames -> detector (boxes: person/vehicle/bag/...) -> per-camera tracker (ByteTrack-class)
   |                                                     |
   +-> crops -> VL encoder (CLIP/SigLIP) + color/carries attrs
   |
   v
SQLite (cameras, observations, tracklets, referents, logs) + vector index (FAISS flat IP, exact)
   ^                                                     ^
   | KB lookup                                           | filtered search
query -> parser (LLM + rules) -> resolver (KB / manifest) -> retriever -> fuse -> dedup -> [VLM verify] -> templated answer + evidence
   |            \-> unresolved referent -> clarification card -> KB write -> resume
   v
FastAPI -> web UI (chat, clarification card, camera wall, KB panel)
```

## 2. Repo layout
```
omniwatch/
  config.py            dataclass/pydantic settings, presets (fast|balanced|accurate)
  db.py                SQLite connection (WAL, synchronous=FULL), migrations
  ingest/decode.py     frame iterator + timestamps
  ingest/pipeline.py   orchestrates detect->track->embed->write; idempotent
  detect/detector.py   Detector interface + YOLO-World / closed-set adapters
  embed/encoder.py     image+text encoders (open_clip or transformers), batching
  embed/attrs.py       color naming, carries association
  track/tracker.py     per-camera tracking, tracklet assembly
  index/vector.py      FAISS flat IP wrapper, row-id mapping, persist/load
  query/parser.py      LLM parser + rule fallback -> ParsedQuery
  query/resolver.py    time resolution, referent resolution (KB, manifest)
  query/retrieve.py    filters, scoring, fusion, dedup
  query/rerank.py      VLM yes/no verification (P1)
  query/answer.py      deterministic templates
  kb/normalize.py      phrase normalization
  kb/store.py          referents CRUD, history, clarification log
  evidence/thumbs.py   crop/frame + bbox drawing
  evidence/clips.py    ffmpeg cut + cache
  evidence/blur.py     face blur (stretch)
  api/main.py          FastAPI app
  api/schemas.py       pydantic models
  reid/                (stretch) cross-camera linking
  alerts/              (stretch) standing queries
eval/ baseline.py run_eval.py metrics.py queries/{dev,test}.jsonl results/
web/  index.html app.js style.css
tests/  scripts/{ingest.sh,serve.sh,eval.sh,demo_reset.sh}
models/MANIFEST.md   (weights themselves are gitignored)
```

## 3. SQLite schema (DDL to implement; adjust only via PRD change)
```sql
PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;

CREATE TABLE sites(site_id TEXT PRIMARY KEY, reference_now REAL, created_at REAL);

CREATE TABLE cameras(
  site_id TEXT, camera_id TEXT, label TEXT, video_path TEXT,
  start_time REAL NOT NULL,      -- epoch seconds UTC
  native_fps REAL, width INT, height INT, duration REAL, thumb_path TEXT,
  PRIMARY KEY(site_id, camera_id));

CREATE TABLE observations(
  obs_id INTEGER PRIMARY KEY, site_id TEXT, camera_id TEXT, tracklet_id INTEGER,
  frame_idx INT, ts REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL,   -- normalized 0..1
  cls TEXT, det_conf REAL, quality REAL,
  color_primary TEXT, color_secondary TEXT, carries TEXT,        -- carries: JSON list
  emb_row INT);                                                  -- row in observation vector index
CREATE INDEX obs_cam_ts ON observations(site_id, camera_id, ts);

CREATE TABLE tracklets(
  tracklet_id INTEGER PRIMARY KEY, site_id TEXT, camera_id TEXT, local_track_id INT,
  cls TEXT, t_start REAL, t_end REAL, n_obs INT, best_obs_id INT,
  topk_obs_json TEXT, attrs_json TEXT, emb_row INT, global_id INT);   -- global_id: re-ID (stretch)
CREATE INDEX trk_cam_t ON tracklets(site_id, camera_id, t_start, t_end);

CREATE TABLE referents(
  referent_id INTEGER PRIMARY KEY, site_id TEXT, phrase_norm TEXT, phrase_display TEXT,
  kind TEXT DEFAULT 'location', targets_json TEXT NOT NULL,   -- [{"camera_id":"cam2","roi":[x1,y1,x2,y2]|null}]
  source TEXT, created_at REAL, updated_at REAL,
  UNIQUE(site_id, phrase_norm, kind));
CREATE TABLE referent_history(id INTEGER PRIMARY KEY, referent_id INT, old_targets_json TEXT, new_targets_json TEXT, changed_at REAL, actor TEXT);
CREATE TABLE clarification_log(id INTEGER PRIMARY KEY, site_id TEXT, phrase_norm TEXT, session_id TEXT, asked_at REAL, resolved_at REAL, referent_id INT);
CREATE TABLE pending_queries(session_id TEXT PRIMARY KEY, message TEXT, parsed_json TEXT, unresolved_json TEXT, created_at REAL);

CREATE TABLE query_log(id INTEGER PRIMARY KEY, ts REAL, session_id TEXT, message TEXT, parsed_json TEXT,
  mode TEXT, latency_json TEXT, result_ids_json TEXT);
CREATE TABLE standing_queries(id INTEGER PRIMARY KEY, site_id TEXT, message TEXT, parsed_json TEXT, active INT, created_at REAL);  -- stretch
CREATE TABLE alerts(id INTEGER PRIMARY KEY, standing_id INT, tracklet_id INT, fired_at REAL, payload_json TEXT);                   -- stretch
```
Vector indexes (FAISS `IndexFlatIP` over L2-normalized float32; exact search is fine at hackathon scale): `obs.index` (observation crops), `trk.index` (tracklet aggregates), optional `frame.index`. `emb_row` is the row number; files are rebuilt from SQLite + stored embeddings if lost (store embeddings as `.npy` shards alongside).

## 4. ParsedQuery schema (parser output, shared by LLM and rules)
```json
{
  "intent": "exists|find|when|count|path",
  "object_phrase": "red car",
  "object_class_hint": "vehicle|person|bag|null",
  "attributes": {"color": ["red"], "carries": [], "other": []},
  "location_phrases": ["main gate"],
  "camera_ids": [],
  "time": {"kind": "relative|absolute|none", "start": null, "end": null, "raw": "in the last hour"},
  "negation": false,
  "raw": "did a red car pass through the main gate in the last hour?"
}
```
Rules: the LLM must return JSON only, validated by pydantic; on validation failure fall back to the rule parser and log it.

## 5. Retrieval scoring (default, tunable on dev set only)
```
score(tracklet) = w_e * emb_sim(text_prompt_ensemble, tracklet_emb or top-k crops)
                + w_a * attribute_match(color, carries, class)
                + w_t * temporal_prior (optional, 0 by default)
filter: camera set (from referents/manifest), [t_start,t_end] overlap, ROI intersection, class hint
dedup: merge tracklets of same camera whose intervals overlap/touch within a gap and whose embeddings are near-identical
threshold: per intent, calibrated on dev set (no match below it)
verify (P1): top-K tracklets -> VLM yes/no on best crops -> final = a*score + b*P(yes)
```
Prompt ensemble for text: several templates ("a photo of {x}", "a surveillance image of {x}", "{x}") averaged, then normalized.

## 6. API contract
```
POST /api/chat        {session_id, message, mode?: "fast"|"verified"}
  -> {type:"answer", summary, results:[Result], parsed, latency_ms:{parse,retrieve,verify,total}, mode}
   | {type:"clarification", clarification_id, phrase, question, cameras:[{camera_id,label,thumb_url}], known_referents:[...], suggestion?}
   | {type:"no_match", summary, searched:{cameras,t_start,t_end}, parsed, latency_ms}
   | {type:"error", message}
POST /api/clarify     {session_id, clarification_id, targets:[{camera_id, roi?}], same_as?: referent_id}
  -> same shapes as /api/chat (query resumes automatically)
GET  /api/cameras     -> [{camera_id,label,thumb_url,start,end}]
GET  /api/results/{result_id}/thumb | /clip
GET  /api/kb          -> referents ; PUT /api/kb/{referent_id} ; DELETE /api/kb/{referent_id}
POST /api/ingest {manifest_path} ; GET /api/ingest/status
GET  /api/health      -> {models, fallbacks_active, index_counts, device}
Result = {result_id, camera_id, camera_label, t_start_iso, t_end_iso, t_peak_iso, score, bbox, tracklet_id,
          matched:{color, carries, class}, thumb_url, clip_url}
```

## 7. Model choices (defaults; verify availability/licence in hour 1, record in models/MANIFEST.md)
| Role | Default | Fallback |
|---|---|---|
| Open-vocab detector (index time) | YOLO-World (small) via Ultralytics, broad vocab in config | closed-set YOLO (COCO classes) for speed; OWLv2 / Grounding DINO only for the baseline or query-time verification |
| Tracker | ByteTrack (Ultralytics built-in or `supervision`) | BoT-SORT |
| Image/text encoder | open_clip ViT-B/32 or B/16 (LAION weights), or SigLIP base | smaller CLIP if CPU only |
| Query parser LLM | small local instruct model through Ollama, JSON mode | rule parser only |
| VLM verifier (P1) | small open VLM (e.g. Qwen2.5-VL-3B-Instruct class) scoring P("Yes") | CLIP-only |
| Re-ID (stretch) | torchreid OSNet for persons; CLIP + color + travel-time gating for vehicles | CLIP only |
| Face blur (stretch) | any local face detector | blur upper third of person boxes |
The names above are starting points, not guarantees. The agent must confirm each loads and record version/hash.

## 8. Clips and evidence
Thumbnails: crop of best observation with bbox drawn on the full frame (both stored). Clips: ffmpeg cut `[t_start-2s, t_end+2s]` from the source video, re-encode to H.264 for accurate seeking, cached by `(camera,start,end)`. Blur (stretch) applied to thumbs and clips before serving.

## 9. Time handling
All internal times are epoch seconds UTC. UI shows manifest-local time with offset. `reference_now` defaults to max(camera end). Relative expressions ("last hour", "5 minutes ago", "this morning") resolve against `reference_now`. If the query has no time, search the full range.
