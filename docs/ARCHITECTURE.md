# Architecture Specification — OmniWatch (HNX26EPS05)

## 1. Executive Architecture Overview

OmniWatch is an offline, multi-camera tactical video intelligence engine designed for conversational search over recorded multi-camera CCTV footage. All answers are grounded in database evidence rows with exact camera IDs, timestamps, bounding box thumbnails, and playable video clips.

```
+--------------------------------------------------------------------------------------------------+
|                                    RECORDED MULTI-CAMERA CCTV                                    |
|                               (manifest.yaml / MP4 video streams)                               |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     VIDEO DECODER & METADATA      |
                               | (PyAV / OpenCV / Motion Gating)   |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |  OPEN-VOCABULARY OBJECT DETECTOR  |
                               |    (YOLO-World-S / Ultralytics)   |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |    MULTI-OBJECT TRACKER (SPATIAL) |
                               |      (ByteTrack Kalman Filter)    |
                               +-----------------------------------+
                                                 |
                        +------------------------+------------------------+
                        |                                                 |
                        v                                                 v
         +-----------------------------+                   +-----------------------------+
         |    VISION-LANGUAGE ENCODER  |                   |   EXPLICIT ATTRIBUTES       |
         |    (OpenCLIP ViT-B-32)      |                   | (Color histogram & carries) |
         +-----------------------------+                   +-----------------------------+
                        |                                                 |
                        +------------------------+------------------------+
                                                 |
                                                 v
                        +-------------------------------------------------+
                        |              PERSISTENT STORAGE                 |
                        |  SQLite 3 (WAL mode, synchronous=FULL)          |
                        |  FAISS Flat IP (Exact Cosine Vector Index)      |
                        +-------------------------------------------------+
                                                 ^
                                                 |  Grounded Search
                                                 |
+--------------------------------------------------------------------------------------------------+
|                                    CONVERSATIONAL QUERY ENGINE                                   |
+--------------------------------------------------------------------------------------------------+
|  Natural Language Query: "Did a red car pass through the main gate in the last hour?"            |
|                                                                                                  |
|  1. Deterministic Rule / LLM Parser -> ParsedQuery (intent, object, attrs, location, time)       |
|  2. Clarify-Once Memory Resolver -> Checks SQLite referents. If unknown -> Clarification Card   |
|  3. Spatio-Temporal & Camera Filtering -> Bounds candidate universe in SQLite                   |
|  4. Vector Search & Candidate Retrieval -> FAISS exact cosine similarity over tracklet embeddings|
|  5. Explicit Attribute Fusion -> score = w_e * emb_sim + w_a * attr_match + w_t * temporal_prior |
|  6. Candidate Verification (P1) -> SmolVLM2 / OpenCLIP crop verification                         |
|  7. Deterministic Answer Construction -> Templated from evidence row (No LLM in answer path!)    |
|  8. Evidence Packaging -> Dynamic ffmpeg clip cutting + bbox overlaid thumbnail                  |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |       FASTAPI REST & WEBSOCKET    |
                               |  (/api/chat, /api/clarify, etc.)  |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     TACTICAL SURVEILLANCE HUD     |
                               |   (Search, Wall, KB, Player)      |
                               +-----------------------------------+
```

---

## 2. Component Breakdown

### 2.1 Video Ingestion & Decoder (`omniwatch/ingest/`)
- **Manifest-Driven Ingestion**: Configured via `manifest.yaml` (site, camera IDs, labels, video filepaths, UTC start timestamps).
- **Frame Decoding**: Decodes frames at configurable sample FPS (default 2 FPS) with optional frame-difference motion gating to prevent redundant processing of idle footage.
- **Timestamp Integrity**: Preserves exact wall-clock UTC time: `ts = camera_start_time + frame_pts`.
- **Idempotency**: Existing cameras in SQLite are skipped unless `--force` is specified.
- **Status Telemetry**: Live progress persisted to `data/ingest_status.json` and exposed at `GET /api/ingest/status`.

### 2.2 Open-Vocabulary Detection & Tracking (`omniwatch/detect/`, `omniwatch/track/`)
- **Detector**: YOLO-World-S (`models/yolov8s-world.pt`), an open-vocabulary object detector operating over configurable classes (`person`, `car`, `truck`, `bus`, `bicycle`, `motorcycle`, `backpack`, `handbag`, `suitcase`).
- **Tracker**: ByteTrack Kalman filtering per camera stream. Merges temporally continuous bounding boxes into `tracklets` with defined start/end timestamps, duration, and key observation crops.

### 2.3 Vision-Language Embedding & Attribute Extraction (`omniwatch/embed/`)
- **Encoder**: OpenCLIP ViT-B-32 trained on LAION-2B (`open_clip_model.safetensors`). Computes 512-dimensional L2-normalized embeddings for both text prompts and visual crops.
- **Explicit Color Attributes**: 11-bin HSV histogram classifier identifying primary and secondary dominant colors (`red`, `blue`, `green`, `yellow`, `black`, `white`, `gray`, `silver`, etc.).
- **Carrying Association**: Spatial overlap/containment heuristic linking bag detections (`backpack`, `handbag`, `suitcase`) to person tracklets.

### 2.4 Indexing & Storage Engine (`omniwatch/db.py`, `omniwatch/index/`)
- **SQLite Database**: Single source of truth. Configured with `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=FULL;` for crash safety and persistence.
- **Vector Index**: FAISS `IndexFlatIP` performing exact inner product search over L2-normalized float32 embeddings. Maintained alongside raw `.npy` arrays for instantaneous zero-loss index reconstruction.

### 2.5 Query Parser & Resolver (`omniwatch/query/parser.py`, `omniwatch/query/resolver.py`)
- **Deterministic Regex Parser**: Fast rule-based decomposition extracting:
  - Intent (`exists`, `find`, `when`, `count`, `path`)
  - Object phrase & class hint
  - Explicit attributes (colors, carries)
  - Location phrases (referents)
  - Structured temporal constraints (`relative`, `absolute`, `none`)
- **LLM Fallback**: Local Ollama `llama3.2:1b` structured JSON fallback when regex patterns cannot resolve syntax.
- **Time Normalization**: Evaluated against `reference_now` (end of latest video footage) to ensure offline reproducibility across timezones.

### 2.6 Clarify-Once Memory System (`omniwatch/kb/`)
- **Referent Resolution**: Matches location phrases against the SQLite `referents` table using case-insensitive whitespace/punctuation normalization.
- **Clarification State Machine**:
  1. If an unknown referent is encountered, query halts without guessing.
  2. A clarification card is returned (`type: "clarification"`) containing available camera thumbnails and an ROI drawing interface.
  3. User maps the referent via `POST /api/clarify`.
  4. Mapping is durably committed to SQLite `referents` and audited in `referent_history`.
  5. The query automatically resumes and returns evidence.
  6. On all subsequent queries (and after backend restart), the referent resolves silently.

### 2.7 Retrieval & Verification Pipeline (`omniwatch/query/retrieve.py`, `omniwatch/query/rerank.py`)
- **Stage 1 (Filter)**: Spatial-temporal filtering by camera ID and time interval.
- **Stage 2 (Vector Search)**: Prompt ensemble cosine similarity (`"a surveillance image of {x}"`, `"a photo of {x}"`, `"{x}"`).
- **Stage 3 (Attribute Fusion)**: Multi-modal fusion scoring:
  $$\text{score} = w_e \cdot \text{sim}_{\text{emb}} + w_a \cdot \text{score}_{\text{attr}} + w_t \cdot \text{score}_{\text{temp}}$$
- **Stage 4 (Verification Pass)**: Top-3 candidates verified with crop visual scoring.

### 2.8 Deterministic Grounding & Evidence Generation (`omniwatch/query/answer.py`, `omniwatch/evidence/`)
- **Grounded Answer Guarantee**: Answer text is deterministically interpolated from database rows. No LLM touches the answer string.
- **Evidence Frame**: Key observation crop with rendered target bounding box.
- **Evidence Clip**: On-demand ffmpeg extraction of a 10-second contextual video clip ($[t_{\text{start}} - 2\text{s}, t_{\text{end}} + 2\text{s}]$), cached in `data/cache/clips/`.

---

## 3. SQLite Database Schema

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE sites (
    site_id TEXT PRIMARY KEY,
    reference_now REAL,
    created_at REAL
);

CREATE TABLE cameras (
    site_id TEXT,
    camera_id TEXT,
    label TEXT,
    video_path TEXT,
    start_time REAL NOT NULL,
    native_fps REAL,
    width INT,
    height INT,
    duration REAL,
    thumb_path TEXT,
    PRIMARY KEY(site_id, camera_id)
);

CREATE TABLE observations (
    obs_id INTEGER PRIMARY KEY,
    site_id TEXT,
    camera_id TEXT,
    tracklet_id INTEGER,
    frame_idx INT,
    ts REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    cls TEXT,
    det_conf REAL,
    quality REAL,
    color_primary TEXT,
    color_secondary TEXT,
    carries TEXT,
    emb_row INT
);
CREATE INDEX obs_cam_ts ON observations(site_id, camera_id, ts);

CREATE TABLE tracklets (
    tracklet_id INTEGER PRIMARY KEY,
    site_id TEXT,
    camera_id TEXT,
    local_track_id INT,
    cls TEXT,
    t_start REAL,
    t_end REAL,
    n_obs INT,
    best_obs_id INT,
    topk_obs_json TEXT,
    attrs_json TEXT,
    emb_row INT,
    global_id INT
);
CREATE INDEX trk_cam_t ON tracklets(site_id, camera_id, t_start, t_end);

CREATE TABLE referents (
    referent_id INTEGER PRIMARY KEY,
    site_id TEXT,
    phrase_norm TEXT,
    phrase_display TEXT,
    kind TEXT DEFAULT 'location',
    targets_json TEXT NOT NULL,
    source TEXT,
    created_at REAL,
    updated_at REAL,
    UNIQUE(site_id, phrase_norm, kind)
);

CREATE TABLE referent_history (
    id INTEGER PRIMARY KEY,
    referent_id INT,
    old_targets_json TEXT,
    new_targets_json TEXT,
    changed_at REAL,
    actor TEXT
);

CREATE TABLE clarification_log (
    id INTEGER PRIMARY KEY,
    site_id TEXT,
    phrase_norm TEXT,
    session_id TEXT,
    asked_at REAL,
    resolved_at REAL,
    referent_id INT
);

CREATE TABLE pending_queries (
    session_id TEXT PRIMARY KEY,
    message TEXT,
    parsed_json TEXT,
    unresolved_json TEXT,
    created_at REAL
);

CREATE TABLE query_log (
    id INTEGER PRIMARY KEY,
    ts REAL,
    session_id TEXT,
    message TEXT,
    parsed_json TEXT,
    mode TEXT,
    latency_json TEXT,
    result_ids_json TEXT
);

CREATE TABLE standing_queries (
    id INTEGER PRIMARY KEY,
    site_id TEXT,
    message TEXT,
    parsed_json TEXT,
    active INT,
    created_at REAL
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    standing_id INT,
    tracklet_id INT,
    fired_at REAL,
    payload_json TEXT
);
```

---

## 4. API Endpoints Contract

| Endpoint | Method | Input Payload | Output Summary |
|---|:---:|---|---|
| `/api/chat` | `POST` | `{"session_id": str, "message": str, "mode": "fast" \| "verified"}` | Grounded answer dossier, clarification card, or no-match notice |
| `/api/clarify` | `POST` | `{"session_id": str, "clarification_id": int, "targets": [{"camera_id": str, "roi": [x1,y1,x2,y2]}]}` | Commits referent to SQLite and auto-resumes query execution |
| `/api/cameras` | `GET` | None | List of registered cameras with labels, durations, thumbnails |
| `/api/kb` | `GET` | None | List of all persisted Clarify-Once spatial bindings |
| `/api/kb/{id}` | `PUT` | `{"targets": [...]}` | Update an existing spatial referent mapping |
| `/api/kb/{id}` | `DELETE`| None | Delete referent mapping from SQLite |
| `/api/results/{id}/thumb` | `GET` | None | JPEG image crop with rendered bounding box |
| `/api/results/{id}/clip` | `GET` | None | MP4 playable video clip cut on-demand via ffmpeg |
| `/api/reid/path/{id}` | `POST` | None | Cross-camera trajectory timeline across spatial hops |
| `/api/alerts/standing` | `POST` | `{"site_id": str, "message": str}` | Registers standing surveillance rule for background alerting |
| `/api/health` | `GET` | None | Hardware, loaded models, active fallbacks, index counts |
