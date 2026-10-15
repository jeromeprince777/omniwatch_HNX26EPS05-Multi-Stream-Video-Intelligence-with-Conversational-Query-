# OmniWatch (HNX26EPS05) — Architectural Flowcharts & System Schematics

Comprehensive visual engineering documentation, pipeline flowcharts, state diagrams, and repository structure for **OmniWatch: Multi-Stream Video Intelligence with Conversational Query**.

---

## 1. Repository Tree Structure

```text
omniwatch_pack/
│
├── omniwatch/                          # Core Python Engine
│   ├── api/                            # FastAPI REST & WebSocket Server (main.py, schemas.py)
│   ├── ingest/                         # Multi-Stream Ingestion Engine (decode.py, pipeline.py)
│   ├── detect/                         # Neural Detection Layer (YOLOv8s, YOLO-World)
│   ├── embed/                          # OpenCLIP ViT-L-14 Embeddings & Color/Carries Attributes
│   ├── track/                          # Per-Camera ByteTrack Kalman State Filter
│   ├── index/                          # FAISS FlatIP Exact Vector Indexing Wrapper
│   ├── query/                          # Parser, Resolver, Hybrid Retrieval, VLM Reranking
│   ├── kb/                             # Clarify-Once Spatial Memory (SQLite CRUD & Normalizer)
│   ├── evidence/                       # Keyframe Reticle Highlighter & FFmpeg 5s Clip Slicer
│   ├── alerts/                         # Standing Queries & Continuous Audit Rule Engine
│   └── reid/                           # Cross-Camera Appearance & Spatio-Temporal Linker
│
├── web/                                # Tactical Surveillance HUD (Frontend)
│   ├── index.html                      # Cyberpunk-tactical surveillance interface layout
│   ├── app.js                          # SPA engine: conversational chat, Canvas ROI drawing, Re-ID
│   └── style.css                       # Obsidian dark mode design system & micro-animations
│
├── docs/                               # Engineering Specifications & Documentation
│   ├── 00_PRD.md                       # Product Requirements Document (Priority 1)
│   ├── 01_ARCHITECTURE.md              # System Architecture & SQLite DDL
│   ├── 02_RESEARCH_AND_EVAL_PLAN.md    # Ablation study & benchmark methodology
│   ├── 03_CLARIFY_ONCE_MEMORY_SPEC.md  # Clarify-Once permanent memory specification
│   ├── 04_DATA_PLAN.md                 # Public CCTV & Campus dataset manifest
│   ├── 06_TEST_PLAN.md                 # Test matrices & verification criteria
│   ├── FLOWCHARTS.md                   # Complete architectural flowcharts (This document)
│   └── screenshots/                    # High-resolution HUD interface captures
│
├── data/                               # Dataset & Persistent Cache (Gitignored per AGENTS.md)
│   ├── public_cctv/                    # Raw multi-camera MP4 video streams
│   ├── cache/                          # Pre-rendered keyframe thumbnails & 5s MP4 clips
│   └── omniwatch.db                    # Primary SQLite WAL database
│
├── eval/                               # Evaluation & Research Ablation
│   ├── queries.json                    # Benchmark evaluation queries
│   ├── run_ablation.py                 # 4-stage ablation harness
│   └── results.md                      # Verified empirical benchmark numbers
│
├── models/                             # Checkpoints & Weights (MANIFEST.md tracked)
├── scripts/                            # Operational & Verification Scripts
├── omni_live.py                        # Real-time multi-camera tracking stream (port 8100)
├── omni_min.py                         # Minimal CLI search & query verification tool
├── start_omniwatch.bat                 # 1-Click Windows server launcher
└── config.yaml                         # Global hardware thresholds & hyperparameters
```

---

## 2. High-Level End-to-End System Flowchart

<p align="center">
  <img src="screenshots/01_search_console_tactical_hud.png" width="49%" alt="Tactical Search Console" />
  <img src="screenshots/02_tours_bag_search_dossier.png" width="49%" alt="Grounded Intelligence Report - Tours" />
</p>

```mermaid
flowchart TD
    subgraph INGESTION["1. INGESTION PIPELINE (Offline / Real-Time)"]
        V[Raw MP4 Multi-Camera Streams] --> DEC[Native Frame Decoder<br/>@ 2.0 FPS / PyAV]
        DEC --> DET[YOLOv8s Detector<br/>People, Vehicles, Bags]
        DET --> TRK[ByteTrack Tracker<br/>Kalman Filter State]
        TRK --> EMB[OpenCLIP ViT-L-14<br/>25% Padded Crop Encoder]
        TRK --> ATTR[Explicit Attributes<br/>Dominant Color + Carries Association]
        EMB --> FAISS[(FAISS FlatIP<br/>Vector Indices)]
        ATTR --> SQL[(SQLite DB - WAL Mode<br/>Observations & Tracklets)]
    end

    subgraph QUERY["2. CONVERSATIONAL QUERY PIPELINE"]
        USR([Investigator Natural Language Query]) --> PARSE[Parser Engine<br/>Structured Intent Extraction]
        PARSE -->|Regex / LLM| PQ[ParsedQuery JSON<br/>Intent, Object, Color, Loc, Time]
        PQ --> RES{Location in KB?}
        RES -->|Known Referent| QFILTER[Camera / Time / ROI Filter]
        RES -->|Unknown Referent| CLARIFY[Clarify-Once Protocol<br/>Interactive Canvas Modal]
        CLARIFY -->|User Draws ROI| KBSAVE[(Save to SQLite KB)]
        KBSAVE --> QFILTER
        
        QFILTER --> VECSEARCH[FAISS Inner-Product Search<br/>Sub-15ms Query Vector vs Crops]
        VECSEARCH --> FUSE[Attribute Fusion & Temporal Dedup<br/>Color + Spatial Bag Overlap]
        
        FUSE --> MODE{Pipeline Mode}
        MODE -->|FAST| ANS[Deterministic Answer Synthesis<br/>Camera ID + ISO Timestamp + Score]
        MODE -->|VERIFIED| VLM[SmolVLM2 Visual Reranker<br/>Top-3 Candidate Verification]
        VLM --> ANS
    end

    subgraph EVIDENCE["3. EVIDENCE DOSSIER & UI"]
        ANS --> HUD[Tactical Surveillance Web HUD<br/>Grounded Result Dossier]
        HUD -->|View Keyframe| RETICLE[Cyber Reticle Target Frame<br/>Exact Native Bounding Box]
        HUD -->|Play Clip| CLIP[On-Demand 5s MP4 Clip<br/>Multi-Speed Playback 0.5x - 2.0x]
        HUD -->|Trace Path| REID[Cross-Camera Re-ID Timeline<br/>Multi-Hop Transit Graph]
    end

    SQL -.-> VECSEARCH
    FAISS -.-> VECSEARCH
    SQL -.-> RETICLE
```

---

## 3. Video Ingestion & Multi-Object Tracking Pipeline

<p align="center">
  <img src="screenshots/03_camera_wall_grid.png" width="98%" alt="Camera Wall Grid" />
</p>

```mermaid
flowchart LR
    A[MP4 Stream] --> B[OpenCV / PyAV Decoder]
    B -->|Native Timestamp| C[Frame Iterator]
    
    subgraph DETECTION["Detection & Spatial Parsing"]
        C --> D[YOLOv8s Closed-Set Detector]
        D -->|Bounding Boxes| E[ByteTrack MOT]
        E -->|Tracklet IDs| F[Observation Slicer]
    end
    
    subgraph EMBEDDING["Visual Feature Extraction"]
        F -->|25% Padded Crop| G[OpenCLIP ViT-L-14 Encoder]
        G -->|512-d L2 Normalized| H[Crop Embedding]
        F -->|HSV / Spatial Overlap| I[Dominant Color Extraction]
        F -->|Bbox IOU Intersection| J[Carries: Bag / Backpack Association]
    end
    
    subgraph STORAGE["Durable SQLite Storage"]
        H --> K[(obs.index / trk.index)]
        I --> L[(SQLite observations table)]
        J --> L
        E --> M[(SQLite tracklets table)]
    end
```

---

## 4. Conversational NL Query Parsing & Hybrid Retrieval

<p align="center">
  <img src="screenshots/06_evidence_keyframe_reticle_modal.png" width="49%" alt="Evidence Reticle Target" />
  <img src="screenshots/07_evidence_video_clip_modal.png" width="49%" alt="5s Evidence Video Player" />
</p>

```mermaid
sequenceDiagram
    autonumber
    actor Investigator as Operator / Investigator
    participant HUD as Tactical Web HUD
    participant API as FastAPI /api/chat
    participant Parser as Intent Parser (LLM/Rules)
    participant KB as Clarify-Once KB
    participant FAISS as FAISS Vector Index
    participant SQLite as SQLite WAL Engine
    participant VLM as SmolVLM2 Reranker

    Investigator->>HUD: "Show me a person carrying a bag at Tours plaza"
    HUD->>API: POST /api/chat {message, session_id, mode}
    API->>Parser: parse_query(message)
    Parser-->>API: ParsedQuery {intent: 'find', class: 'person', carries: ['bag'], loc: 'Tours'}
    
    API->>KB: resolve_referents("Tours")
    KB-->>API: Camera ID: "Tours", ROI: Full Camera
    
    API->>FAISS: Search vector similarity (text_emb · crop_emb)
    FAISS-->>API: Top-K Observation IDs
    
    API->>SQLite: Fetch metadata, bounding boxes, carries attributes
    SQLite-->>API: Observations with carries=['bag', 'backpack']
    
    API->>API: Score Fusion: 0.65 * SIM + 0.20 * COLOR + 0.15 * CARRIES
    
    opt Mode == VERIFIED
        API->>VLM: Verify Top-3 Crops ("Does this person carry a bag?")
        VLM-->>API: Verification Score (0.91)
    end
    
    API-->>HUD: Grounded Dossier {camera: "Tours", timestamp: "09:05:03", score: 0.891, clip_url}
    HUD->>Investigator: Render Grounded Result Card + Keyframe Reticle + Play Clip
```

---

## 5. Clarify-Once Spatial Memory Protocol (State Machine)

<p align="center">
  <img src="screenshots/05_clarify_once_roi_canvas_modal.png" width="49%" alt="Interactive ROI Canvas Modal" />
  <img src="screenshots/04_clarify_once_spatial_memory.png" width="49%" alt="Clarify-Once Knowledge Base Table" />
</p>

```mermaid
stateDiagram-v2
    [*] --> QueryReceived: User submits natural-language query
    QueryReceived --> ParseLocation: Extract location phrases
    
    ParseLocation --> CheckKB: Check phrase in SQLite referents
    CheckKB --> QueryResolved: Phrase exists in KB (Learned previously)
    
    CheckKB --> UnknownReferent: Phrase not found in KB
    UnknownReferent --> TriggerClarify: Pause query, emit Clarify-Once modal
    
    state TriggerClarify {
        [*] --> DisplayCameras: Render 4 camera snapshots
        DisplayCameras --> SelectCamera: Operator clicks camera feed
        SelectCamera --> DrawROI: (Optional) Drag rectangle on HTML5 Canvas
        DrawROI --> AssignAlias: (Optional) Select 'Same as known referent'
        AssignAlias --> ConfirmBinding: Click 'CONFIRM & RESUME'
    }
    
    ConfirmBinding --> PersistSQLite: INSERT into referents (WAL mode)
    PersistSQLite --> QueryResolved: Auto-resume original query
    QueryResolved --> [*]: Return grounded result dossier
```

---

## 6. Continuous Standing Surveillance Alerts Engine

<p align="center">
  <img src="screenshots/08_standing_alerts_audit_feed.png" width="98%" alt="Standing Alerts Audit Feed" />
</p>

```mermaid
graph TD
    RULE[Operator Registers Standing Rule<br/>e.g. 'Alert if person enters east corridor'] --> MONITORS[(SQLite standing_queries table)]
    INGEST[Video Ingestion / Live Stream Feed] --> DETECT[Observation Tracklet Created]
    DETECT --> EVAL{Matches Active Monitor?}
    EVAL -->|Yes| ALERT[Emit Alert Event & Write to SQLite alerts table]
    ALERT --> FEED[Real-Time Audit Event Feed & Audio Chime]
    FEED --> PLAY[Operator Clicks 'PLAY CLIP' -> 5s Forensic Verification]
```

---

## 7. Requirements Traceability Matrix

| Requirement ID | Module | Implementation Description | Verification |
|---|---|---|---|
| **FR-ING-1** | `omniwatch/ingest/pipeline.py` | Ingests from multi-camera `manifest.yaml` | `scripts/check_env.py` |
| **FR-ING-2** | `omniwatch/ingest/decode.py` | Decodes at configurable sample FPS (2.0) with exact native frame indices | `scripts/debug_bbox.py` |
| **FR-DET-1** | `omniwatch/detect/detector.py` | Open-vocabulary & closed-set YOLOv8 detector for people, cars, bags | `scripts/contact_sheet.py` |
| **FR-DET-2** | `omniwatch/embed/encoder.py` | OpenCLIP ViT-L-14 / ViT-B-32 512-d normalized embeddings on 25% padded crops | Unit tests |
| **FR-DET-3** | `omniwatch/embed/attrs.py` | Dominant HSV color classification and person-bag bounding box carrier overlap | `tests/test_attrs.py` |
| **FR-IDX-1** | `omniwatch/track/tracker.py` | Per-camera ByteTrack Kalman multi-object tracking | Integration tests |
| **FR-IDX-2** | `omniwatch/index/vector.py` | FAISS FlatIP exact inner-product vector indexing over normalized float32 | `tests/test_vector.py` |
| **FR-QRY-1** | `omniwatch/query/parser.py` | Natural-language query parsing with regex rules & local LLM slot extraction | `tests/test_parser.py` |
| **FR-KB-1..6** | `omniwatch/kb/store.py` | Clarify-Once spatial memory protocol: single modal query, durable SQLite storage, restart survival | Interactive UI test |
| **FR-EVD-1..2** | `omniwatch/evidence/` | Native keyframe reticle highlights + on-demand 5s FFmpeg video clips | Web HUD inspection |
| **FR-UI-1..4** | `web/app.js` | Full tactical surveillance HUD, multi-cam wall, KB editor, standing alert triggers | Browser subagent |
| **FR-EVAL-1..3** | `eval/run_ablation.py` | 4-stage empirical ablation benchmark against baseline on held-out queries | `eval/results.md` |
