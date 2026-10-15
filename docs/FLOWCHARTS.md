# OmniWatch (HNX26EPS05) — Architectural Flowcharts & System Schematics

Comprehensive visual engineering documentation, pipeline flowcharts, state diagrams, and repository structure for **OmniWatch: Multi-Stream Video Intelligence with Conversational Query**.

---

## 1. Repository Tree Structure

```text
omniwatch_pack/
│
├── omniwatch/                          # Core Python Engine
│   ├── api/                            # FastAPI REST & WebSocket Server
│   │   ├── main.py                     # App entry point, REST routes, static HUD mount
│   │   └── schemas.py                  # Pydantic v2 request/response contracts
│   ├── ingest/                         # Multi-Stream Ingestion Engine
│   │   ├── decode.py                   # PyAV / OpenCV native frame decoder (@ 2 FPS)
│   │   └── pipeline.py                 # Idempotent batch ingestion orchestrator
│   ├── detect/                         # Neural Detection Layer
│   │   └── detector.py                 # YOLOv8s / YOLO-World open-vocab adapters
│   ├── embed/                          # Vision-Language Embedding & Feature Extraction
│   │   ├── encoder.py                  # OpenCLIP ViT-L-14 / ViT-B-32 image/text embeddings
│   │   └── attrs.py                    # Dominant color clustering & person-bag spatial carrier
│   ├── track/                          # Multi-Object Tracking (MOT)
│   │   └── tracker.py                  # Per-camera ByteTrack Kalman state filter
│   ├── index/                          # Vector Indexing
│   │   └── vector.py                   # FAISS FlatIP L2-normalized vector wrapper
│   ├── query/                          # Conversational Query Pipeline
│   │   ├── parser.py                   # Intent extraction (LLM + Deterministic Regex fallback)
│   │   ├── resolver.py                 # Clarify-Once KB & reference_now temporal resolver
│   │   ├── retrieve.py                 # Multi-stage filtered vector search & attribute fusion
│   │   ├── rerank.py                   # Deep visual verification (SmolVLM2 / OpenCLIP)
│   │   └── answer.py                   # Deterministic, zero-hallucination grounded templates
│   ├── kb/                             # Spatial Memory Management
│   │   ├── normalize.py                # Token & phrase normalization
│   │   └── store.py                    # SQLite durable referents CRUD & audit logger
│   ├── evidence/                       # Forensic Evidence Generation
│   │   ├── thumbs.py                   # Native keyframe crop & cyber reticle renderer
│   │   └── clips.py                    # On-demand FFmpeg 5-second video slicer & cache
│   ├── alerts/                         # Continuous Surveillance Rule Engine
│   │   └── monitor.py                  # Standing query rule evaluation & audit table writer
│   ├── reid/                           # Cross-Camera Association
│   │   └── linker.py                   # Spatio-temporal transit & appearance feature linker
│   ├── config.py                       # Pydantic / dataclass settings & presets
│   └── db.py                           # SQLite WAL connection & schema migrations
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
├── models/                             # Checkpoints & Weights
│   └── MANIFEST.md                     # Model manifest (name, hash, license, source)
│
├── scripts/                            # Operational & Verification Scripts
│   ├── check_env.py                    # CUDA & environment validation
│   ├── contact_sheet.py                # Visual bounding-box contact sheet generator
│   └── debug_bbox.py                   # Frame-box coordinate alignment inspector
│
├── omni_live.py                        # Real-time multi-camera tracking stream (port 8100)
├── omni_min.py                         # Minimal CLI search & query verification tool
├── start_omniwatch.bat                 # 1-Click Windows server launcher
└── config.yaml                         # Global hardware thresholds & hyperparameters
```

---

## 2. High-Level End-to-End System Flowchart

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

## 6. Dual Pipeline Modes: Fast vs Verified Architecture

```mermaid
graph TD
    Q[User Natural Language Query] --> P[Query Parser]
    P --> R[KB Location Resolver]
    R --> S[Vector Candidate Retrieval]
    
    subgraph DUAL_MODES["Pipeline Execution Modes"]
        S -->|Mode: FAST| FAST_PATH[Direct Attribute Fusion<br/>& Temporal Smoothing]
        FAST_PATH -->|Latency: sub-25ms| OUT_FAST[Grounded Result Dossier<br/>Source: YOLOv8s + OpenCLIP]
        
        S -->|Mode: VERIFIED| VLM_PATH[SmolVLM2 Visual Verification<br/>Top-3 Candidate Reranking]
        VLM_PATH -->|Deep Cross-Modal VQA| OUT_VERIFIED[Verified Intelligence Dossier<br/>Source: SmolVLM2 Reranked]
    end
    
    OUT_FAST --> HUD[Surveillance HUD]
    OUT_VERIFIED --> HUD
```

---

## 7. Cross-Camera Spatio-Temporal Re-ID & Timeline Graph

```mermaid
flowchart TD
    T1[Observation Tracklet #376<br/>Camera: Cam 1 - North Gate Entry<br/>Time: 09:00:00 -> 09:00:15 UTC]
    
    T2[Observation Tracklet #388<br/>Camera: Cam 2 - East Perimeter Corridor<br/>Time: 09:03:33 -> 09:03:48 UTC]
    
    T3[Observation Tracklet #108<br/>Camera: Tours Pedestrian Plaza<br/>Time: 09:05:03 -> 09:05:22 UTC]
    
    T1 -->|Δt: 198.0s Transit<br/>Cosine Similarity: 0.892| T2
    T2 -->|Δt: 75.0s Transit<br/>Cosine Similarity: 0.884| T3
    
    subgraph SUMMARY["RECONSTRUCTED CHRONOLOGICAL TRAJECTORY"]
        T1
        T2
        T3
    end
```

---

## 8. Requirements Traceability Matrix

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
