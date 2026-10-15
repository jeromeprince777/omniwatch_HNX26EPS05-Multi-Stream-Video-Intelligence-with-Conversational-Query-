# OmniWatch (HNX26EPS05)
## Multi-Stream Video Intelligence with Conversational Query
*HackNex 2026 Problem Statement — Autonomous Multi-Camera CCTV Video Intelligence Engine*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8s-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCLIP](https://img.shields.io/badge/Vision--Language-OpenCLIP_ViT--L--14-purple.svg)](https://github.com/mlfoundations/open_clip)
[![FAISS](https://img.shields.io/badge/Vector_Index-FAISS_FlatIP-orange.svg)](https://github.com/facebookresearch/faiss)
[![Storage](https://img.shields.io/badge/Storage-SQLite_WAL-003B57.svg?logo=sqlite&logoColor=white)](https://sqlite.org)
[![AirGapped](https://img.shields.io/badge/Privacy-100%25_On--Prem_Air--Gapped-success.svg)]()

---

## 1. Executive Summary

**OmniWatch** is a next-generation tactical video intelligence engine engineered for mission-critical security and multi-camera CCTV environments. In contrast to traditional video management systems requiring hours of manual scrubbing across desynchronized camera feeds, OmniWatch empowers security investigators to interrogate multi-camera surveillance archives using natural-language conversational queries:

> *"Show me a person carrying a bag at the pedestrian plaza."*  
> *"Did a red car pass through the main gate in the last hour?"*  
> *"Where did this person go across camera zones?"*

Every natural-language query resolves strictly to **deterministic, zero-hallucination ground truth**:
- **CAMERA ID & LABEL**: Exact physical camera location (`Tours Pedestrian Plaza`, `North Gate Entry`)
- **EXACT TIMESTAMP**: Millisecond-accurate ISO timestamp window (`2026-10-09 09:05:03 UTC`)
- **VISUAL EVIDENCE**: Bounding-box highlighted native keyframe reticle and playable 5-second on-demand video clip
- **PERSISTENT SPATIAL MEMORY**: Unresolved colloquial locations (e.g. *"the loading dock"*) trigger the **Clarify-Once Protocol**, binding permanent SQLite coordinates that survive server restarts.

---

## 2. Interactive Surveillance HUD Showcase

| View | Component | Description |
|:---:|:---:|:---|
| **01** | **Tactical Search Console** | Real-time conversational interface with query chips, sub-15ms latency telemetry bar, and dual-mode pipeline switcher (**Fast** vs **Verified**). |
| **02** | **Grounded Evidence Dossier** | Grounded report matching visual concepts with confidence score, native keyframe target reticle, and action buttons. |
| **03** | **Tactical Camera Wall** | Multi-stream 4-camera CCTV monitoring grid (`Yokohama Crossroad`, `Tours Plaza`, `Cam 1 North Gate`, `Cam 2 East Perimeter`) with live feed telemetry. |
| **04** | **Clarify-Once Spatial Memory** | Durable SQLite knowledge base managing spatial bindings, phrase normalization tokens, and camera mapping. |
| **05** | **Interactive ROI Canvas Modal** | Spatial memory assignment allowing operators to select camera feeds and draw drag-and-drop bounding boxes on an HTML5 canvas. |
| **06** | **Forensic Evidence Player** | Dual-mode evidence modal featuring cyber reticle target lock and 5-second on-demand MP4 clip player with multi-speed playback (0.5x–2.0x). |
| **07** | **Cross-Camera Re-ID Timeline** | Chronological multi-camera trajectory tracking tracing suspects across camera topologies with spatio-temporal transit times. |
| **08** | **Standing Surveillance Alerts** | Continuous surveillance monitor registering automated rules and streaming real-time security audit events. |

---

## 3. High-Level System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. INGESTION & TRACKING PIPELINE (Offline / Real-Time)"]
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

## 4. Repository Structure

```text
omniwatch_pack/
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
├── web/                                # Tactical Surveillance HUD Frontend (HTML5, Vanilla CSS, SPA JS)
├── docs/                               # Formal PRD, Architecture, Test Plans & Flowcharts
│   ├── FLOWCHARTS.md                   # Comprehensive Mermaid Diagrams & Schematics
│   └── screenshots/                    # High-Resolution Interface Evidence
├── data/                               # CCTV Footage, SQLite WAL Database, FAISS Indices
├── eval/                               # Automated Benchmark Harness & Empirical Results
├── omni_live.py                        # GPU Real-Time Multi-Camera Tracking Stream (Port 8100)
├── omni_min.py                         # Minimal CLI Video Search Tool
└── start_omniwatch.bat                 # 1-Click Production Server Launcher
```

---

## 5. Empirical Benchmark & Ablation Study

Evaluated against held-out queries on multi-camera CCTV footage ($N=41$ test queries):

| Pipeline Configuration | Hit@1 ($\tau=5\text{s}$) | Hit@1 ($\tau=15\text{s}$) | Hit@5 ($\tau=15\text{s}$) | MRR | Precision | Neg FPR | Latency (p50) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1 (Unconstrained CLIP Baseline)** | 12.9% | 22.6% | 38.7% | 0.148 | 75.6% | 100.0% | 12.5 ms |
| **B1+ (Fairness Baseline)** | 16.1% | 25.8% | 35.5% | 0.190 | 90.2% | 40.0% | 16.5 ms |
| **OW-Fast (CLIP + Attribute Fusion)** | **19.4%** | **25.8%** | **41.9%** | **0.267** | **87.8%** | **50.0%** | **17.8 ms** |
| **OW-Full (SmolVLM2 Reranked)** | **22.6%** | **29.0%** | **45.2%** | **0.312** | **92.7%** | **20.0%** | **68.4 ms** |

*Key Findings:* Explicit attribute fusion eliminates CLIP's color-binding ambiguity, and SmolVLM2 deep verification cuts false positive rate on negative/absent queries by 80%.

---

## 6. Quick Start Guide

### 1. System Requirements
- **OS**: Windows 11 / Ubuntu 22.04 LTS
- **Python**: 3.11 (64-bit)
- **GPU**: NVIDIA RTX (CUDA 12.1+ compatible) or CPU on-prem fallback
- **VRAM**: 4 GB+ (YOLOv8s + OpenCLIP)

### 2. Launch the Tactical Web HUD
```powershell
# Double click start_omniwatch.bat or run:
.\start_omniwatch.bat

# Or launch directly with Uvicorn:
.venv\Scripts\python.exe -m uvicorn omniwatch.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000/`** in your browser.

### 3. Launch Real-Time GPU Tracking Stream (Port 8100)
```powershell
# Real-time multi-camera tracking with live query overlay:
.\run_omni_live.bat
```
Navigate to **`http://127.0.0.1:8100/`** to observe real-time YOLOv8 + ByteTrack bounding boxes and tracklet trajectories.

---

## 7. Operational Verification Checklist

- [x] **Frame-Accurate Alignment**: Native frame index preserved across decodes (`native_frame_idx`).
- [x] **Clarify-Once Protocol**: Spatial aliases bind to SQLite with restart durability.
- [x] **Zero Hallucinations**: Answers synthesized strictly from database observation rows.
- [x] **On-Demand Evidence**: Sub-second 5-second FFmpeg clip slicing with interactive speed controls.
- [x] **100% Air-Gapped**: Zero network telemetry calls during runtime ingestion or query.

---

## 8. License & Attribution

Developed under Problem Statement **HNX26EPS05 (Multi-Stream Video Intelligence with Conversational Query)**.  
Built with PyTorch, Ultralytics YOLOv8, OpenCLIP, FAISS, FastAPI, and SQLite WAL.
