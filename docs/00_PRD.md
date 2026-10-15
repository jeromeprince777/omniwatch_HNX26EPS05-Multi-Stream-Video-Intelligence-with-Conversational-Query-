# 00 — Product Requirements Document: OmniWatch (HNX26EPS05)

## 1. Summary
OmniWatch ingests recorded footage from multiple non-overlapping CCTV cameras, builds a searchable index of objects/events, and answers natural-language questions through a chat UI. Every answer resolves to **camera + timestamp + visual evidence (thumbnail with bbox and a short clip)**. Unknown referents ("the main gate") trigger a **single clarification** whose answer is stored permanently.

## 2. Hard constraints from the problem statement
- Recorded multi-camera footage is enough for qualification (live is stretch).
- Natural-language query -> correct camera + timestamp + clip, beating a team-chosen baseline on held-out queries.
- Clarify-once memory that survives restarts.
- Short write-up with a comparison/ablation.
- Judges bring **unseen footage and queries with known answers**: ingestion of new footage must be fast and must not depend on anything tuned to our own footage.

## 3. Goals / non-goals
**P0:** FR-ING, FR-DET, FR-IDX, FR-QRY, FR-KB, FR-EVD, FR-UI, FR-EVAL.
**P1 (after P0 DoD):** VLM verification (needed for the research contribution), latency modes.
**P2 stretch:** live ingestion, cross-camera re-ID + path timeline, standing queries/alerts, privacy (face blur, on-prem-only).
**Non-goals:** user accounts, cloud deployment, training a foundation model, audio.

## 4. User stories
- *Operator:* "Did a red car pass through the main gate in the last hour?" -> yes/no, which camera, when, thumbnail, clip.
- *First use of a name:* system does not know "main gate", asks once, I click a camera (optionally draw the region), it never asks again, even after restart.
- *Judge:* sees why a result matched (matched attributes, score) and can open the clip.
- *(Stretch)* "Where did this person go?" -> path across cameras. "Alert me if anyone enters this zone after 8 pm."

## 5. Functional requirements
**Ingestion (FR-ING)**
- FR-ING-1 Read `manifest.yaml` (site, cameras, video path, absolute start time, optional label). See `04_DATA_PLAN.md`.
- FR-ING-2 Decode each camera at a configurable sample FPS (default 2) with optional motion gating. Timestamp = start_time + frame_time.
- FR-ING-3 Ingest cameras in parallel where hardware allows; progress at `/api/ingest/status`.
- FR-ING-4 Idempotent: re-running on the same manifest does not duplicate rows.

**Detection and embedding (FR-DET)**
- FR-DET-1 Per sampled frame, boxes for people, vehicles, bags and other configurable types (open-vocabulary detector with a broad configurable vocabulary by default; closed-set detector selectable by config for speed).
- FR-DET-2 Crop each box, compute a vision-language embedding (CLIP/SigLIP family), L2-normalized.
- FR-DET-3 Explicit attributes per crop: dominant color name(s); for persons, "carries" (bag-class boxes overlapping/adjoining the person box).
- FR-DET-4 (P1) Frame-level embedding every N seconds as fallback for scene-level queries.

**Tracking and indexing (FR-IDX)**
- FR-IDX-1 Per-camera multi-object tracking (ByteTrack-class) -> tracklets with start/end time.
- FR-IDX-2 Tracklet embedding = quality-weighted aggregate of crop embeddings; keep top-k crops for evidence and verification.
- FR-IDX-3 Filter by camera set / time window / ROI before vector search.
- FR-IDX-4 Everything persisted (SQLite + vector files); restart loads without re-ingest.

**Query (FR-QRY)**
- FR-QRY-1 Parse NL query into: intent (exists/find/when/count/path), object phrase, attributes, location phrases, time constraint, negation. LLM parser with rule-based fallback, same output schema.
- FR-QRY-2 Time expressions resolve against `reference_now` (default = end of latest footage), not wall clock. Absolute times honored.
- FR-QRY-3 Retrieval = camera/time/ROI filter -> embedding similarity -> attribute fusion -> temporal dedup -> (P1) VLM verification.
- FR-QRY-4 A calibrated threshold decides match vs no match. "No match" is a valid answer.
- FR-QRY-5 Answer text is templated from rows (AGENTS.md rule 3).

**Clarify-once memory (FR-KB)** — detail in `03_CLARIFY_ONCE_MEMORY_SPEC.md`
- FR-KB-1 Detect unresolved location referents. FR-KB-2 One clarification card (camera pick, optional region, optional "same as <known>"). FR-KB-3 Persist durably before replying. FR-KB-4 Never ask again for a known referent, including after kill/restart. FR-KB-5 Auto-resume the original query after clarification. FR-KB-6 KB viewer with edit/delete.

**Evidence (FR-EVD)**
- FR-EVD-1 Each result: camera id/label, ISO start/end, bbox, thumbnail with bbox drawn, playable clip.
- FR-EVD-2 Clips cut on demand with ffmpeg and cached.

**UI (FR-UI)**
- FR-UI-1 Chat with result cards. FR-UI-2 Clarification card with camera thumbnails and region drawing. FR-UI-3 Camera wall + KB panel. FR-UI-4 Shows which pipeline mode answered (fast / verified).

**Evaluation (FR-EVAL)**
- FR-EVAL-1 Baseline and OmniWatch run on identical footage/queries with one command. FR-EVAL-2 Metrics + latency written to `eval/results/*.json`. FR-EVAL-3 Ablation table generated from those files.

## 6. Non-functional requirements
- NFR-1 Report warm-query p50/p95 latency in every mode. Target: fast mode no worse than baseline; do not promise absolute numbers before measuring.
- NFR-2 Measure ingest throughput on the demo machine; the unseen-footage plan depends on it.
- NFR-3 Fully offline after model download. NFR-4 One script each to ingest, serve, evaluate.

## 7. Definition of Done (P0 gate before any stretch)
1. `scripts/ingest.sh manifest.yaml` builds the index from raw videos.
2. 10 hand-written queries return camera + timestamp + working clip.
3. Eval harness reports baseline vs OmniWatch on held-out set with ablation rows.
4. Clarify-once passes the kill/restart test (T-KB-3 in 06_TEST_PLAN).
5. Write-up filled with numbers from `eval/results/`.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Unknown GPU / slow ingest on judge footage | Benchmark in hour 1; presets `fast`/`balanced`/`accurate`; motion gating; lower sample FPS |
| Judge footage lacks absolute timestamps / unsynced cameras | Ask organizers; `--assume-synced` flag; show the assumption in the UI |
| CLIP weak on color binding / compound phrases | Explicit color attributes + VLM verification; measured by ablation |
| Overfitting to our footage | Dev/test split by segment; test set touched once |
| Dataset licence/availability | Own campus footage as fallback (04) |
| Scope creep | Cut lines in 05 |

## 9. Open decisions (settle in hour 0)
1. Team size and roles. 2. GPU available? (model sizes depend on it). 3. Primary dataset(s) (04). 4. Time tolerance tau for "correct timestamp": default +-5 s; judges' tolerance unknown, so report +-5 s and +-15 s. 5. Baseline detector (02). 6. Local LLM for parsing (e.g. via Ollama) vs rule-only parser.
