# 05 — 24-hour plan, roles, cut lines
Times are relative to kickoff. Adjust to the real schedule. Team size is unknown: roles below assume 4; merge as needed.

## Roles
- **A — Pipeline:** ingest, detection, tracking, embeddings, index (prompts 01-04).
- **B — Query/ML:** parser, retrieval, verification (05, 09).
- **C — Product:** KB, API, UI, evidence (06, 07).
- **D — Evaluation/data:** footage, labeling, baseline, eval harness, write-up (08, 12).
Everyone reviews with prompt 13 before freeze.

## Timeline
| Hours | Goal | Gate |
|---|---|---|
| 0-1 | Hour-0 decisions (PRD section 9); repo + AGENTS.md; **benchmark machine** (GPU? decode speed, detector fps); download/cache models; start recording footage (D) | Models load, benchmark numbers noted |
| 1-4 | Scaffold; ingest + detect + embed + SQLite/FAISS on 1 camera, then all (A). Parser + retrieval skeleton on fake rows (B). API + UI shell (C). Recording + scenario CSV (D) | One query returns camera+time from real index |
| 4-8 | Tracking + tracklet index (A). Retrieval, color attrs, templates (B). Clarify-once KB + clip/thumb endpoints (C). Dev queries labeled (D) | **Checkpoint 1:** end-to-end chat demo on dev footage |
| 8-12 | Clarification card UI with ROI; KB panel (C). Time parsing, negatives threshold (B). Baseline B1/B1+ + metrics code (D). Throughput tuning (A) | Baseline and ours run on dev set |
| 12-16 | Dev-set tuning of weights/thresholds (B+D). VLM verification (B). Restart/KB tests (C). Camera wall, polish (C) | **Checkpoint 2:** Definition of Done items 1-4 |
| 16-19 | Ablation runs on dev; fix top failure categories; latency measurements; rehearse unseen-footage ingest on a fresh clip (all) | Ablation table script works |
| 19-20 | **Feature freeze.** Choose stretch only if DoD is green: privacy blur (cheap) > alerts > re-ID | Freeze tag in git |
| 20-22 | Run test set once; generate tables; write-up (D, B). Optional stretch in a side branch (A, C) | `eval/results/` final |
| 22-24 | Prompt 13 audit; dry-run judging runbook; record a 2-min backup demo video; package | Demo rehearsed twice |

## Cut lines (what to drop, in order, when late)
1. Re-ID/path timeline. 2. Alerts. 3. Live ingestion. 4. Frame-level fallback index. 5. B0 baseline. 6. Leave-one-out ablation rows (keep B1, B1+, full). **Never cut:** clarify-once persistence, grounded evidence, baseline comparison, write-up.

## Risks to watch hourly
Ingest too slow for unseen footage -> lower FPS/motion gate. Parser failures -> rules fallback. Clock/sync mismatch across cameras -> check scenario CSV against detected events early (hour 5).
