# 06 — Test plan (mapped to the judging rubric)
All tests runnable with `pytest` or a script; record results in `eval/results/test_report.md`.

## Retrieval and localization (50%)
- T-RET-1 For each dev query, top result camera correct and time within tau. Reported by `run_eval.py`, not asserted per query.
- T-RET-2 Negative queries return `no_match` (report false-positive rate).
- T-LOC-1 Every result row has camera id, t_start<=t_peak<=t_end inside the camera's recorded range, bbox in [0,1], existing thumbnail and clip files.
- T-LOC-2 Clip for a result actually contains the event: manually verify 10 clips by playing them.
- T-GRD-1 No-evidence-no-answer: monkeypatch retrieval to return nothing and confirm the answer is `no_match`, never a positive statement. Grep that answer templates only interpolate row fields.

## Clarify-once (20%)
- T-KB-1 Unknown referent -> exactly one clarification card, pending query stored.
- T-KB-2 After clarification the original query resumes and answers without retyping.
- T-KB-3 **Restart test:** clarify "main gate", `kill -9` the server right after the confirmation response, restart, ask again with different wording ("the Main Gate cam"), expect no card and a correct answer. Also verify with `sqlite3` that the row exists.
- T-KB-4 Known referent never asks (run 20 queries mentioning it).
- T-KB-5 Query without referent never asks.
- T-KB-6 Two unknown phrases -> two sequential cards -> one answer.
- T-KB-7 Multi-camera and ROI referents filter results as specified.
- T-KB-8 Edit/delete in KB panel changes behavior accordingly; delete allows a new ask.
- T-KB-9 `clarification_log` shows each phrase asked at most once.

## Latency (10%)
- T-LAT-1 Warm-process timing over the full query set for B1, OW-fast, OW-full; p50/p95 saved to JSON. 3 warm-up queries discarded. Nothing else running.
- T-LAT-2 Cold start time (process start to first answer) recorded once.

## Ingestion robustness (supports unseen footage)
- T-ING-1 Ingest a 2-minute clip from a camera never used in development; completes without code changes.
- T-ING-2 Re-ingest is idempotent (row counts unchanged).
- T-ING-3 Missing `start_time` without `--assume-synced` fails with a clear message.
- T-ING-4 Corrupt/short video skipped with a warning, others continue.

## Research contribution (20%)
- T-RES-1 `run_eval.py` produces ablation rows and per-query outputs; the write-up table is generated from the JSON by script.
- T-RES-2 Same footage, FPS, hardware for all systems (assert in the script and store in `config_used.json`).

## UI smoke (use the Playwright MCP if available in Antigravity)
- T-UI-1 Send a query, see cards with thumbnail; clip plays. T-UI-2 Clarification card appears, region drawing works, answer appears after submit. T-UI-3 KB panel lists the new referent.

## Stretch tests
T-PRV-1 No outbound network connections during ingest/query (run with network disabled). T-PRV-2 Blur present on served thumbs/clips. T-REID-1 Path for a scripted person/vehicle matches scenario CSV. T-ALT-1 Standing query fires on replay at the expected time, once.
