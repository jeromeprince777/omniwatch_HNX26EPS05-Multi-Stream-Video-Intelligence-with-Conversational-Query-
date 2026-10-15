# OmniWatch — short write-up (HNX26EPS05)
> Rules: every number must be copied from `eval/results/*.json`. If a number is missing, write "not measured". Max ~2 pages.

## 1. Problem and approach (5-8 lines)
Plain description: multi-camera recorded footage -> index -> chat with grounded answers and clarify-once memory.

## 2. System overview
Diagram or 6-line pipeline: decode -> detect -> track -> embed + attributes -> index -> parse -> resolve (KB) -> retrieve -> verify -> evidence.
Models and versions used: (copy from `models/MANIFEST.md`).

## 3. Data
Footage used (own / public, number of cameras, total duration), dev/test split rule, number of dev and test queries, share of negative queries, labeling method.

## 4. Baseline
Definition of B1 (and B1+, B0 if done), same footage/FPS/hardware. Why this is a strong baseline for the task.

## 5. Results
Generated table (script output):
| System | Hit@1 | Hit@5 | MRR | Exist. acc. | FP rate on negatives | Median loc. error (s) | p50 / p95 latency (ms) |
|---|---|---|---|---|---|---|---|
| B1 | | | | | | | |
| B1+ | | | | | | | |
| OW-fast | | | | | | | |
| OW-full | | | | | | | |
n = ___ test queries, tau = ___ s. CI: ___ (or "not computed").

## 6. Ablation
Table from the matrix in 02 (tracklets, parser filters, attributes, verify, leave-one-out). One sentence per row on what it shows, including components that did **not** help.

## 7. Clarify-once memory
Design (SQLite KB, site-scoped), behavior, restart test result (T-KB-3 pass/fail), known limits (cancel path is non-persisting, new site = new KB).

## 8. What is new vs the baseline
Bullet list tied to ablation rows. Nothing that is not shown by a table.

## 9. Bonus items (only what actually works)
Live ingestion / re-ID / alerts / privacy: what was done, how tested, what was not.

## 10. Limitations and threats to validity
Small query set, own labeling, scene bias, hardware dependence, failure categories with 2-3 examples (with thumbnails).

## 11. Reproduce
Three commands: ingest, serve, eval.
