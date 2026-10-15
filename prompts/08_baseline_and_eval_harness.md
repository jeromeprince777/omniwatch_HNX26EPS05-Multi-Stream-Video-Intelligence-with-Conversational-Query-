# 08 — Baseline systems and evaluation harness
**Read:** docs/02_RESEARCH_AND_EVAL_PLAN.md completely, docs/04 (5), docs/06 (T-RET, T-LAT, T-RES).

**Task:**
1. `eval/baseline.py`: implement B1 (same detector + same encoder as OmniWatch, per-detection CLIP crop embeddings from `observations`, text-to-crop cosine, top-K, no tracking/parsing/attributes), B1+ (B1 plus camera/time filters from the shared parser/KB), B2 (whole-frame CLIP; add a frame-embedding pass only for eval). Optional B0 only if explicitly asked later. B1 must reuse the already-indexed observation embeddings so the comparison is about method, not about re-running different models; document any exceptions.
2. `eval/metrics.py`: Hit@1/5, MRR, existence accuracy, false-positive rate on negatives, localization error (median/p90), with tolerance tau in {5, 15} s. Unit-test each metric with hand-made cases, including negatives and overlapping intervals.
3. `eval/run_eval.py --split dev|test --systems B1,B1+,OW-fast,OW-full --ablate` writing per-query outputs and aggregates to `eval/results/<run_id>/` (`per_query.jsonl`, `summary.json`, `config_used.json` with code git hash, thresholds, hardware, sample FPS). Latency protocol exactly as docs/02 section 5 (warm process, 3 warm-up discards, p50/p95). The script refuses to run `--split test` if `eval/results/test_lock` already exists, and creates it after a successful run (override requires `--i-know-this-burns-the-test-set`).
4. `eval/make_tables.py`: reads `summary.json` files and prints the markdown tables used in the write-up. No numbers may be typed by hand.
5. `eval/queries/dev.jsonl` and `test.jsonl` schema validator, and a `scripts/eval.sh`.
6. Provide a template/example file with 3 clearly fake example queries, flagged `"example": true`, which the harness ignores.

**Acceptance:** run on whatever dev queries exist (or on the 3 examples marked as smoke test only) and show real output files. Report that thresholds were not tuned on test. Do not invent query results.
