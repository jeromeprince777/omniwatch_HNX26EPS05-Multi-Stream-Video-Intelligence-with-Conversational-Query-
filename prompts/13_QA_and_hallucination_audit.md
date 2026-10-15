# 13 — Adversarial QA and hallucination audit (run before freeze and before judging)
You are a skeptical reviewer, not the author. Do not fix anything until you have listed findings. Read all of `docs/` and the code.

**Audit checklist (report PASS/FAIL with evidence for each):**
1. **Grounding:** find every code path that produces user-visible answer text. Confirm it interpolates only database fields. Try 10 adversarial queries (nonexistent object, impossible time, unknown camera, empty index, contradictory constraints). None may produce a positive claim without a result row.
2. **Clarify-once:** run T-KB-1..9 again by hand. Try phrase variants ("Main Gate", "the main gate camera", "main-gate"), restart, delete-and-reask, two unknowns, and a query with no referent (must not ask).
3. **Time handling:** relative expressions vs `reference_now`; timezone offsets; camera with later start time; query spanning two video parts.
4. **Timestamp correctness:** pick 10 results, play the clips, confirm the event is inside; compare thumbnail timestamps with the clip.
5. **Unseen footage:** ingest a clip never used before with a new `site_id`, no code changes; check logs for fallbacks.
6. **Numbers:** run `scripts/check_writeup_numbers.py`; confirm test-set lock file exists and tuning logs show dev-only access.
7. **Baseline fairness:** confirm B1 uses the same footage, FPS, detector and encoder; list any differences.
8. **Dependencies and licences:** `models/MANIFEST.md` complete; no network calls at query time (check with network disabled).
9. **Code smells that cause demo failures:** hard-coded paths, absolute camera names, magic thresholds, swallowed exceptions, silent fallbacks.
10. **Spec drift:** list every deviation from `docs/` found in the code.

**Output:** a table of findings with severity (blocker/major/minor), reproduction steps and a suggested fix, then wait for human approval before changing code.
