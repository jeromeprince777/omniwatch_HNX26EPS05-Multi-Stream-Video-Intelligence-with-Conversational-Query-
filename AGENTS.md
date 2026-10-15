# AGENTS.md — rules for every coding-agent session on OmniWatch

## Source of truth
`docs/` is the spec. Priority when documents disagree: PRD > ARCHITECTURE > CLARIFY-ONCE SPEC > RESEARCH PLAN > others. If still ambiguous, stop and ask the human. Do not guess and do not silently change the spec; propose the change in your DONE REPORT.

## Anti-hallucination rules (non-negotiable)
1. **No invented APIs.** Before using any third-party function, class or CLI flag, check the installed version (`pip show <pkg>`) and confirm the signature (`help()`, source, or official docs). Write a 5-line smoke test first. If a model name, checkpoint or package does not exist or does not load, say so; never substitute a made-up name.
2. **No invented results.** Never hard-code, estimate or "illustrate" metrics, latencies or accuracies. Numbers appear only in files written by `eval/run_eval.py`. Write-up numbers are copied from those files.
3. **Answers are grounded.** The text shown to the user is built from database rows (camera, timestamp, bbox, thumbnail, clip). An LLM may parse the query and may verify candidates, but it must never write an answer that has no evidence row. If nothing passes the threshold the answer is "no match", with the searched window.
4. **No silent fallbacks.** If a component falls back (smaller model, CPU, regex parser instead of LLM), log it at WARNING and expose it in `/api/health`.
5. **Run what you write.** Every module ships with at least one test that you executed. Report the command and its output summary. "Should work" is not a result.

## Engineering rules
- Python 3.11, type hints, `ruff` clean, small functions, no global state except config.
- All paths and thresholds live in `config.py` / `config.yaml`. No magic numbers in modules.
- Offline at query time: no network calls during ingest or query once models are cached in `models/`. (Privacy bonus depends on it.)
- SQLite is the single source of truth for metadata and the knowledge base. Vector index files are derived and can be rebuilt.
- Deterministic where possible: fixed seeds, sorted outputs, stable tie-breaking.
- Do not commit videos, model weights or `data/`. Do commit `models/MANIFEST.md` (name, source, version/hash, license note).
- Small commits with clear messages. Do not refactor unrelated files.

## Output contract for each task
End every task with a **DONE REPORT**:
1. Files created/changed
2. Commands run + pass/fail
3. Deviations from the spec (and why)
4. Known gaps / risks
5. What you verified vs what you assumed
