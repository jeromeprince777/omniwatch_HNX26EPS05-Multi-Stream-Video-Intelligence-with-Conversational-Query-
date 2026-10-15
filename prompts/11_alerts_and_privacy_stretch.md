# 11 — STRETCH: standing queries/alerts and privacy
**Start only after DoD is green and (preferably) after privacy; privacy is cheaper than alerts. Read:** docs/00_PRD (stretch), docs/06 (T-PRV, T-ALT).

## A. Privacy (do first)
1. `evidence/blur.py`: face blur applied to every thumbnail, frame and clip served through the API (not to stored index crops), using a local face detector that you verify loads; fallback blur of the upper third of person boxes if no detector works. Toggle in config (`privacy.blur=true`) and shown in the UI.
2. Enforce on-prem-only: a startup check and a test that ingest and query run with network access disabled (e.g. run in a sandbox/with network blocked) after models are cached. `/api/health` shows `offline_ready: true/false`.
3. Document what is and is not protected in the write-up (index crops still contain faces on disk; say so, or encrypt/clean them if time allows).

## B. Standing queries and alerts
1. "Notify me if anyone enters this zone after 8 pm": parse into a standing query: object phrase, referent/zone (KB referent with ROI), time-of-day condition, repeat policy. Store in `standing_queries`.
2. Evaluator: whenever new tracklets are indexed (and in replay mode over recorded footage, simulating a clock), test them against active standing queries with the same retrieval/verification code path; write `alerts` rows, dedupe so one object passage fires one alert.
3. UI: alerts panel with evidence cards; creation via chat with a confirmation message that echoes the interpreted rule.
4. Tests: replay a recorded segment that contains a scripted entry after the configured time; assert exactly one alert, correct camera/time, and none for entries before the time.

**Acceptance:** show real test output; no mock alerts in the UI.
