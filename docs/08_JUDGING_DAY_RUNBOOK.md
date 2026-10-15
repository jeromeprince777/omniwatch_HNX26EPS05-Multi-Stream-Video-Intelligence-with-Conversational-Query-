# 08 — Judging-day runbook (unseen footage + queries with known answers)

## Before judging (T-60 min)
- [ ] Machine plugged in, no other heavy apps, GPU drivers OK, models cached in `models/`, network not needed.
- [ ] Fresh clone/tag from the freeze commit. `scripts/demo_reset.sh` only resets the demo site, never `eval/`.
- [ ] Rehearsed ingest of a clip never seen in development (T-ING-1) and recorded how long it takes per minute of video per camera.
- [ ] Backup: 2-minute demo video, screenshots, results JSON on a USB/second device.

## When judges hand over footage
1. Ask: absolute start time per camera? Are cameras synchronized? Any camera names or map? (If no timestamps, use `--assume-synced` and say so out loud.)
2. Write `manifest.yaml`; choose preset by total duration and the benchmark: long footage -> `fast` preset (lower sample FPS, motion gating); short -> `balanced`/`accurate`.
3. Run `scripts/ingest.sh manifest.yaml`; watch `/api/ingest/status`. Use the waiting time to show the UI and the camera wall.
4. New footage = new `site_id`, so the KB starts empty for it. Expect and welcome clarification cards: they are the demonstration of clarify-once. After answering, repeat a query using the same referent to show no re-ask; if allowed, restart the server and ask again.
5. Judges' queries: type them verbatim. Do not retype in a friendlier form.
6. For each answer, open the thumbnail and clip, state camera + timestamp. If the answer is "no match", show the searched window.

## If something breaks
| Symptom | Action |
|---|---|
| Parser LLM unavailable | Rules fallback is automatic; mention it |
| Ingest slower than expected | Restart with `fast` preset on remaining cameras; answer queries on what is already indexed and say which cameras are pending |
| Wrong time offset | Check manifest start times; compare one known event; fix manifest and re-ingest that camera only |
| Verification too slow | Switch to fast mode in the UI and say so |
| Out-of-vocab object not found | Show frame-level fallback if implemented; otherwise be honest about the detector vocabulary limit |

## Honesty rules in the room
Never claim a number that is not in `eval/results/`. If asked about something not built (live, re-ID), say it is not built. Say which assumptions were made (sync, timestamps).
