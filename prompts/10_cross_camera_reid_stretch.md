# 10 — STRETCH: cross-camera re-identification and path timeline
**Start only if the PRD Definition of Done is green. Read:** docs/00_PRD (stretch), docs/01 (3: tracklets.global_id).

**Task:**
1. `reid/embed.py`: an appearance embedding for persons (e.g. a torchreid OSNet-class model; verify it loads) and for vehicles (CLIP crop embedding plus color; no vehicle-specific model unless you verify one). Compute from the tracklet's top-k crops.
2. `reid/link.py`: link tracklets across cameras into `global_id` using appearance similarity plus spatio-temporal constraints: a candidate must start after the source tracklet ends on a different camera and within a max transit time from config; optional camera topology (allowed transitions) from config or manifest. Greedy or Hungarian matching with a similarity threshold tuned on dev. Never link tracklets on the same camera at overlapping times.
3. Path intent: "where did this person go?" / "follow the red car": start from the matched tracklet, return an ordered timeline `Gate cam 09:14:07 -> Lobby cam 09:16:30 -> Rear cam 09:22:10` using KB labels where available, with a thumbnail per hop. Extend the answer templates (deterministic).
4. Evaluation: on the scripted scenario CSV, report pairwise linking precision/recall and path correctness. Add as a row/section in the write-up only if measured.

**Acceptance:** run on scenario footage with scripted cross-camera walkers, report real precision/recall. If it does not work well, say so and keep it behind a config flag, off by default.
