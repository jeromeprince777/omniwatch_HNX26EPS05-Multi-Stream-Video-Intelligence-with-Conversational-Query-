# 03 — Clarify-once persistent memory
20% of the rubric: "asks appropriately, never re-asks, persists across restart". This is a small knowledge base, not chat history.

## 1. What counts as a referent
A **location referent** is a named place/camera phrase in the query that designates part of the site: "main gate", "lobby", "rear entrance", "parking lot", "loading dock". Resolution order:
1. Explicit camera id in the query ("camera 3", "cam3") -> manifest, no question.
2. KB exact match on normalized phrase for this `site_id` -> use stored targets.
3. Manifest camera label exact/normalized match (if the footage ships with names) -> use it.
4. Otherwise -> clarification.
Generic words ("outside", "near", "there") are not referents; the parser must not emit them. Time phrases are not referents (out of scope).

## 2. Normalization (`kb/normalize.py`)
lowercase, strip punctuation, collapse spaces, drop leading articles ("the", "a"), drop trailing "camera/cam/area/zone", singularize simple plurals, optional synonym map defined in config (empty by default). Unit-test with >= 15 cases. Do not do fuzzy auto-merge: "front gate" is not silently "main gate".

## 3. State machine
```
RECEIVED -> PARSED -> resolve referents
   all resolved -----------------------------> RETRIEVE -> ANSWER
   some unresolved -> save pending_queries -> CLARIFY_ASKED (ONE card listing every unresolved phrase, one at a time in the same card flow)
CLARIFY_ASKED -> user submits targets -> WRITE referents (commit, fsync) -> log clarification_log -> load pending query -> RETRIEVE -> ANSWER
CLARIFY_ASKED -> user submits "same_as" -> write alias row pointing to the same targets -> resume
CLARIFY_ASKED -> user cancels -> nothing stored; answer says the phrase was ignored (documented non-persisting path)
```
Rules:
- Ask only on KB miss. A hit must never produce a card.
- One card per unknown phrase; never ask the same phrase twice in a session or ever after it is stored.
- Write is committed (`synchronous=FULL`) **before** the HTTP response that confirms it.
- The pending query resumes automatically; the user does not retype it.
- The card offers: pick one or more cameras (with thumbnails), optionally draw a region on a camera, optionally "same as <known referent>", and, if a near-match exists (string similarity above a config threshold), "Did you mean <known>?" as part of the same card.

## 4. What is stored
`referents(site_id, phrase_norm, phrase_display, targets_json=[{camera_id, roi|null}], source, timestamps)`; `referent_history` for edits; `clarification_log` for audit and for the "never re-asks" test (a phrase must appear at most once as asked unless the referent was deleted by the user).
Scope: keyed by `site_id`. New footage = new manifest = new site unless the manifest declares the same `site_id`. Reason: camera ids can collide across datasets and silently mapping them would give wrong answers. State this behavior in the write-up and runbook.

## 5. Using a stored region
ROI is normalized `[x1,y1,x2,y2]` on that camera. A tracklet matches the location if any observation's box center (or the box overlap ratio above a config threshold) falls inside the ROI during the tracklet. No ROI -> any detection on that camera counts.

## 6. Editing and corrections
KB panel lists referents with edit/delete. A natural-language correction ("no, main gate is cam 3") is parsed by a small rule; confirm in the UI, then update with history. Deleting a referent allows a future ask; nothing else does.

## 7. Failure modes to test (see 06)
- kill -9 immediately after a clarification and restart; query again -> no card.
- Two unknown phrases in one query -> two sequential cards, then one answer.
- Same phrase with different casing/articles -> hit.
- Referent mapped to two cameras -> results from both.
- Query with no referent -> never a card (no over-asking).
- Corrupt/missing DB -> clear error, no silent empty KB.
