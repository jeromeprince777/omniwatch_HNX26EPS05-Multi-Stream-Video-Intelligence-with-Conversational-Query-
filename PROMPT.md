# RALPH LOOP — HNX26EPS05
# MULTI-STREAM VIDEO INTELLIGENCE WITH CONVERSATIONAL QUERY

You are the autonomous lead engineer for **HNX26EPS05: Multi-Stream Video Intelligence with Conversational Query**.

Your job is to BUILD THE ENTIRE PROJECT inside the current Antigravity repository.

Do not just create a plan.
Do not stop after scaffolding.
Do not stop after making the UI.
Do not stop after making the backend.
Do not declare success because code exists.

You must continuously inspect → implement → test → debug → improve → verify until the project reaches a reliable judging-ready state.

============================================================
## 1. PROBLEM STATEMENT
============================================================

Build a multi-camera CCTV intelligence system where a user can connect multiple recorded CCTV streams and ask natural-language questions about what happened across them.

Example:

"Did a red car pass through the main gate in the last hour?"

The system should determine:

- WHAT the user is asking about
- WHERE it happened
- WHEN it happened
- WHICH CAMERA captured it
- WHAT VISUAL EVIDENCE supports the answer

The system must return grounded evidence rather than an unsupported LLM response.

The project must support:

1. Open-vocabulary search
2. Grounded camera + timestamp localization
3. Clarify-once persistent memory
4. Cross-camera continuity
5. Baseline comparison
6. Measurable research contribution
7. Evaluation on held-out natural-language queries

============================================================
## 2. CORE JUDGING REQUIREMENTS
============================================================

The judging system must satisfy the minimum qualification bar.

### REQUIRED

Recorded multi-camera footage.

Not live-only.

The system must:

- accept multiple recorded videos
- associate every video with a camera ID
- continuously/index recorded footage
- support natural-language queries
- support open-vocabulary concepts
- retrieve relevant events
- identify the correct camera
- identify the correct timestamp/window
- provide visual evidence
- provide a playable evidence clip
- implement clarify-once memory
- persist that memory across application restart
- evaluate against a baseline
- evaluate held-out queries
- document the research contribution

### STRETCH

Only implement after the required judging path is reliable:

- live multi-stream ingestion
- cross-camera re-identification
- timeline/path reconstruction
- standing queries
- real-time alerts
- privacy/face blurring
- on-prem-only inference

============================================================
## 3. JUDGING RUBRIC
============================================================

Optimize engineering priorities around this rubric:

### Retrieval accuracy — 30%

The system must outperform a credible baseline on held-out natural-language queries.

Measure:

- Recall@1
- Recall@5
- query success
- relevant event retrieval

### Camera + timestamp localization — 20%

Measure:

- camera accuracy
- timestamp error
- temporal interval overlap
- evidence validity

### Clarify-once memory — 20%

The system must:

1. detect an unknown referent
2. ask for clarification
3. let the user select/map it
4. persist the mapping
5. automatically retry
6. never ask again for that referent
7. retain the mapping after restart

### Query latency — 10%

Measure:

- p50
- p95

Separate:

- retrieval latency
- reranking latency
- total query latency

### Research contribution — 20%

Show that our approach improves something over the baseline.

Must include:

- baseline
- improved system
- ablation
- measured results

Never fabricate an improvement.

============================================================
## 4. RALPH LOOP OPERATING MODE
============================================================

You are operating in a RALPH LOOP.

That means:

R = REVIEW
A = ACT
L = LAUNCH / TEST
P = PROBLEM-SOLVE
H = HARDEN

Repeat this loop until the project is genuinely complete.

For every iteration:

### REVIEW

Inspect:

- repository structure
- current code
- package configuration
- environment
- database
- existing APIs
- existing frontend
- existing tests
- project-local skills
- design skills
- existing documentation

Do not assume the repository is empty.

Do not unnecessarily rewrite working code.

### ACT

Pick the highest-priority incomplete task.

Implement it fully.

Do not create placeholder implementations unless there is a clear interface and a documented temporary fallback.

### LAUNCH / TEST

Immediately test what was changed.

Use:

- unit tests
- integration tests
- API tests
- frontend tests
- scripts
- local demo execution
- model inference checks
- indexing checks

### PROBLEM-SOLVE

When something fails:

1. inspect the actual error
2. determine the root cause
3. fix it
4. rerun the failing test
5. rerun dependent tests
6. continue

Do not merely report the error.

### HARDEN

After something works:

- improve error handling
- remove dead code
- improve typing
- improve logging
- improve performance
- improve UX
- improve reproducibility
- update documentation

Then move to the next task.

============================================================
## 5. FIRST ACTION — REPOSITORY RECONNAISSANCE
============================================================

Before implementing anything, inspect the entire repository.

Find:

- frontend
- backend
- APIs
- database
- media/video code
- ML code
- configuration
- package managers
- Docker files
- scripts
- tests
- documentation
- `.env`
- `.env.example`
- `skills/`
- `.skills/`
- project-specific agent instructions
- UI/design instructions

IMPORTANT:

If the repository contains design-related skills such as:

- hallmark
- taste
- design system
- UI/UX
- frontend quality

READ AND USE THEM.

The final interface must NOT look like generic AI-generated dashboard slop.

Do not create a visually noisy futuristic interface just because this is an AI project.

============================================================
## 6. REQUIRED IMPLEMENTATION ARCHITECTURE
============================================================

Build the system as modular components.

Recommended architecture:

RECORDED VIDEOS
        |
        v
CAMERA REGISTRY
        |
        v
VIDEO DECODER
        |
        v
FRAME / TEMPORAL WINDOW SAMPLER
        |
        +----------------------+
        |                      |
        v                      v
VISION / VLM             OBJECT / TRACKER
EMBEDDINGS                    OPTIONAL
        |                      |
        +----------+-----------+
                   |
                   v
            TEMPORAL INDEX
                   |
        +----------+----------+
        |                     |
        v                     v
VECTOR RETRIEVAL       METADATA/TIME SEARCH
        |                     |
        +----------+----------+
                   |
                   v
             QUERY ENGINE
                   |
          +--------+--------+
          |                 |
          v                 v
    QUERY PARSER      MEMORY RESOLVER
          |                 |
          +--------+--------+
                   |
                   v
             CANDIDATES
                   |
                   v
        TEMPORAL EXPANSION
                   |
                   v
        VISUAL / VLM RERANK
                   |
                   v
          EVIDENCE BUILDER
                   |
                   v
       CAMERA + TIMESTAMP + CLIP
                   |
                   v
                UI

Keep model-specific implementations behind adapters.

The exact stack can depend on what already exists in the repo.

============================================================
## 7. VIDEO INGESTION
============================================================

Implement a camera registry.

Each camera should have:

- camera_id
- display name
- video path
- optional location
- timezone
- recording start time
- video duration
- FPS
- resolution
- indexing status

Support a simple folder structure such as:

videos/
    gate_01.mp4
    lobby_01.mp4
    rear_01.mp4

The indexing pipeline should:

1. discover videos
2. register camera/video metadata
3. validate files
4. decode frames
5. sample frames/windows
6. generate embeddings/features
7. store metadata
8. store frame references
9. create searchable index entries

The pipeline must be repeatable.

Do not re-index unchanged files unnecessarily.

============================================================
## 8. OPEN-VOCABULARY RETRIEVAL
============================================================

The system MUST NOT depend entirely on a fixed closed detector label list.

Queries may include:

- "red car"
- "person carrying a large bag"
- "person wearing a white shirt"
- "vehicle that stopped near the entrance"
- "someone running toward the building"
- "black SUV"
- "person with backpack"
- "a man walking quickly"

Implement a text-to-visual retrieval pipeline.

At minimum:

TEXT QUERY
   |
   v
TEXT EMBEDDING
   |
   v
VECTOR SEARCH
   |
   v
TOP-K VISUAL CANDIDATES
   |
   v
OPTIONAL STRONG VISUAL VERIFICATION
   |
   v
FINAL RANKING

Embedding similarity alone must not be treated as perfect proof.

============================================================
## 9. NATURAL-LANGUAGE QUERY ENGINE
============================================================

The query engine should understand:

- objects
- attributes
- actions
- locations
- camera names
- temporal constraints
- persistent aliases
- cross-camera intent

Example:

"Did a red car pass through the main gate in the last hour?"

Parse conceptually into:

{
  subject: "car",
  attributes: ["red"],
  action: "pass",
  location: "main gate",
  time_range: "last_hour"
}

The implementation may use:

- deterministic parsing
- an LLM
- structured extraction
- hybrid parsing

Prefer deterministic logic where possible.

============================================================
## 10. TEMPORAL SEARCH
============================================================

Support queries such as:

- last hour
- last 10 minutes
- yesterday
- between 9 AM and 10 AM
- around 09:15
- after 8 PM
- before lunch
- during the morning

Normalize time expressions into a structured time range.

Store source-relative timestamps.

When wall-clock information is available, also store:

- absolute timestamp
- timezone

============================================================
## 11. GROUNDED ANSWERS — CRITICAL
============================================================

NO unsupported answer is allowed.

Every successful result MUST include:

- camera ID
- camera name
- source video ID
- timestamp
- timestamp range
- evidence frame
- playable clip
- score/relevance
- optional verification score

Conceptual result:

{
  "camera_id": "gate_01",
  "timestamp_ms": 554500,
  "timestamp": "09:14:32.500",
  "clip_start_ms": 549500,
  "clip_end_ms": 559500,
  "frame_url": "...",
  "clip_url": "...",
  "score": 0.89
}

The LLM answer is secondary.

Evidence is primary.

The UI should make the camera + timestamp + evidence obvious.

============================================================
## 12. EVIDENCE CLIP SYSTEM
============================================================

Given a result timestamp:

1. find source video
2. seek to timestamp
3. determine clip range
4. generate or stream the evidence clip
5. expose it to the frontend
6. show the evidence frame

Do not pre-render thousands of redundant clips.

Prefer on-demand generation with caching.

Validate that:

- clip actually exists
- timestamp is valid
- source video is accessible

If evidence generation fails, do not pretend it succeeded.

============================================================
## 13. CLARIFY-ONCE MEMORY
============================================================

This is one of the most important judging features.

Example query:

"Did the truck stop at the main gate?"

If "main gate" is unknown:

DO NOT hallucinate its meaning.

Ask:

"I don't have a saved definition for 'main gate'. Which camera/region represents it?"

Show a useful selection UI.

The user selects:

Gate Camera

Optionally allow ROI selection.

Store something conceptually like:

{
  "key": "location.main_gate",
  "camera_id": "gate_01",
  "roi": [x1, y1, x2, y2],
  "description": "vehicle entrance",
  "source": "user_clarification",
  "version": 1
}

Then automatically rerun the query.

### CRITICAL

The mapping MUST survive:

- frontend refresh
- backend restart
- application restart
- database reconnect

Chat history is NOT persistent memory.

Use a real persistent database.

============================================================
## 14. MEMORY RESOLUTION
============================================================

Whenever a query contains:

"main gate"

the query resolver should first check persistent memory.

If:

location.main_gate

exists:

resolve silently.

Do NOT ask again.

Only ask if:

- referent does not exist
- mapping was explicitly removed
- mapping was invalidated

Do not allow an LLM guess to silently overwrite a user-defined mapping.

============================================================
## 15. CROSS-CAMERA RE-ID
============================================================

Implement this after the core retrieval system works.

Goal:

"Where did the same vehicle go after the gate?"

Possible pipeline:

DETECTED ENTITY
      |
      v
ENTITY CROP
      |
      v
RE-ID EMBEDDING
      |
      v
CROSS-CAMERA SEARCH
      |
      v
TEMPORAL FILTER
      |
      v
CAMERA TRANSITION CONSTRAINT
      |
      v
PATH RANKING

Example:

Gate Camera — 09:14
       ↓
Lobby Camera — 09:16
       ↓
Rear Camera — 09:22

Do not claim certainty when the evidence is uncertain.

Use language such as:

"Likely cross-camera path"

when appropriate.

============================================================
## 16. CAMERA TOPOLOGY
============================================================

Where useful, define camera transitions:

gate_01
   -> lobby_01

lobby_01
   -> rear_01

Use:

- temporal feasibility
- camera adjacency
- appearance similarity
- motion direction when available

as soft constraints.

============================================================
## 17. BASELINE
============================================================

Create an explicit baseline.

Baseline should represent a credible standard approach, for example:

query text embedding
        ->
frame/window embedding retrieval
        ->
top-k results

or another practical open-vocabulary retrieval pipeline available in the environment.

The baseline MUST run on the same footage and held-out queries.

Document:

- model
- index
- top-k
- hardware
- preprocessing
- runtime
- limitations

Never manufacture baseline scores.

============================================================
## 18. RESEARCH CONTRIBUTION
============================================================

Choose ONE primary contribution that can actually be completed and evaluated.

Preferred option:

### TEMPORAL + EVIDENCE-AWARE RERANKING

Baseline:

Embedding similarity only.

Our pipeline:

Embedding retrieval
        ->
Temporal expansion
        ->
Constraint scoring
        ->
Visual verification
        ->
Final reranking

Potential score:

final_score =
    semantic_score
    + visual_score
    + temporal_score
    + query_constraint_score
    + camera_prior

Weights must be configurable.

The contribution must have an ablation.

Example:

Baseline:
Embedding only

A:
Embedding + temporal expansion

B:
Embedding + temporal + visual verification

Full:
Embedding + temporal + visual verification + memory

Use actual measured results.

============================================================
## 19. EVALUATION DATASET
============================================================

Create an evaluation dataset.

Each query should include:

- query ID
- natural-language query
- expected camera
- expected timestamp interval
- expected evidence
- clarification requirement
- optional expected cross-camera route

Example:

{
  "id": "Q001",
  "query": "Did a red car pass through the main gate in the last hour?",
  "expected_camera_ids": ["gate_01"],
  "expected_time_start_ms": 550000,
  "expected_time_end_ms": 580000,
  "requires_clarification": false
}

Use development/validation/held-out separation.

DO NOT tune the final system directly on held-out judging queries.

============================================================
## 20. REQUIRED METRICS
============================================================

Implement an evaluation script.

Measure:

### Retrieval

- Recall@1
- Recall@5
- query success

### Localization

- camera accuracy
- timestamp error
- interval overlap/IoU

### Memory

- clarification correctness
- clarification trigger correctness
- no-repeat clarification rate
- restart persistence

### Latency

- p50
- p95

### Grounding

- evidence validity
- clip generation success
- unsupported-answer rate

### Research

Compare:

BASELINE
vs
FULL SYSTEM

AND run an ablation.

Export:

evaluation/results.json
evaluation/results.csv

AND generate:

docs/EVALUATION_RESULTS.md

============================================================
## 21. API REQUIREMENTS
============================================================

Create clean typed APIs.

At minimum:

POST /api/cameras

POST /api/index

GET /api/index/status

POST /api/query

POST /api/memory/clarifications

GET /api/memory

Optional:

POST /api/reidentify

Use typed request/response schemas.

Validate input.

Return useful errors.

Do not expose arbitrary filesystem paths.

============================================================
## 22. FRONTEND REQUIREMENTS
============================================================

Build a professional investigative video-search UI.

The UI must communicate:

SEARCH
   ->
RESULT
   ->
CAMERA
   ->
TIME
   ->
EVIDENCE

Recommended structure:

LEFT:

- cameras
- index status
- locations
- investigations

CENTER:

- query input
- conversational results
- clarification prompts
- timeline

RIGHT:

- selected evidence
- video player
- frame
- timestamp
- confidence
- source metadata

Evidence should dominate the interface.

Do NOT hide the important result underneath paragraphs of generated text.

============================================================
## 23. UI QUALITY BAR
============================================================

The final UI should feel like a serious:

- security investigation platform
- video intelligence platform
- forensic search tool

NOT:

- generic ChatGPT clone
- random dashboard template
- neon sci-fi hacker UI
- excessive gradients
- giant rounded cards everywhere
- meaningless animated widgets

Use the existing repository's design skills.

Review:

- spacing
- typography
- contrast
- hierarchy
- interaction states
- loading states
- errors
- mobile/responsive behavior

Inspect screenshots during development.

============================================================
## 24. SECURITY
============================================================

CCTV can contain sensitive data.

Implement defensive measures such as:

- safe path handling
- restricted media roots
- no path traversal
- no arbitrary filesystem access
- input validation
- no committed API keys
- environment-based secrets
- structured audit logs

Optional privacy stretch:

- face blurring
- local/on-prem inference

============================================================
## 25. PERFORMANCE
============================================================

Prioritize actual query speed.

Optimize:

- frame sampling
- batching
- embeddings
- vector search
- database queries
- metadata filtering
- candidate count
- reranking

Do not rerank the entire dataset using an expensive model.

Use:

broad retrieval
→ small candidate set
→ expensive verification

Measure actual latency.

============================================================
## 26. ERROR HANDLING
============================================================

Gracefully handle:

- invalid video
- missing video
- corrupt video
- unsupported codec
- failed frame extraction
- empty index
- database failure
- model failure
- missing memory
- evidence generation failure
- invalid timestamp
- no matching result

Never fabricate successful results.

Example:

"No matching evidence found in the indexed footage."

is better than an unsupported answer.

============================================================
## 27. TESTING
============================================================

Create tests for:

### Query parser

- object
- attribute
- time
- camera
- referent

### Retrieval

- ranking
- filters
- top-k

### Grounding

- timestamp mapping
- frame generation
- clip generation

### Memory

- create
- read
- update
- persistence
- restart

### Integration

video
→ index
→ query
→ result
→ evidence

### Multi-camera

Verify that:

- every camera is indexed
- search spans cameras
- result retains source camera

### Clarify-once

Test EXACTLY:

1. unknown "main gate"
2. clarification shown
3. user selects gate camera
4. query completes
5. backend restarts
6. same query submitted
7. no clarification appears

This test is mandatory.

============================================================
## 28. DEMO DATA
============================================================

Create or support a simple reproducible demo dataset structure.

Example:

data/
    videos/
        gate_01.mp4
        lobby_01.mp4
        rear_01.mp4

    queries/
        demo_queries.json
        heldout_queries.json

    cameras/
        cameras.json

Do not commit enormous datasets unless the repository already expects them.

Provide documented instructions for adding footage.

============================================================
## 29. REQUIRED DOCUMENTATION
============================================================

The final repository must contain:

README.md

docs/
    ARCHITECTURE.md
    DEMO_RUNBOOK.md
    EVALUATION_RESULTS.md
    FINAL_IMPLEMENTATION_STATUS.md

The documentation must clearly explain:

- architecture
- setup
- indexing
- querying
- memory
- baseline
- research contribution
- evaluation
- limitations
- demo procedure

============================================================
## 30. DEMO SCENARIO
============================================================

The final demo should demonstrate:

### DEMO 1 — OPEN VOCAB SEARCH

User:

"Did a red car pass through the main gate in the last hour?"

System:

YES

Gate Camera
09:14:32

[PLAY EVIDENCE]

### DEMO 2 — CLARIFY ONCE

User:

"Did the truck stop at the loading area?"

System:

"I don't have a saved definition for loading area."

User selects:

Lobby Camera + region

System stores it.

Query completes.

Restart backend.

Run same query again.

System must NOT ask again.

### DEMO 3 — CROSS CAMERA

User:

"Where did the same vehicle go after the gate?"

System:

Gate 09:14
→ Lobby 09:16
→ Rear 09:22

with evidence and confidence.

### DEMO 4 — RESEARCH

Show:

Baseline
vs
Full system

Then show ablation.

============================================================
## 31. 24-HOUR PRIORITY
============================================================

If time is limited, implement in EXACTLY this priority:

P0:
Recorded video ingestion

P0:
Multi-camera metadata

P0:
Visual indexing

P0:
Natural-language retrieval

P0:
Camera localization

P0:
Timestamp localization

P0:
Evidence frame

P0:
Evidence clip

P0:
Clarify-once memory

P0:
Persistence across restart

P0:
Baseline

P0:
Evaluation

P0:
Research contribution

P1:
UI polish

P1:
Temporal reranking

P1:
Visual verification

P2:
Cross-camera re-ID

P2:
Timeline reconstruction

P3:
Live ingestion

P3:
Alerts

P3:
Privacy

NEVER sacrifice P0 features to build P2/P3 features.

============================================================
## 32. WHEN SOMETHING IS IMPOSSIBLE
============================================================

Do NOT fake it.

When a model, library, dataset, GPU, or external service is unavailable:

1. determine whether a local alternative exists
2. use a practical fallback
3. preserve the interface
4. clearly document the fallback
5. continue building the rest

Do not block the entire project waiting for one dependency.

============================================================
## 33. ANTI-SLOP ENGINEERING RULES
============================================================

Do not:

- generate hundreds of meaningless files
- duplicate logic
- create unnecessary abstractions
- hard-code paths
- hard-code secrets
- use fake metrics
- use fake evidence
- create mock success paths and call them production
- rely on chat history for persistent memory
- claim cross-camera identity with no evidence
- claim research novelty without an ablation

Do:

- keep code modular
- use typed schemas
- add tests
- add logs
- document decisions
- keep model adapters swappable
- keep evaluation reproducible
- make evidence inspectable

============================================================
## 34. RALPH LOOP TASK SELECTION
============================================================

At the beginning of every iteration:

Read current project state.

Then classify work as:

DONE
IN PROGRESS
BLOCKED
BROKEN
MISSING
NEEDS IMPROVEMENT

Choose the highest-impact unresolved item.

Priority:

1. broken functionality
2. missing judging requirement
3. failing tests
4. grounding/retrieval quality
5. memory correctness
6. evaluation
7. latency
8. UI quality
9. stretch

Then implement it.

============================================================
## 35. NEVER STOP EARLY
============================================================

Do not stop because:

- the app starts
- the page looks good
- one query works
- embeddings work
- the API returns 200
- tests partially pass
- the MVP looks impressive

Continue until:

ALL CRITICAL REQUIREMENTS
are implemented and verified.

Then improve weak areas.

Then run the demo.

Then run evaluation.

Then inspect final UX.

Then verify documentation.

Then perform final cleanup.

============================================================
## 36. FINAL DEFINITION OF DONE
============================================================

Only declare COMPLETE when ALL are true:

[ ] recorded multi-camera footage supported
[ ] 3+ camera demo path works
[ ] indexing works
[ ] visual embeddings/index work
[ ] open-vocabulary search works
[ ] natural-language queries work
[ ] time filtering works
[ ] camera filtering works
[ ] camera localization works
[ ] timestamp localization works
[ ] evidence frame works
[ ] evidence clip works
[ ] grounded answers work
[ ] unsupported answers are prevented
[ ] clarify-once works
[ ] persistent memory works
[ ] restart persistence verified
[ ] baseline implemented
[ ] held-out queries implemented
[ ] metrics implemented
[ ] baseline measured
[ ] full system measured
[ ] ablation measured
[ ] research contribution documented
[ ] API tested
[ ] frontend tested
[ ] integration tests pass
[ ] demo runbook exists
[ ] evaluation report exists
[ ] final implementation status exists
[ ] secrets checked
[ ] startup instructions verified
[ ] final demo manually verified

============================================================
## 37. FINAL OUTPUT
============================================================

At the end, provide a concise completion report containing:

1. What was implemented
2. What models were used
3. What baseline was used
4. What research contribution was implemented
5. Baseline metrics
6. Full-system metrics
7. Ablation results
8. Query latency
9. Clarify-once persistence verification
10. Cross-camera status
11. Remaining limitations
12. Exact commands to start the project
13. Exact commands to run indexing
14. Exact commands to run evaluation
15. Exact commands to run the demo

Do not claim anything that was not actually tested.

============================================================
## 38. START NOW
============================================================

BEGIN.

First inspect the repository.

Then inspect all project-local skills/instructions.

Then inspect the existing application.

Then build the project.

Do not merely tell me how to build it.

BUILD IT.

Continue using the RALPH LOOP:

REVIEW
→ ACT
→ TEST
→ FIX
→ HARDEN
→ REPEAT

until the judging-ready implementation is actually working.

# END OF RALPH LOOP PROMPT