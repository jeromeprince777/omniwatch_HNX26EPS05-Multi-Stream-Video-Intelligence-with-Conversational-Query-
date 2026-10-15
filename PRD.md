# PRD — HNX26EPS05
# Multi-Stream Video Intelligence with Conversational Query

**Project ID:** HNX26EPS05  
**Project Type:** AI / Computer Vision / Video Intelligence / Conversational Search  
**Primary Mode:** Recorded multi-camera CCTV footage  
**Target:** HackNex 2026 Problem Statement  
**Development Method:** Ralph Loop + Antigravity  
**Document Status:** Implementation Source of Truth

---

# 1. Executive Summary

Build an intelligent multi-camera video search and investigation platform that allows a user to connect multiple CCTV recordings and ask natural-language questions about events occurring across those cameras.

Instead of manually searching each CCTV stream, the system continuously indexes recorded footage and creates a searchable visual-temporal representation.

The user can ask questions such as:

> Did a red car pass through the main gate in the last hour?

or:

> Find the person carrying a large bag.

or:

> Where did the same vehicle go after leaving the gate?

The system must return an answer grounded in actual recorded evidence.

Every successful answer must provide:

- Camera
- Timestamp
- Relevant time interval
- Evidence frame
- Playable video clip
- Confidence/relevance information

The system must also support a persistent clarification mechanism.

For example, if the user asks:

> Did a vehicle stop at the main gate?

and the system does not know what "main gate" refers to, it asks the user once.

The user selects the appropriate camera or region.

The mapping is then stored permanently.

If the user later asks another question involving "main gate", the system must resolve it automatically without asking again.

This mapping must survive application restart.

The system should also support cross-camera continuity as a stretch capability, allowing the system to reconstruct a likely path of a vehicle/person across multiple cameras.

---

# 2. Problem Statement

## HNX26EPS05

### Multi-Stream Video Intelligence with Conversational Query

Plug in multiple CCTV streams. The system continuously detects and indexes events across all cameras and exposes a conversational interface where a user describes what they want.

Example:

> "Did a red car pass through the main gate in the last hour?"

The system should answer:

> Yes. A red vehicle appears in Gate Camera around 09:14:32.

And provide:

- Gate Camera
- 09:14:32
- relevant evidence frame
- playable evidence clip

---

# 3. Why This Problem Is Difficult

There are four primary technical challenges.

## 3.1 Open-Vocabulary Search

Queries are not limited to a fixed detector class list.

Examples:

- red car
- person carrying a large bag
- person wearing a white shirt
- black SUV
- someone running toward the entrance
- vehicle that stopped briefly
- person walking with a backpack

This requires semantic visual retrieval, open-vocabulary detection, VLM reasoning, or a hybrid system.

A closed detector with only:

- person
- car
- bicycle

is insufficient as the only retrieval mechanism.

---

# 3.2 Grounded and Localized Answers

The system must not merely generate a plausible answer.

Every answer must be traceable to a source.

The answer must identify:

- camera
- timestamp
- evidence
- clip

No source evidence means no valid success.

The system must prioritize evidence over generated prose.

---

# 3.3 Clarify Once, Then Remember

When a query contains an unknown referent such as:

- main gate
- loading area
- parking entrance
- warehouse side
- rear entrance

the system must ask the user to identify it.

The user should be able to:

- select a camera
- optionally select a region/ROI
- provide a description

The system must persist this knowledge.

The mapping must remain available after:

- page refresh
- frontend restart
- backend restart
- application restart

The system must NOT rely on chat history for this capability.

---

# 3.4 Cross-Camera Continuity

A person or vehicle can move through multiple cameras.

Example:

```text
Gate Camera   09:14
      ↓
Lobby Camera  09:16
      ↓
Rear Camera   09:22

## IMPORTANT IMPLEMENTATION RULE

Do NOT create placeholder, mock, stub, pseudo, or documentation-only implementations for mandatory requirements.

For every P0 requirement, implement the real functional path and verify it with an actual test.

A file such as `scripts/ingest.sh` does not count as video ingestion unless it actually:
1. discovers or accepts recorded video files,
2. validates them,
3. extracts metadata,
4. registers cameras/videos,
5. decodes frames,
6. preserves timestamps,
7. reports success/failure.

Do not mark a requirement complete merely because a file or interface was created.