# 04 — Data plan

## 1. What we need
Several **non-overlapping** cameras with overlapping time ranges, objects that recur across cameras, and enough variety (colors, bags, vehicles, day/night) to write 70+ queries with known answers. Teams use their own footage or public datasets.

## 2. Options (verify access, licence and size yourselves before committing; availability can change)
| Source | Why it fits | Watch out for |
|---|---|---|
| **Own campus footage** (phones/CCTV at 3-5 spots: gate, lobby, corridor, parking) | Full control; real "main gate" referent; you know ground truth; no licence issue | Needs hour(s) to record; get permission; faces/privacy; keep it short |
| **MTMMC** (large multi-camera tracking benchmark, CVPR 2024) | Many cameras, real-world, multi-target | Large download; check terms |
| **CityFlow / CityFlow-NL** (AI City Challenge) | Vehicles across cameras; NL descriptions exist, close to this task | Registration/licence agreement usually required; vehicle-only |
| **NLPR_MCT** | Non-overlapping multi-camera tracking | Older, smaller |
| **MEVA** | Many cameras and annotated activities | Very large; check access method |
| WILDTRACK | Multi-camera pedestrians | Cameras overlap, not ideal for the non-overlapping story |
Avoid datasets that have been withdrawn for privacy reasons.
**Recommendation:** primary = own campus recording (guarantees a gate/lobby story and fast labeling), secondary = a small slice of one public set so the results are not only on our own scenes. Decide in hour 0 and write the choice in the write-up.

## 3. Recording protocol (if own footage)
- 3-5 phones/cameras fixed on tripods, different locations, non-overlapping views, same start (clap or phone clock screenshot) to sync.
- 20-40 minutes total is enough. Script a **scenario sheet**: who walks where, which vehicle passes which gate at which time, who carries a large bag, a red car (or red bike), a person returning through a second camera, plus several "nothing happens" periods for negatives.
- Log scenario events in a CSV as you record: `time, camera, event`. That CSV is your first ground truth.
- Get consent from participants; do not publish faces.

## 4. Manifest format (`manifest.yaml`)
```yaml
site_id: campus_demo
reference_now: null            # optional ISO time; default = end of latest footage
cameras:
  - camera_id: cam1
    label: null                # optional human name if known; leave null to force clarify-once
    video: data/campus/cam1.mp4
    start_time: "2026-10-08T09:00:00+05:30"
  - camera_id: cam2
    video: data/campus/cam2.mp4
    start_time: "2026-10-08T09:00:12+05:30"
```
Rules: `start_time` is required unless ingest runs with `--assume-synced` (all cameras start at one shared t0). Video files may be split into parts: list them as multiple entries with the same `camera_id` and ordered start times (ingest concatenates by time).

## 5. Ground-truth labeling
- Use the scenario CSV plus a second pass: scrub each camera at 5x speed and add events you did not script.
- GT row = `camera_id, t_start, t_end (absolute or offset from camera start), description, query ids it answers`.
- Label **negatives** deliberately: pick (camera, window) pairs where you verified nothing matching occurs.
- Dev/test split by video segment (for example first 60% of each camera for dev, last 40% for test), not by random frames. Never tune on test segments.
- Two people label independently for 10 queries; disagreement above tau means the query is ambiguous, rewrite it.

## 6. Storage and hygiene
`data/` and `models/` in `.gitignore`. Keep a `data/README.md` listing sources, licence notes and recording dates. Keep a copy of the raw footage on a second device (demo-day insurance).
