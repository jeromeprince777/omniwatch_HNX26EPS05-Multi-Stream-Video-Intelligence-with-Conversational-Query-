# OmniWatch — Judging Demonstration Runbook (HNX26EPS05)

This runbook guides operators and judges through demonstrating OmniWatch's core capabilities in accordance with the HackNex 2026 rubric.

---

## 1. Quick Start Commands

Ensure your terminal is in the repository root with the environment activated:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Start the OmniWatch Server & UI (http://127.0.0.1:8000/)
.\scripts\serve.ps1
```

Or using bash:
```bash
# Linux / macOS / Git Bash
source .venv/bin/activate
./scripts/serve.sh
```

Navigate your browser to **`http://127.0.0.1:8000/`** to view the Tactical Surveillance HUD.

---

## 2. Interactive Demo Scenarios

### DEMO 1 — Open-Vocabulary Grounded Search

**Objective**: Demonstrate that the system understands open-vocabulary natural language concepts, localizes the exact camera and timestamp, and provides verifiable visual evidence without hallucination.

1. **Enter Query**:
   ```text
   Did a red car pass through the main gate in the last hour?
   ```
2. **Observe System Response**:
   - **Answer Banner**: `YES` — displays exact camera label (`Main Gate` / `cam1`), timestamp (`09:14:32`), and match score (`0.88+`).
   - **Grounded Evidence**:
     - Visual crop thumbnail with green bounding box around the vehicle.
     - Match metadata breakdown: `Class: car`, `Color: red`, `Detector: YOLO-World-S`.
     - Inspection telemetry: Latency `<25ms` (fast mode).
3. **Play Evidence Clip**:
   - Click **`PLAY EVIDENCE`** on the dossier card.
   - The on-demand video player opens, playing the exact 10-second contextual video clip around the event.
   - Adjust playback speed (0.5x, 1x, 2x) or download the MP4 clip for chain of custody.

---

### DEMO 2 — Clarify-Once Persistent Memory

**Objective**: Demonstrate that unknown physical location referents trigger an interactive clarification, commit permanently to SQLite, and persist across an application restart.

1. **Enter Query with Unknown Referent**:
   ```text
   Did a vehicle stop at the loading dock?
   ```
2. **Observe Clarification Card**:
   - OmniWatch detects that `loading dock` has no registered definition.
   - The engine **refuses to guess** or hallucinate.
   - An interactive Clarification Modal appears with thumbnails of all registered cameras.
3. **Map the Referent**:
   - Click on **`Main Gate`** (or `East Corridor`).
   - *(Optional ROI)*: Drag a rectangular bounding box across the entrance zone on the canvas.
   - Click **`Confirm & Store Referent`**.
4. **Observe Query Auto-Resume**:
   - The original query immediately resumes execution against the newly mapped camera/region.
   - An evidence dossier is returned for vehicles in that zone.
5. **Verify Restart Persistence (The Kill Test)**:
   - In the terminal running the server, press `Ctrl+C` to terminate the process.
   - Start the server again:
     ```powershell
     .\scripts\serve.ps1
     ```
   - Refresh the browser at `http://127.0.0.1:8000/`.
   - Submit the exact same query again:
     ```text
     Did a vehicle stop at the loading dock?
     ```
   - **Result**: The system resolves `loading dock` **silently and immediately**. No clarification card appears!
   - Navigate to the **`Clarify-Once KB`** tab on the left rail to view, inspect, or edit the persisted mapping in SQLite.

---

### DEMO 3 — Cross-Camera Continuity & Re-ID

**Objective**: Demonstrate tracking a vehicle or person across multiple non-overlapping camera streams with temporal feasibility gating.

1. **Enter Continuity Query**:
   ```text
   Where did the vehicle go after passing the main gate?
   ```
2. **Observe Trajectory Dossier**:
   - The system detects path intent, retrieves the initial tracklet at `Main Gate`, and queries downstream cameras within the temporal feasibility window ($\Delta t \in [0, 300\text{s}]$).
   - Displays a multi-camera hop timeline:
     $$\text{Gate Camera (09:14:32)} \longrightarrow \text{East Corridor (09:14:48)}$$
   - Displays transit interval ($\Delta t = 16\text{s}$), OpenCLIP visual appearance cosine similarity, and thumbnails for each sequential hop.
   - Click **`Trace Cross-Camera Path`** on any search result to inspect its continuous journey.

---

### DEMO 4 — Research Contribution & Ablation Validation

**Objective**: Verify the measured performance gains against the baseline and confirm all numbers strictly originate from `eval/results/`.

1. **Run Number Verification Script**:
   ```powershell
   python scripts/check_writeup_numbers.py
   ```
   **Output**: Confirms that 100% of the numbers in `WRITEUP.md` match `eval/results/eval_test_20261008_150316/summary.json` and `eval_dev_20261008_150043/summary.json`.

2. **Benchmark Comparison Summary**:
   - **B1 (Standard Baseline)**: Raw frame OpenCLIP embeddings -> Cosine similarity.
     - Hit@1 ($\tau=15\text{s}$): $22.6\%$ | MRR: $0.148$ | Negative FPR: $100.0\%$ (hallucinates on negatives).
   - **B1+ (Fairness Check)**: Baseline + OmniWatch query parser and filters.
     - Hit@1 ($\tau=15\text{s}$): $25.8\%$ | MRR: $0.190$ | Negative FPR: $40.0\%$.
   - **OW-fast (Full Pipeline)**: Tracklets + Explicit Color/Carrying Attribute Fusion.
     - Hit@1 ($\tau=15\text{s}$): $25.8\%$ | Hit@5: $41.9\%$ | MRR: **$0.267$** | Latency p50: **$17.8\text{ ms}$**.
   - **OW-full (Verified)**: Pipeline + Verification Pass.
     - Existence Accuracy: **$90.2\%$** | Negative FPR: **$40.0\%$**.

---

## 3. Ingesting Unseen Judge Footage

When judges provide unseen video files during evaluation:

1. **Place Video Files in a Directory**:
   ```text
   data/unseen_judge/
       gate_camera.mp4
       lobby_camera.mp4
   ```
2. **Create a `manifest.yaml`**:
   ```yaml
   site_id: judge_unseen_site
   reference_now: null
   cameras:
     - camera_id: gate_cam
       label: "Judge Gate"
       video: data/unseen_judge/gate_camera.mp4
       start_time: "2026-10-08T09:00:00+00:00"
     - camera_id: lobby_cam
       label: "Judge Lobby"
       video: data/unseen_judge/lobby_camera.mp4
       start_time: "2026-10-08T09:00:00+00:00"
   ```
3. **Execute Ingestion**:
   ```powershell
   python -m omniwatch.ingest.pipeline manifest.yaml --preset fast
   ```
   Or on bash:
   ```bash
   ./scripts/ingest.sh manifest.yaml --preset fast
   ```
4. **Monitor Ingestion**:
   - Check status at `http://127.0.0.1:8000/api/ingest/status` or watch console progress bar.
5. **Immediately Query Footage**:
   - As soon as ingestion reports completed, submit natural language queries in the web UI.
