# 01 — Scaffold the repository
**Read:** AGENTS.md, docs/00_PRD.md, docs/01_ARCHITECTURE.md (sections 2, 3).

**Task:** Create the project skeleton exactly as in the architecture layout.
1. Python 3.11 virtualenv instructions, `requirements.txt` (empty to start, filled as packages are verified), `pyproject.toml` with ruff config, `.gitignore` (data/, models/ weights, *.mp4, *.faiss, *.npy, *.db, cache/).
2. `omniwatch/config.py`: typed settings loaded from `config.yaml` with presets `fast|balanced|accurate` (sample_fps, detector size/vocab, encoder choice, verify top-K, thresholds). Defaults must be obviously labelled as unmeasured placeholders to be tuned on the dev set.
3. `omniwatch/db.py`: SQLite connection helper (WAL, `synchronous=FULL`, foreign keys) and a migration that creates every table in docs/01 section 3. Add `tests/test_db.py`.
4. Empty modules with docstrings stating their responsibility and the interfaces they must expose (no fake logic).
5. `scripts/`: `ingest.sh`, `serve.sh`, `eval.sh`, `demo_reset.sh` (reset only the demo site rows and caches; refuse to touch eval/).
6. `models/MANIFEST.md` template: name, source, version/hash, licence note, date downloaded.
7. A `benchmark/hardware_probe.py` that prints: GPU/CPU info, torch device availability, ffmpeg presence, decode speed on a sample video, and writes `benchmark/hardware.json`.

**Acceptance:** `pytest -q` passes; `python -m omniwatch.db --init data/test.db` creates all tables; `python benchmark/hardware_probe.py` runs and writes JSON. Report the output of the probe.
