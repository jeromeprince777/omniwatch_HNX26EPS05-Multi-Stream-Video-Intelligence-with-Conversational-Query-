# 09 — VLM verification (P1) and latency modes
**Read:** docs/01 (5, 7), docs/02 (2, 3, 5).

**Task:**
1. `query/rerank.py`: take top-K candidate tracklets (K from config), load their best crops, ask a local open VLM a yes/no question ("Does this image show {object_phrase with attributes}? Answer Yes or No.") and compute P(Yes) from the next-token probabilities of "Yes"/"No" (verify how the chosen library exposes logits). Fuse with the retrieval score using weights from config. Batch the crops. Load the VLM lazily and keep it in memory. If the model fails to load, WARN, expose in `/api/health`, and serve fast mode.
2. Modes: `fast` (no verify) and `verified`. UI badge and `/api/chat` `mode` parameter already exist; wire them. Optionally auto-verify only when the top candidates are ambiguous (margin below threshold); make this a config flag, default off, and evaluate it separately if used.
3. Calibration script `eval/tune_dev.py`: grid-search fusion weights and thresholds **on dev only**, writing the chosen values to `config.yaml` and `eval/results/tuning_log.json`. Show the dev metric surface so overfitting is visible. No test split access in this script (assert).
4. Latency breakdown: measure parse, retrieve, verify, and total with warm models; record in results.

**Acceptance:** run fast vs verified on dev queries and show the real metrics and latency from your run. If verification does not improve dev accuracy, report that plainly; do not tune until it does on test. Include one example where verification fixed a wrong ranking and one where it hurt, with evidence ids.
