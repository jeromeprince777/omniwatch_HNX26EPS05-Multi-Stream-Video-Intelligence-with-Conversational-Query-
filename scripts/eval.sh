#!/usr/bin/env bash
# Run automated evaluation and ablation benchmark
set -euo pipefail

QUERIES="${1:-eval/queries/dev.jsonl}"
OUT_DIR="${2:-eval/results}"

echo "[OmniWatch] Running evaluation harness on $QUERIES -> $OUT_DIR"
python -m eval.run_eval --queries "$QUERIES" --out "$OUT_DIR"
