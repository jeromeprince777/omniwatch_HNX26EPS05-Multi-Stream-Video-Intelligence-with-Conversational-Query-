#!/usr/bin/env bash
# Ingest multi-camera footage from manifest.yaml (Bash)
set -euo pipefail

MANIFEST_FILE="${1:-manifest.yaml}"

if [[ ! -f "$MANIFEST_FILE" && "$MANIFEST_FILE" != "--help" && "$MANIFEST_FILE" != "-h" ]]; then
  echo "ERROR: Manifest file '$MANIFEST_FILE' not found." >&2
  exit 1
fi

echo "[OmniWatch] Starting multi-camera video ingestion pipeline..."
python -m omniwatch.ingest.pipeline "$@"

