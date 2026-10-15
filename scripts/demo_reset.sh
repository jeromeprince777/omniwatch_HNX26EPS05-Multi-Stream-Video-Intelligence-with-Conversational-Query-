#!/usr/bin/env bash
# Reset only the demo site rows and caches; refuse to touch eval/
set -euo pipefail

TARGET_PATH="${1:-data/cache}"

if [[ "$TARGET_PATH" =~ "eval" ]]; then
  echo "ERROR: Refusing to reset or touch eval/ directory! (AGENTS.md contract)" >&2
  exit 1
fi

echo "[OmniWatch] Resetting demo cache and demo database..."
python -c "
import shutil, os, sqlite3
cache_dir = 'data/cache'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
db_path = 'data/omniwatch.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    # Reset only demo site tables, preserving schema
    for tbl in ['observations', 'tracklets', 'cameras', 'sites', 'clarification_log', 'pending_queries', 'query_log']:
        try:
            conn.execute(f'DELETE FROM {tbl}')
        except Exception:
            pass
    conn.commit()
    conn.close()
print('Demo cache and tables successfully reset.')
"
