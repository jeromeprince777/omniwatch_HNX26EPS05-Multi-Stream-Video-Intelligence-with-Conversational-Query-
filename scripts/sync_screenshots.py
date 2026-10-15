"""Copies captured UI screenshots from the agent brain directory to docs/screenshots."""

import glob
import os
import shutil

SRC_DIR = r"C:\Users\ffgam\.gemini\antigravity-ide\brain\974cade4-d0df-4c2c-bb0b-1139c78c40c1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(REPO_ROOT, "docs", "screenshots")

os.makedirs(DEST_DIR, exist_ok=True)

# Pre-existing brain screenshots (disabled to preserve canonical Playwright captures)
MAPPINGS = []


copied = 0
for pattern, dest_name in MAPPINGS:
    search_pattern = os.path.join(SRC_DIR, pattern)
    matches = glob.glob(search_pattern)
    if matches:
        # Use latest match
        latest = max(matches, key=os.path.getmtime)
        dest_path = os.path.join(DEST_DIR, dest_name)
        shutil.copy2(latest, dest_path)
        print(f"[OK] Copied {os.path.basename(latest)} -> docs/screenshots/{dest_name}")
        copied += 1
    else:
        print(f"[WARN] No match found for {pattern}")

print(f"\n[SUMMARY] Successfully copied {copied} screenshots into {DEST_DIR}")
