"""Copies captured UI screenshots from the agent brain directory to docs/screenshots."""

import glob
import os
import shutil

SRC_DIR = r"C:\Users\ffgam\.gemini\antigravity-ide\brain\974cade4-d0df-4c2c-bb0b-1139c78c40c1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(REPO_ROOT, "docs", "screenshots")

os.makedirs(DEST_DIR, exist_ok=True)

MAPPINGS = [
    ("search_console_initial_*.png", "01_search_console_tactical_hud.png"),
    ("tours_bag_search_result_*.png", "02_tours_bag_search_dossier.png"),
    ("camera_wall_view_*.png", "03_camera_wall_grid.png"),
    ("clarify_once_kb_view_*.png", "04_clarify_once_spatial_memory.png"),
    ("clarify_once_modal_roi_*.png", "05_clarify_once_roi_canvas_modal.png"),
    ("evidence_dossier_modal_frame_*.png", "06_evidence_keyframe_reticle_modal.png"),
    ("evidence_dossier_modal_video_*.png", "07_evidence_video_clip_modal.png"),
    ("standing_alerts_view_*.png", "08_standing_alerts_audit_feed.png"),
    ("omniwatch_full_demo_*.webp", "09_interactive_demo_session.webp"),
]

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
