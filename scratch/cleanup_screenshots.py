import os
from pathlib import Path

DIR = Path(r"c:\Users\ffgam\OneDrive\Desktop\omniwatch_pack\docs\screenshots")
OLD_FILES = [
    "01_search_console_tactical_hud.png",
    "02_tours_bag_search_dossier.png",
    "03_camera_wall_grid.png",
    "04_clarify_once_spatial_memory.png",
    "05_clarify_once_roi_canvas_modal.png",
    "06_evidence_keyframe_reticle_modal.png",
    "07_evidence_video_clip_modal.png",
    "08_standing_alerts_audit_feed.png"
]

for f in OLD_FILES:
    p = DIR / f
    if p.exists():
        p.unlink()
        print(f"Removed old file: {f}")

print("Cleaned up old screenshot duplicates.")
