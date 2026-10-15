import sqlite3
import os
import httpx
from datetime import datetime, timezone, timedelta

DB_PATH = "data/omniwatch.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Get 10 tracklets
cur.execute("""
    SELECT t.tracklet_id, t.camera_id, t.cls, t.t_start, t.t_end, t.best_obs_id, c.video_path, c.start_time
    FROM tracklets t
    JOIN cameras c ON t.camera_id = c.camera_id
    ORDER BY t.tracklet_id ASC
    LIMIT 10
""")
rows = cur.fetchall()

print(f"Checking {len(rows)} tracklets for timestamp correctness:")
for r in rows:
    trk_id, cam_id, cls, t_start, t_end, best_obs_id, v_path, c_start = r
    # Check best observation
    cur.execute("SELECT ts, x1, y1, x2, y2, frame_idx FROM observations WHERE obs_id = ?", (best_obs_id,))
    obs = cur.fetchone()
    obs_ts, x1, y1, x2, y2, f_idx = obs if obs else (None, 0, 0, 0, 0, 0)
    
    # Check if obs_ts is inside [t_start, t_end]
    inside = (t_start <= obs_ts <= t_end) if obs_ts else False
    
    # Check clip and thumb API availability
    thumb_url = f"http://127.0.0.1:8000/api/results/{trk_id}/thumb"
    clip_url = f"http://127.0.0.1:8000/api/results/{trk_id}/clip"
    
    r_th = httpx.get(thumb_url, timeout=5)
    r_cl = httpx.get(clip_url, timeout=15)
    
    print(f"Tracklet #{trk_id} ({cls} on {cam_id}):")
    print(f"  t_start={t_start:.1f}, t_end={t_end:.1f}, duration={t_end - t_start:.2f}s")
    print(f"  best_obs_ts={obs_ts:.1f} (inside range: {inside})")
    print(f"  thumb: HTTP {r_th.status_code} ({len(r_th.content)} bytes), clip: HTTP {r_cl.status_code} ({len(r_cl.content)} bytes)")

conn.close()
