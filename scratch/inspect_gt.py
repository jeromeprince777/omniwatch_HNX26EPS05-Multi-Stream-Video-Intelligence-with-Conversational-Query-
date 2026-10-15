import sqlite3
import json

conn = sqlite3.connect("data/omniwatch.db")
cur = conn.cursor()

# Get camera start times
cur.execute("SELECT camera_id, label, start_time, duration FROM cameras")
cams = cur.fetchall()
print("Cameras:")
for c in cams:
    print(f"  {c[0]} ({c[1]}): start={c[2]}, dur={c[3]}")

# Get sample vehicles
print("\nSample vehicles:")
cur.execute("SELECT camera_id, cls, t_start, t_end, attrs_json FROM tracklets WHERE cls in ('car', 'bus', 'truck', 'bicycle') LIMIT 10")
for r in cur.fetchall():
    cam, cls, t0, t1, attrs = r
    d = json.loads(attrs) if attrs else {}
    print(f"  {cam} | {cls} | [{t0:.1f}, {t1:.1f}] | color={d.get('dominant_color')}")

# Get sample persons with carries or colors
print("\nSample persons:")
cur.execute("SELECT camera_id, cls, t_start, t_end, attrs_json FROM tracklets WHERE cls = 'person' LIMIT 10")
for r in cur.fetchall():
    cam, cls, t0, t1, attrs = r
    d = json.loads(attrs) if attrs else {}
    print(f"  {cam} | {cls} | [{t0:.1f}, {t1:.1f}] | color={d.get('dominant_color')} | carries={d.get('carries')}")

conn.close()
