import sqlite3

conn = sqlite3.connect("data/omniwatch.db")
cur = conn.cursor()
cur.execute("SELECT tracklet_id, cls, n_obs, best_obs_id, t_start, t_end FROM tracklets WHERE camera_id = 'yokohama_cctv' ORDER BY tracklet_id")
rows = cur.fetchall()
print(f"Total tracklets in yokohama_cctv: {len(rows)}")
for r in rows:
    cur.execute("SELECT x1, y1, x2, y2 FROM observations WHERE obs_id = ?", (r[3],))
    b = cur.fetchone()
    print(r[0], r[1], f"obs={r[2]}", f"t={r[4]:.1f}-{r[5]:.1f}", [round(x, 2) for x in b])
