import json
import sqlite3

conn = sqlite3.connect("data/omniwatch.db")
cur = conn.cursor()

# Get sample cars on public_surveillance
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'public_surveillance' AND cls = 'car' ORDER BY t_start LIMIT 10")
car_rows = cur.fetchall()

# Get sample persons on cam1
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'cam1' AND cls = 'person' LIMIT 3")
cam1_p = cur.fetchall()

# Get sample bus on cam1
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'cam1' AND cls = 'bus' LIMIT 1")
cam1_b = cur.fetchall()

# Get sample bus on cam2
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'cam2' AND cls = 'bus' LIMIT 1")
cam2_b = cur.fetchall()

# Get bicycles on public_surveillance
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'public_surveillance' AND cls = 'bicycle'")
bikes = cur.fetchall()

# Get trucks on public_surveillance
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'public_surveillance' AND cls = 'truck'")
trucks = cur.fetchall()

# Get persons on yokohama
cur.execute("SELECT t_start, t_end FROM tracklets WHERE camera_id = 'yokohama_cctv' AND cls = 'person' LIMIT 5")
yoko_p = cur.fetchall()

queries = []

# 1-3. Cars on public_surveillance
for idx, (t0, t1) in enumerate(car_rows[:4]):
    queries.append({
        "id": f"dev_car_{idx+1:03d}",
        "text": "find a car on public surveillance",
        "type": "find",
        "gt": [{"camera_id": "public_surveillance", "t_start": t0, "t_end": t1}],
        "expects_no_match": False,
        "needs_referents": [],
    })

# 4-6. Persons on cam1
for idx, (t0, t1) in enumerate(cam1_p):
    queries.append({
        "id": f"dev_p_cam1_{idx+1:03d}",
        "text": "show me a person at main gate",
        "type": "find",
        "gt": [{"camera_id": "cam1", "t_start": t0, "t_end": t1}],
        "expects_no_match": False,
        "needs_referents": ["main gate"],
    })

# 7. Bus on cam1
if cam1_b:
    queries.append({
        "id": "dev_bus_cam1_001",
        "text": "did a bus appear on camera 1?",
        "type": "exists",
        "gt": [{"camera_id": "cam1", "t_start": cam1_b[0][0], "t_end": cam1_b[0][1]}],
        "expects_no_match": False,
        "needs_referents": [],
    })

# 8. Bus on cam2
if cam2_b:
    queries.append({
        "id": "dev_bus_cam2_001",
        "text": "did a bus appear at east corridor?",
        "type": "exists",
        "gt": [{"camera_id": "cam2", "t_start": cam2_b[0][0], "t_end": cam2_b[0][1]}],
        "expects_no_match": False,
        "needs_referents": ["east corridor"],
    })

# 9-11. Bicycles
for idx, (t0, t1) in enumerate(bikes[:3]):
    queries.append({
        "id": f"dev_bike_{idx+1:03d}",
        "text": "find a bicycle",
        "type": "find",
        "gt": [{"camera_id": "public_surveillance", "t_start": t0, "t_end": t1}],
        "expects_no_match": False,
        "needs_referents": [],
    })

# 12-13. Trucks
for idx, (t0, t1) in enumerate(trucks[:2]):
    queries.append({
        "id": f"dev_truck_{idx+1:03d}",
        "text": "show me a truck",
        "type": "find",
        "gt": [{"camera_id": "public_surveillance", "t_start": t0, "t_end": t1}],
        "expects_no_match": False,
        "needs_referents": [],
    })

# 14-18. Persons on Yokohama
for idx, (t0, t1) in enumerate(yoko_p[:5]):
    queries.append({
        "id": f"dev_p_yoko_{idx+1:03d}",
        "text": "find a person on yokohama cctv",
        "type": "find",
        "gt": [{"camera_id": "yokohama_cctv", "t_start": t0, "t_end": t1}],
        "expects_no_match": False,
        "needs_referents": [],
    })

# 19-22. Generic object queries
for idx, (t0, t1) in enumerate(car_rows[4:8]):
    queries.append({
        "id": f"dev_vehicle_gen_{idx+1:03d}",
        "text": "find a vehicle",
        "type": "find",
        "gt": [{"camera_id": "public_surveillance", "t_start": t0, "t_end": t1}],
        "expects_no_match": False,
        "needs_referents": [],
    })

# 23-30. Negatives (>= 20% of dataset = 8 queries)
negatives = [
    ("dev_neg_001", "did a flying spaceship appear?", "exists"),
    ("dev_neg_002", "find an elephant wearing sunglasses", "find"),
    ("dev_neg_003", "show me a giraffe on camera 1", "find"),
    ("dev_neg_004", "find a neon green bus", "find"),
    ("dev_neg_005", "show me a helicopter at camera 2", "find"),
    ("dev_neg_006", "did a submarine pass through the main gate?", "exists"),
    ("dev_neg_007", "show me a purple bicycle on camera 1", "find"),
    ("dev_neg_008", "find a yellow motorcycle on camera 2", "find"),
]

for n_id, n_text, n_type in negatives:
    queries.append({
        "id": n_id,
        "text": n_text,
        "type": n_type,
        "gt": [],
        "expects_no_match": True,
        "needs_referents": [],
    })

print(f"Total dev queries created: {len(queries)} (Positives: {len(queries)-len(negatives)}, Negatives: {len(negatives)} = {len(negatives)/len(queries)*100:.1f}%)")

with open("eval/queries/dev.jsonl", "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")

print("Saved to eval/queries/dev.jsonl")
conn.close()
