import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0)

queries = [
    "yokohama_cctv news reader",
    "news reader",
    "red car",
    "person carrying a bag",
    "bus on public surveillance",
    "anyone walking",
]

for q in queries:
    r = client.post("/api/chat", json={"message": q, "mode": "fast", "session_id": "bench"})
    data = r.json()
    t = data.get("type")
    n = len(data.get("results", []))
    print(f"Query: {q}")
    print(f" -> Type: {t}, Results: {n}")
    if data.get("results"):
        top = data["results"][0]
        print(f"    Top: {top['camera_id']} score={top['score']} tracklet={top['tracklet_id']}")
    else:
        print(f"    Summary: {data.get('summary')}")
