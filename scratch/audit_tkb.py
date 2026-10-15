import httpx
import json
import sqlite3

BASE_URL = "http://127.0.0.1:8000"
DB_PATH = "data/omniwatch.db"

def test_clarify_once():
    results = {}
    
    # Check DB initial state
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("--- 1. Query with no referent (T-KB-5) ---")
    r = httpx.post(f"{BASE_URL}/api/chat", json={"session_id": "tkb_5", "message": "show me a person", "mode": "fast"}, timeout=20)
    d = r.json()
    t5_pass = d.get("type") in ["answer", "no_match"]
    results["T-KB-5 (no referent never asks)"] = "PASS" if t5_pass else f"FAIL (type={d.get('type')})"
    print("Result:", results["T-KB-5 (no referent never asks)"])

    print("--- 2. Unknown referent -> single clarification card + pending query stored (T-KB-1) ---")
    # Clean up test referent if exists
    cur.execute("DELETE FROM referents WHERE phrase_norm LIKE '%secret bunker%'")
    cur.execute("DELETE FROM pending_queries WHERE session_id = 'tkb_1'")
    conn.commit()

    r = httpx.post(f"{BASE_URL}/api/chat", json={"session_id": "tkb_1", "message": "did a person enter the secret bunker?", "mode": "fast"}, timeout=20)
    d = r.json()
    t1_pass = (d.get("type") == "clarification" and "secret bunker" in d.get("phrase", "").lower())
    
    # Check pending query in DB
    cur.execute("SELECT session_id, message FROM pending_queries WHERE session_id = 'tkb_1'")
    row_pending = cur.fetchone()
    pending_stored = (row_pending is not None and row_pending[0] == "tkb_1")
    results["T-KB-1 (unknown referent card + pending)"] = "PASS" if (t1_pass and pending_stored) else f"FAIL (type={d.get('type')}, pending={pending_stored})"
    print("Result:", results["T-KB-1 (unknown referent card + pending)"])

    print("--- 3. Clarification submit -> resumes and answers (T-KB-2) ---")
    if t1_pass:
        clar_id = d.get("clarification_id")
        r_clar = httpx.post(
            f"{BASE_URL}/api/clarify",
            json={"session_id": "tkb_1", "clarification_id": clar_id, "targets": [{"camera_id": "cam1"}]},
            timeout=20,
        )
        d_clar = r_clar.json()
        t2_pass = (d_clar.get("type") in ["answer", "no_match"])
        results["T-KB-2 (resumes without retyping)"] = "PASS" if t2_pass else f"FAIL (type={d_clar.get('type')})"
    else:
        results["T-KB-2 (resumes without retyping)"] = "SKIPPED (T-KB-1 failed)"
    print("Result:", results["T-KB-2 (resumes without retyping)"])

    print("--- 4. Phrase variants of known referent ('Secret Bunker', 'the secret bunker camera', 'secret-bunker') (T-KB-4) ---")
    variants = [
        "did a person enter Secret Bunker?",
        "did a person enter the secret bunker camera?",
        "did a person enter secret-bunker?",
    ]
    var_passes = []
    for v in variants:
        r_var = httpx.post(f"{BASE_URL}/api/chat", json={"session_id": "tkb_var", "message": v, "mode": "fast"}, timeout=20)
        d_var = r_var.json()
        is_p = (d_var.get("type") in ["answer", "no_match"])
        var_passes.append((v, is_p, d_var.get("type")))
    t4_pass = all(p[1] for p in var_passes)
    results["T-KB-4 (variants match known referent)"] = "PASS" if t4_pass else f"FAIL ({var_passes})"
    print("Result:", results["T-KB-4 (variants match known referent)"])

    print("--- 5. Two unknown phrases in one query (T-KB-6) ---")
    cur.execute("DELETE FROM referents WHERE phrase_norm IN ('zone alpha', 'zone beta')")
    cur.execute("DELETE FROM pending_queries WHERE session_id = 'tkb_6'")
    conn.commit()
    r6_1 = httpx.post(f"{BASE_URL}/api/chat", json={"session_id": "tkb_6", "message": "find a car at zone alpha and near zone beta", "mode": "fast"}, timeout=20)
    d6_1 = r6_1.json()
    print("First call response:", d6_1.get("type"), d6_1.get("phrase"))
    # If first asks zone alpha:
    if d6_1.get("type") == "clarification":
        # Clarify first
        r6_2 = httpx.post(f"{BASE_URL}/api/clarify", json={"session_id": "tkb_6", "clarification_id": d6_1.get("clarification_id"), "targets": [{"camera_id": "cam1"}]}, timeout=20)
        d6_2 = r6_2.json()
        print("Second call response after first clarify:", d6_2.get("type"), d6_2.get("phrase"))
        if d6_2.get("type") == "clarification":
            # Clarify second
            r6_3 = httpx.post(f"{BASE_URL}/api/clarify", json={"session_id": "tkb_6", "clarification_id": d6_2.get("clarification_id"), "targets": [{"camera_id": "cam2"}]}, timeout=20)
            d6_3 = r6_3.json()
            print("Third call response after second clarify:", d6_3.get("type"))
            t6_pass = (d6_3.get("type") in ["answer", "no_match"])
        else:
            t6_pass = False
    else:
        t6_pass = False
    results["T-KB-6 (two unknowns -> two sequential cards -> answer)"] = "PASS" if t6_pass else f"FAIL (call1={d6_1.get('type')})"
    print("Result:", results["T-KB-6 (two unknowns -> two sequential cards -> answer)"])

    print("--- 6. Delete referent and re-ask (T-KB-8) ---")
    # Find referent_id for secret bunker
    cur.execute("SELECT referent_id FROM referents WHERE phrase_norm LIKE '%secret bunker%'")
    row_ref = cur.fetchone()
    if row_ref:
        ref_id = row_ref[0]
        # Delete via API
        r_del = httpx.delete(f"{BASE_URL}/api/kb/{ref_id}", timeout=10)
        print("Delete API status:", r_del.status_code)
        # Ask again
        r_reask = httpx.post(f"{BASE_URL}/api/chat", json={"session_id": "tkb_reask", "message": "did a person enter the secret bunker?", "mode": "fast"}, timeout=20)
        d_reask = r_reask.json()
        t8_pass = (d_reask.get("type") == "clarification")
        results["T-KB-8 (delete allows new ask)"] = "PASS" if t8_pass else f"FAIL (type={d_reask.get('type')})"
    else:
        results["T-KB-8 (delete allows new ask)"] = "FAIL (could not find secret bunker referent)"
    print("Result:", results["T-KB-8 (delete allows new ask)"])

    print("--- 7. Clarification log check (T-KB-9) ---")
    cur.execute("SELECT phrase_norm, count(*) FROM clarification_log GROUP BY phrase_norm HAVING count(*) > 1")
    duplicates = cur.fetchall()
    results["T-KB-9 (clarification log each phrase at most once)"] = "PASS" if not duplicates else f"FAIL (duplicate asked phrases: {duplicates})"
    print("Result:", results["T-KB-9 (clarification log each phrase at most once)"])

    conn.close()
    return results

if __name__ == "__main__":
    res = test_clarify_once()
    print("\nSUMMARY OF CLARIFY-ONCE AUDIT:")
    for k, v in res.items():
        print(f"  {k}: {v}")
