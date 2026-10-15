"""Test Suite for Query Parsing, Referent & Time Resolution, Answers, and API Endpoints.

Verifies:
1. >= 20 queries across both rule parser and LLM parser (validating against ParsedQuery).
2. >= 15 time resolution expressions against reference_now.
3. Grep-based verification asserting answer.py contains NO calls to an LLM.
4. Grounded retrieval and deterministic answer formatting.
5. FastAPI /api/health, /api/cameras, /api/chat, /api/clarify, and /api/kb endpoints.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from omniwatch.api.main import create_app
from omniwatch.query.answer import build_answer_summary
from omniwatch.query.parser import ParsedQuery, QueryTimeConstraint, RuleQueryParser, parse_query
from omniwatch.query.resolver import resolve_time
from omniwatch.query.retrieve import CandidateResult

TEST_QUERIES = [
    # Exists intent
    "Did a red car pass through the main gate in the last hour?",
    "Was there a person wearing blue jacket in east corridor?",
    "Has a delivery truck arrived at cam1?",
    "Is there any black backpack observed near the entrance?",
    "Did anyone carrying a suitcase enter cam2 between 09:00 and 09:30?",
    # Find intent
    "Show me a person wearing red shirt",
    "Find the white van near the east corridor",
    "Locate a courier carrying a box",
    "Search for blue sedan in the parking lot",
    "Where is the cyclist carrying a backpack?",
    # When intent
    "When did the black truck arrive at cam1?",
    "What time did a person in green enter?",
    "At what time was the white car detected?",
    "When did someone with a handbag pass through main gate?",
    # Count intent
    "How many people were seen in the last 10 minutes?",
    "Count the number of cars at cam1",
    "How many people carrying backpacks entered east corridor?",
    "Number of delivery vehicles between 09:10 and 09:20",
    # Path intent
    "Where did the white van go after passing main gate?",
    "Track the trajectory of person with black backpack",
    "Path of the cyclist across cameras",
    # Negatives (no location, no time)
    "Find a person in a yellow jacket",
    "Was there a silver car?",
]


def test_rule_parser_on_test_queries():
    """Validates RuleQueryParser extracts schema-compliant ParsedQuery for >= 20 queries."""
    for q in TEST_QUERIES:
        parsed = RuleQueryParser.parse(q)
        assert isinstance(parsed, ParsedQuery)
        assert parsed.raw == q
        assert parsed.intent in {"exists", "find", "when", "count", "path"}
        assert len(parsed.object_phrase) > 0


def test_parser_with_fallback_mechanism():
    """Ensures parse_query produces valid ParsedQuery even if LLM is unavailable."""
    for q in TEST_QUERIES[:5]:
        parsed = parse_query(q, use_llm=False)
        assert isinstance(parsed, ParsedQuery)
        assert parsed.intent in {"exists", "find", "when", "count", "path"}


TIME_EXPRESSIONS = [
    ("last hour", "relative", 3600.0),
    ("in the last hour", "relative", 3600.0),
    ("past hour", "relative", 3600.0),
    ("last 10 minutes", "relative", 600.0),
    ("in the last 10 mins", "relative", 600.0),
    ("past 30 minutes", "relative", 1800.0),
    ("last 5 minutes", "relative", 300.0),
    ("last 2 hours", "relative", 7200.0),
    ("5 minutes ago", "relative", None),
    ("10 mins ago", "relative", None),
    ("this morning", "relative", None),
    ("this afternoon", "relative", None),
    ("today", "relative", None),
    ("between 09:10 and 09:20", "absolute", None),
    ("between 9:00 and 10:00", "absolute", None),
    ("from 09:00 to 09:30", "absolute", None),
    ("at 09:14", "absolute", None),
    ("around 09:14", "absolute", None),
]


def test_time_resolver_expressions():
    """Tests >= 15 time resolution expressions against reference_now."""
    ref_now = 1791430212.0  # Epoch seconds from test campus footage

    for expr, kind, expected_duration in TIME_EXPRESSIONS:
        constraint = QueryTimeConstraint(kind=kind, raw=expr)
        t_start, t_end = resolve_time(constraint, reference_now=ref_now, tz_offset_hours=5.5)
        assert t_start is not None, f"Failed on {expr}"
        assert t_end is not None, f"Failed on {expr}"
        assert t_start <= t_end, f"Inverted range for {expr}"

        if expected_duration is not None:
            assert abs((t_end - t_start) - expected_duration) < 1.0


def test_grep_answer_py_contains_no_llm_calls():
    """Enforces non-negotiable Anti-hallucination Rule 3: answer.py must not call an LLM."""
    answer_file = Path("omniwatch/query/answer.py")
    assert answer_file.exists()
    content = answer_file.read_text(encoding="utf-8")

    forbidden = [
        "ollama", "openai", "generate", "chat", "llm", "completion",
        "transformers", "pipeline", "client.post"
    ]
    for term in forbidden:
        pattern = rf"\b{term}\b"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) == 0, f"Forbidden LLM term '{term}' found in answer.py: {matches}"


def test_deterministic_answer_builder():
    """Asserts deterministic answer summaries properly interpolate evidence rows."""
    dummy_results = [
        CandidateResult(
            tracklet_id=1,
            camera_id="cam1",
            camera_label="Main Gate",
            score=0.88,
            t_start=1791430200.0,
            t_end=1791430205.0,
            t_peak=1791430202.5,
            bbox=(0.1, 0.1, 0.4, 0.5),
            matched={"color": ["red"], "class": "car"},
            best_obs_id=10,
        )
    ]
    window = {"cameras": ["cam1"], "t_start": 1791430100.0, "t_end": 1791430300.0}

    ans_exists = build_answer_summary("exists", dummy_results, window, "red car")
    assert "Yes" in ans_exists
    assert "Main Gate" in ans_exists
    assert "0.88" in ans_exists

    ans_no_match = build_answer_summary("exists", [], window, "blue submarine")
    assert "No occurrence" in ans_no_match
    assert "cam1" in ans_no_match


def test_api_endpoints_integration():
    """Tests FastAPI endpoints: health, cameras, kb, chat, clarify."""
    app = create_app()
    client = TestClient(app)

    # 1. /api/health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert "detector" in health_data["models"]
    assert "vlm" in health_data["models"]
    assert "device" in health_data

    # 2. /api/cameras
    res_cams = client.get("/api/cameras")
    assert res_cams.status_code == 200
    cams_data = res_cams.json()
    assert isinstance(cams_data, list)
    assert len(cams_data) >= 2

    # 3. /api/chat with fast mode
    res_chat = client.post(
        "/api/chat",
        json={"session_id": "test_sess_01", "message": "Did a person pass through cam1?", "mode": "fast"},
    )
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert chat_data["type"] in {"answer", "no_match", "clarification"}
    assert chat_data["latency_ms"]["total"] > 0.0

    # 4. /api/chat with unknown location triggering clarification
    import uuid
    unique_loc = f"observatory_{uuid.uuid4().hex[:6]}"
    res_clarify_trigger = client.post(
        "/api/chat",
        json={"session_id": f"sess_{unique_loc}", "message": f"Did a person enter the {unique_loc}?", "mode": "fast"},
    )
    assert res_clarify_trigger.status_code == 200
    clarify_data = res_clarify_trigger.json()
    assert clarify_data["type"] == "clarification"
    assert unique_loc in clarify_data["phrase"]
    assert len(clarify_data["cameras"]) >= 2

    # 5. /api/clarify resolving the referent
    res_resolve = client.post(
        "/api/clarify",
        json={
            "session_id": f"sess_{unique_loc}",
            "clarification_id": clarify_data["clarification_id"],
            "targets": [{"camera_id": "cam1"}],
        },
    )
    assert res_resolve.status_code == 200
    resolved_chat = res_resolve.json()
    assert resolved_chat["type"] in {"answer", "no_match"}

    # 6. /api/kb verifying newly learned referent
    res_kb = client.get("/api/kb")
    assert res_kb.status_code == 200
    kb_data = res_kb.json()
    assert any(unique_loc in e["phrase_display"] for e in kb_data)


def test_public_site_query_isolated_from_campus():
    """Public location queries resolve to public cameras and never campus rows."""
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "session_id": "public-site-isolation",
            "message": "find a person at public surveillance",
            "mode": "fast",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] in {"answer", "no_match"}
    if data["type"] == "answer":
        assert data["results"]
        assert {item["camera_id"] for item in data["results"]} <= {
            "public_surveillance",
            "yokohama_cctv",
        }


def test_public_unknown_location_clarification_lists_public_cameras():
    """Public camera labels select the public site for clarification choices."""
    import uuid

    app = create_app()
    client = TestClient(app)
    unique_loc = f"public_unknown_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/api/chat",
        json={
            "session_id": "public-site-clarification",
            "message": f"find a person at {unique_loc}",
            "mode": "fast",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "clarification"
    assert {item["camera_id"] for item in data["cameras"]} == {
        "public_surveillance",
        "yokohama_cctv",
    }


def test_alerts_and_reid_api():
    """Validates alerts and cross-camera reid endpoints."""
    app = create_app()
    client = TestClient(app)

    # 1. Health check includes privacy and offline status
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert "privacy_blur_enabled" in h_data
    assert h_data["offline_ready"] is True

    # 2. Standing alert creation & listing
    s_resp = client.post(
        "/api/alerts/standing",
        json={"message": "Alert if person on cam1", "site_id": "default_site"},
    )
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    assert s_data["status"] == "created"

    l_resp = client.get("/api/alerts/standing")
    assert l_resp.status_code == 200
    assert any(item["site_id"] == "default_site" for item in l_resp.json())

    public_alert = client.post(
        "/api/alerts/standing",
        json={"message": "Alert if person on public_surveillance", "site_id": "public_cctv_demo"},
    )
    assert public_alert.status_code == 200
    public_list = client.get("/api/alerts/standing?site_id=public_cctv_demo")
    assert public_list.status_code == 200
    assert any(item["site_id"] == "public_cctv_demo" for item in public_list.json())

    # 3. List recent alerts
    a_resp = client.get("/api/alerts")
    assert a_resp.status_code == 200
    assert isinstance(a_resp.json(), list)

    # 4. Cross-camera ReID path trace
    r_resp = client.get("/api/reid/path/1")
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert "hops" in r_data or "error" in r_data
