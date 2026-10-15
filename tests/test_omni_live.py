"""Tests for omni_live.py — new 4-thread architecture."""
import pytest
from omni_live import parse_live, build_app, set_query, status, Stream


# ─── query parsing ─────────────────────────────────────────────────────────────
def test_parse_live_basic_bag():
    p = parse_live("person carrying a bag")
    assert "person" in p["cls"]
    assert p["bag"] is True
    assert p["color"] is None
    assert len(p["resid"]) == 0


def test_parse_live_color_vehicle():
    p = parse_live("red car")
    assert "car" in p["cls"]
    assert p["color"] == "red"
    assert p["bag"] is False


def test_parse_live_free_text_hat():
    p = parse_live("man with a hat")
    assert "person" in p["cls"]
    assert "hat" in p["resid"]


def test_parse_live_color_accessory_delegates_to_clip():
    """'red hat' → colour should NOT go to p['color'] (it modifies the hat, not the body)."""
    p = parse_live("person with a red hat")
    assert "person" in p["cls"]
    assert p["color"] is None          # body colour unconstrained
    assert "red" in p["resid"]         # delegated to CLIP
    assert "hat" in p["resid"]


def test_parse_live_everyone():
    p = parse_live("everyone")
    # "everyone" is a stop word — no class constraint, no filter → matches all
    assert p["text"] == "everyone"


# ─── track evaluation ──────────────────────────────────────────────────────────
def _stream() -> Stream:
    """Instantiate without starting threads."""
    return Stream("test_cam", "0", False)


def test_evaluate_matching():
    s = _stream()
    P = parse_live("person carrying a bag")
    tr = {"cls": "person", "color": "blue", "has_bag": True, "clip_p": None}
    assert s.evaluate(tr, P) is True


def test_evaluate_wrong_class():
    s = _stream()
    P = parse_live("person carrying a bag")
    tr = {"cls": "car", "color": "unknown", "has_bag": False, "clip_p": None}
    assert s.evaluate(tr, P) is False


def test_evaluate_no_bag():
    s = _stream()
    P = parse_live("person carrying a bag")
    tr = {"cls": "person", "color": "unknown", "has_bag": False, "clip_p": None}
    assert s.evaluate(tr, P) is False


def test_evaluate_clip_pending():
    s = _stream()
    P = parse_live("man with a hat")
    tr = {"cls": "person", "color": "unknown", "has_bag": False, "clip_p": None}
    assert s.evaluate(tr, P) is None


def test_evaluate_clip_above_threshold():
    s = _stream()
    P = parse_live("man with a hat")
    tr = {"cls": "person", "color": "unknown", "has_bag": False, "clip_p": 0.65}
    assert s.evaluate(tr, P) is True


def test_evaluate_clip_below_threshold():
    s = _stream()
    P = parse_live("man with a hat")
    tr = {"cls": "person", "color": "unknown", "has_bag": False, "clip_p": 0.25}
    assert s.evaluate(tr, P) is False


# ─── FastAPI endpoints ─────────────────────────────────────────────────────────
def test_api_status_and_query():
    from fastapi.testclient import TestClient
    app = build_app()
    client = TestClient(app)

    set_query("white car", others=True)
    r = client.get("/api/status")
    assert r.status_code == 200
    d = r.json()
    assert d["query"] == "white car"
    assert "car" in d["parsed"]["class"]
    assert d["parsed"]["color"] == "white"

    r2 = client.post("/api/query",
                     json={"q": "person carrying a bag", "others": False})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["query"] == "person carrying a bag"
    assert d2["others"] is False


def test_index_page():
    from fastapi.testclient import TestClient
    r = TestClient(build_app()).get("/")
    assert r.status_code == 200
    assert "OMNIWATCH" in r.text.upper()
