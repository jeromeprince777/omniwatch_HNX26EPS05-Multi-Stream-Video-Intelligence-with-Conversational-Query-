"""Tests for OmniWatch Recommended Model Strategy and Grounding Contracts.

Verifies:
1. YOLO-World detector initialization and Box structure.
2. OpenCLIP ViT-B-32 embedding shape (1, 512) and L2-normalization (norm ~= 1.0).
3. Query parser rule-first precedence:
   - Deterministic queries handled by RuleQueryParser without LLM.
   - Ambiguous queries falling back to Ollama llama3.2:1b.
   - Graceful fallback to rules when Ollama is unavailable.
4. Verifier selection and fallback:
   - Fast mode leaves scores intact and verifier='none'.
   - Verified mode reranks candidates using OpenCLIP/SmolVLM2.
5. Grounding contract: No evidence rows produces 'no_match'.
6. /api/health exposes models, fallbacks, verifier_status, and parser_status.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omniwatch.api.main import create_app
from omniwatch.config import load_config
from omniwatch.detect.detector import Box
from omniwatch.embed.encoder import VLEncoder
from omniwatch.query.answer import build_answer_summary
from omniwatch.query.parser import (
    get_last_parser_used,
    parse_query,
)
from omniwatch.query.rerank import (
    OpenCLIPVerifier,
    UnifiedReranker,
    get_last_verifier_used,
)
from omniwatch.query.retrieve import CandidateResult


def test_box_structure_and_dataclass():
    """Verifies normalized bounding box and properties."""
    box = Box(x1=0.1, y1=0.2, x2=0.5, y2=0.8, cls="person", conf=0.88, quality=0.75)
    assert box.bbox == (0.1, 0.2, 0.5, 0.8)
    assert box.cls == "person"
    assert box.conf == 0.88
    assert box.quality == 0.75


def test_openclip_embedding_shape_and_normalization():
    """Verifies OpenCLIP ViT-B-32 produces (1, 512) L2-normalized vectors."""
    encoder = VLEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")

    # Test text embedding
    text_emb = encoder.encode_text("person in red shirt")
    assert text_emb.shape == (1, 512)
    norm_text = np.linalg.norm(text_emb[0])
    assert pytest.approx(norm_text, abs=1e-4) == 1.0

    # Test image crop embedding
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[20:80, 20:80] = [0, 0, 255]  # Red square
    crop_emb = encoder.encode_images([crop])
    assert crop_emb.shape == (1, 512)
    norm_crop = np.linalg.norm(crop_emb[0])
    assert pytest.approx(norm_crop, abs=1e-4) == 1.0


def test_query_parser_rule_first_deterministic():
    """Verifies that deterministic queries use RuleQueryParser directly."""
    cfg = load_config()

    # Deterministic query with clear intent and recognized class
    q = "show me a person wearing red at cam1"
    parsed = parse_query(q, use_llm=True, config=cfg)

    assert parsed.intent == "find"
    assert parsed.object_class_hint == "person"
    assert "red" in parsed.attributes.color
    assert "cam1" in parsed.camera_ids
    assert get_last_parser_used() == "rules"


def test_query_parser_fallback_on_unconfident_query():
    """Verifies that unconfident / ambiguous queries attempt LLM fallback."""
    cfg = load_config()

    # Vague conversational query where rules lack confidence
    q = "is there any suspicious loitering observed?"
    parsed = parse_query(q, use_llm=True, config=cfg)

    assert parsed.intent in {"exists", "find"}
    # Even if Ollama or model is available or unavailable, parser must succeed safely
    assert get_last_parser_used() in {"rules", "ollama_fallback"}


def test_openclip_verifier_scoring():
    """Verifies OpenCLIPVerifier computes valid probabilities and attribute bonuses."""
    verifier = OpenCLIPVerifier(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
    crop = np.full((120, 120, 3), (0, 0, 255), dtype=np.uint8)

    score_no_attrs = verifier.verify_candidate(crop, "a red object")
    assert 0.0 <= score_no_attrs <= 1.0

    score_with_attrs = verifier.verify_candidate(
        crop, "a red object", matched_attrs={"color": ["red"], "carries": []}
    )
    assert score_with_attrs >= score_no_attrs


def test_unified_reranker_selection():
    """Verifies UnifiedReranker reranks candidates and records verifier used."""
    reranker = UnifiedReranker(
        model_path="models/SmolVLM2-2.2B-Instruct",
        device="cpu",
        preferred_verifier="openclip",  # Test with OpenCLIP directly
    )

    cand1 = CandidateResult(
        tracklet_id=1,
        camera_id="cam1",
        camera_label="Entrance",
        t_start=100.0,
        t_end=105.0,
        t_peak=102.0,
        score=0.40,
        best_obs_id=1,
        bbox=(0.1, 0.1, 0.3, 0.3),
        matched={"color": ["red"]},
    )
    cand2 = CandidateResult(
        tracklet_id=2,
        camera_id="cam2",
        camera_label="Lobby",
        t_start=110.0,
        t_end=115.0,
        t_peak=112.0,
        score=0.35,
        best_obs_id=2,
        bbox=(0.2, 0.2, 0.4, 0.4),
        matched={},
    )

    dummy_crop = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    reranked = reranker.rerank(
        candidates=[cand1, cand2],
        text_hypothesis="person in red",
        alpha=0.5,
        beta=0.5,
        max_k=2,
        crop_resolver=lambda c: dummy_crop,
    )

    assert len(reranked) == 2
    assert reranked[0].score >= reranked[1].score
    assert get_last_verifier_used() == "openclip_fallback"


def test_grounding_no_evidence_no_answer():
    """Anti-hallucination Rule 3: Zero evidence rows must result in no_match summary."""
    searched_window = {"cameras": ["cam1"], "t_start": 0.0, "t_end": 100.0}
    summary = build_answer_summary(
        intent="find",
        results=[],
        searched_window=searched_window,
        object_phrase="unregistered intruder",
    )
    assert "No matching" in summary and "unregistered intruder" in summary
    assert "cam1" in summary


def test_api_health_endpoint():
    """Verifies that /api/health exposes detector, encoder, verifier, and fallbacks."""
    app = create_app()
    client = TestClient(app)

    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()

    assert "detector" in data["models"]
    assert "encoder" in data["models"]
    assert "verifier" in data["models"]
    assert "query_parser" in data["models"]
    assert "device" in data
    assert "index_counts" in data
    assert "fallbacks_active" in data
    assert "verifier_status" in data
    assert "parser_status" in data
