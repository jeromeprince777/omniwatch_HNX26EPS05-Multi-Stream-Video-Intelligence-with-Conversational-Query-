"""Restart-persistence coverage for the Clarify-Once knowledge base."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from omniwatch.api.main import create_app


def test_clarify_once_persists_across_fresh_app_instance():
    phrase = f"public restart zone {uuid.uuid4().hex[:8]}"
    session_id = f"kb-restart-{uuid.uuid4().hex[:8]}"

    first_client = TestClient(create_app())
    first = first_client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": f"find a person at {phrase}",
            "mode": "fast",
        },
    )
    assert first.status_code == 200
    clarification = first.json()
    assert clarification["type"] == "clarification"
    assert {camera["camera_id"] for camera in clarification["cameras"]} == {
        "public_surveillance",
        "yokohama_cctv",
    }

    resolved = first_client.post(
        "/api/clarify",
        json={
            "session_id": session_id,
            "clarification_id": clarification["clarification_id"],
            "targets": [{"camera_id": "public_surveillance"}],
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["type"] == "answer"
    first_client.close()

    second_client = TestClient(create_app())
    after_restart = second_client.post(
        "/api/chat",
        json={
            "session_id": f"{session_id}-second",
            "message": f"find a person at The {phrase}",
            "mode": "fast",
        },
    )
    assert after_restart.status_code == 200
    resumed = after_restart.json()
    assert resumed["type"] == "answer"
    assert resumed["results"]
    assert {item["camera_id"] for item in resumed["results"]} == {"public_surveillance"}
    second_client.close()
