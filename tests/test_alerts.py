"""Tests for standing queries and alerts (T-ALT)."""

import sqlite3
import pytest

from omniwatch.alerts.alerts import (
    evaluate_standing_alerts,
    parse_standing_query,
    register_standing_query,
)
from omniwatch.db import init_db


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "alerts_test.db"
    init_db(db_file)
    conn = sqlite3.connect(str(db_file))
    yield conn
    conn.close()


def test_parse_standing_query():
    parsed = parse_standing_query("Notify me if anyone enters main gate after 8 pm")
    assert parsed["object_class"] == "person"
    assert "main gate" in parsed["location_phrase"].lower()
    assert parsed["after_seconds_of_day"] == 20 * 3600

    parsed2 = parse_standing_query("Alert if a car is seen on cam1")
    assert parsed2["object_class"] == "car"
    assert "cam1" in parsed2["location_phrase"].lower()


def test_register_and_evaluate_standing_alerts(test_db):
    s_id = register_standing_query("site_demo", "Alert if car on cam1", test_db)
    assert s_id > 0

    # Matching tracklet
    trk = {
        "tracklet_id": 101,
        "camera_id": "cam1",
        "cls": "car",
        "t_start": 1000.0,
    }
    fired = evaluate_standing_alerts(trk, test_db)
    assert len(fired) == 1
    assert fired[0]["standing_id"] == s_id
    assert fired[0]["tracklet_id"] == 101

    # Second evaluation of same tracklet must dedupe (0 new alerts)
    fired_dup = evaluate_standing_alerts(trk, test_db)
    assert len(fired_dup) == 0

    # Non-matching tracklet (different class)
    trk_person = {
        "tracklet_id": 102,
        "camera_id": "cam1",
        "cls": "person",
        "t_start": 1050.0,
    }
    fired_none = evaluate_standing_alerts(trk_person, test_db)
    assert len(fired_none) == 0
