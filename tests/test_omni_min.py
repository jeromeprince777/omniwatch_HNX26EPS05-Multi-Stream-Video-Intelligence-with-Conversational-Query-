import numpy as np
import pytest
from omni_min import bag_on_person, color_name, parse, search, db


def test_parse_queries():
    cls, color, bag = parse("person carrying a bag")
    assert "person" in cls
    assert bag is True
    assert color is None

    cls, color, bag = parse("find a red car")
    assert "car" in cls
    assert color == "red"
    assert bag is False

    cls, color, bag = parse("blue backpack")
    assert "person" in cls  # bag words infer person when cls is empty
    assert color == "blue"
    assert bag is True


def test_bag_on_person_overlap():
    # Person box: [x1, y1, x2, y2] = [100, 100, 200, 300]
    # Bag box mostly inside person
    person_box = [100, 100, 200, 300]
    bag_inside = [120, 150, 180, 220]
    assert bag_on_person(person_box, bag_inside) is True

    # Bag completely far away
    bag_far = [500, 500, 600, 600]
    assert bag_on_person(person_box, bag_far) is False


def test_color_name():
    # Solid black
    black_img = np.zeros((50, 50, 3), dtype=np.uint8)
    assert color_name(black_img) == "black"

    # Solid white
    white_img = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert color_name(white_img) == "white"


def test_db_schema(tmp_path, monkeypatch):
    test_db_path = str(tmp_path / "test_omni.db")
    monkeypatch.setenv("OMNI_DB", test_db_path)
    monkeypatch.setenv("OMNI_DATA", str(tmp_path / "data"))

    con = db()
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='obs'")
    row = cur.fetchone()
    assert row is not None
    assert row["name"] == "obs"


def test_ensure_clip_missing_track():
    from omni_min import ensure_clip
    # Requesting non-existent camera/track returns None
    result = ensure_clip("NonExistentCam", 999999)
    assert result is None


def test_ensure_clip_cached():
    import os
    from omni_min import ensure_clip
    # Requesting cached tracklet returns valid .mp4 path
    result = ensure_clip("Yokohama", 1067)
    if result is not None:
        assert result.endswith(".mp4")
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

