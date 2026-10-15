"""Standing Queries and Rule-Based Event Triggering (Prompt 11B).

Complies with:
- docs/00_PRD (stretch), docs/06 (T-ALT), prompts/11_alerts_and_privacy_stretch.md.
- Parses standing query expressions into structured monitoring conditions.
- Evaluates newly indexed or streaming tracklets against active conditions.
- Writes deduplicated alerts to the SQLite `alerts` table.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any

from omniwatch.kb.normalize import normalize_phrase

logger = logging.getLogger("omniwatch.alerts")


def parse_standing_query(message: str) -> dict[str, Any]:
    """Parses a natural language standing alert command.

    Examples:
        - "Notify me if anyone enters main gate after 8 pm"
        - "Alert if a red car is seen on camera 1"
        - "Notify if person enters east corridor"
    """
    clean = message.strip()
    low = clean.lower()

    # Detect object/class
    cls_hint = "any"
    if "car" in low or "vehicle" in low:
        cls_hint = "car"
    elif "truck" in low:
        cls_hint = "truck"
    elif "motorcycle" in low or "bike" in low:
        cls_hint = "motorcycle"
    elif "person" in low or "anyone" in low or "someone" in low or "pedestrian" in low:
        cls_hint = "person"
    elif "bag" in low or "backpack" in low or "handbag" in low:
        cls_hint = "bag"

    # Detect location phrase
    loc_phrase = None
    loc_match = re.search(r"(?:at|in|enters|near|on)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:after|before|between|from|with)|\?|$)", clean, re.IGNORECASE)
    if loc_match:
        cand_loc = loc_match.group(1).strip()
        loc_phrase = cand_loc

    # Detect time-of-day condition (e.g. "after 8 pm", "after 20:00")
    after_seconds_of_day = None
    time_match = re.search(r"after\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", low)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        meridiem = time_match.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        after_seconds_of_day = hour * 3600 + minute * 60

    return {
        "raw_message": clean,
        "object_class": cls_hint,
        "location_phrase": loc_phrase,
        "after_seconds_of_day": after_seconds_of_day,
    }


def register_standing_query(site_id: str, message: str, conn: sqlite3.Connection) -> int:
    """Registers an active standing query in SQLite."""
    parsed = parse_standing_query(message)
    now = time.time()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO standing_queries (site_id, message, parsed_json, active, created_at)
        VALUES (?, ?, ?, 1, ?)
        """,
        (site_id, message, json.dumps(parsed), now),
    )
    conn.commit()
    return cur.lastrowid


def list_standing_queries(site_id: str, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Returns all standing queries for a given site."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, site_id, message, parsed_json, active, created_at FROM standing_queries WHERE site_id = ? ORDER BY id DESC",
        (site_id,),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0],
            "site_id": row[1],
            "message": row[2],
            "parsed": json.loads(row[3]) if row[3] else {},
            "active": bool(row[4]),
            "created_at": row[5],
        })
    return results


def evaluate_standing_alerts(
    tracklet: dict[str, Any] | sqlite3.Row,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Evaluates an indexed tracklet against active standing queries.

    If a condition matches and has not already fired for this tracklet,
    records an alert in the `alerts` table.
    """
    if isinstance(tracklet, sqlite3.Row):
        trk_dict = dict(tracklet)
    else:
        trk_dict = tracklet

    trk_id = trk_dict.get("tracklet_id")
    cam_id = trk_dict.get("camera_id")
    cls_name = trk_dict.get("cls", "").lower()
    t_start = float(trk_dict.get("t_start", 0.0))

    cur = conn.cursor()
    cur.execute("SELECT id, site_id, message, parsed_json FROM standing_queries WHERE active = 1")
    standing_rows = cur.fetchall()

    def _intersect_ratio(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
        # b = (x1, y1, x2, y2)
        ix_min = max(b1[0], b2[0])
        iy_min = max(b1[1], b2[1])
        ix_max = min(b1[2], b2[2])
        iy_max = min(b1[3], b2[3])
        iw = max(0.0, ix_max - ix_min)
        ih = max(0.0, iy_max - iy_min)
        inter_area = iw * ih
        if inter_area == 0:
            return 0.0
        # Calculate intersection over the area of b1 (the object)
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        if b1_area == 0:
            return 0.0
        return inter_area / b1_area

    fired_alerts: list[dict[str, Any]] = []

    for s_id, site_id, msg, parsed_json in standing_rows:
        rule = json.loads(parsed_json) if parsed_json else {}

        # 1. Class match check
        req_cls = rule.get("object_class", "any").lower()
        if req_cls != "any":
            if req_cls == "person" and cls_name not in ("person", "pedestrian"):
                continue
            elif req_cls == "car" and cls_name not in ("car", "truck", "vehicle", "bus"):
                continue
            elif req_cls not in cls_name:
                continue

        # 2. Location & Spatial ROI match check
        loc_phrase = rule.get("location_phrase")
        if loc_phrase:
            loc_norm = normalize_phrase(loc_phrase)
            cam_norm = normalize_phrase(cam_id or "")
            is_loc_match = False
            
            # Direct camera match
            if loc_norm in cam_norm or cam_norm in loc_norm:
                is_loc_match = True
            
            # KB Map with optional ROI checking
            if not is_loc_match:
                cur.execute(
                    "SELECT targets_json FROM referents WHERE site_id = ? AND phrase_norm = ?",
                    (site_id, loc_norm),
                )
                ref_row = cur.fetchone()
                if ref_row and ref_row[0]:
                    targets = json.loads(ref_row[0])
                    for t in targets:
                        if t.get("camera_id") == cam_id:
                            # Spatial Geofence Check
                            roi = t.get("roi")
                            if roi:
                                # Fetch the actual tracklet bounding box to check intersection
                                cur.execute("SELECT bbox_json FROM observations WHERE tracklet_id = ? ORDER BY conf DESC LIMIT 1", (trk_id,))
                                obs_row = cur.fetchone()
                                if obs_row and obs_row[0]:
                                    obj_bbox = json.loads(obs_row[0])
                                    # roi is [x1, y1, x2, y2]
                                    if _intersect_ratio(tuple(obj_bbox), tuple(roi)) > 0.15:
                                        is_loc_match = True
                                        break
                                    else:
                                        # Box did not intersect drawn ROI
                                        continue
                            else:
                                is_loc_match = True
                                break

            if not is_loc_match:
                continue

        # 3. Time of day check (in UTC/local time of t_start)
        after_sec = rule.get("after_seconds_of_day")
        if after_sec is not None:
            # Seconds of day = t_start % 86400
            day_sec = int(t_start) % 86400
            if day_sec < after_sec:
                continue

        # 4. Deduplication: Check if alert already fired for this (standing_id, tracklet_id)
        cur.execute(
            "SELECT id FROM alerts WHERE standing_id = ? AND tracklet_id = ?",
            (s_id, trk_id),
        )
        if cur.fetchone():
            continue

        # Fire alert!
        now = time.time()
        payload = {
            "tracklet_id": trk_id,
            "camera_id": cam_id,
            "cls": cls_name,
            "t_start": t_start,
            "message": f"ALERT: {cls_name} detected at {cam_id} matching rule '{msg}'",
        }
        cur.execute(
            """
            INSERT INTO alerts (standing_id, tracklet_id, fired_at, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (s_id, trk_id, now, json.dumps(payload)),
        )
        alert_id = cur.lastrowid
        fired_alerts.append({
            "alert_id": alert_id,
            "standing_id": s_id,
            "tracklet_id": trk_id,
            "payload": payload,
            "fired_at": now,
        })

    conn.commit()
    return fired_alerts


def get_recent_alerts(
    limit: int = 50,
    site_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Returns recent alerts from the database."""
    if conn is None:
        from omniwatch.config import load_config
        from omniwatch.db import get_connection
        cfg = load_config()
        conn = get_connection(cfg.storage.db_path)

    cur = conn.cursor()
    query = """
        SELECT a.id, a.standing_id, a.tracklet_id, a.fired_at, a.payload_json, s.message
        FROM alerts a
        LEFT JOIN standing_queries s ON a.standing_id = s.id
    """
    params: list[Any] = []
    if site_id is not None:
        query += " WHERE s.site_id = ?"
        params.append(site_id)
    query += " ORDER BY a.fired_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = cur.fetchall()
    alerts = []
    for r in rows:
        payload = json.loads(r[4]) if r[4] else {}
        alerts.append({
            "alert_id": r[0],
            "standing_id": r[1],
            "tracklet_id": r[2],
            "fired_at": r[3],
            "payload": payload,
            "standing_message": r[5] or "Custom Standing Alert",
        })
    return alerts
