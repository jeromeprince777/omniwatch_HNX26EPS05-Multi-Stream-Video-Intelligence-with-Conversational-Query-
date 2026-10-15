"""Time and Location Referent Resolver Module.

Responsibility:
- Resolves relative time expressions ("in the last hour") against reference_now (end of latest footage).
- Resolves location phrases against knowledge base (referents table) and camera manifest labels.
- Detects unresolved location referents and triggers clarification flow.
- Complies with docs/01_ARCHITECTURE.md sections 4 & 9.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

from omniwatch.kb.normalize import normalize_phrase
from omniwatch.query.parser import QueryTimeConstraint


@dataclass
class ResolvedTarget:
    camera_id: str
    roi: tuple[float, float, float, float] | None = None


@dataclass
class ReferentResolution:
    resolved_targets: list[ResolvedTarget]
    unresolved_phrases: list[str]


def parse_word_number(word: str) -> int:
    """Parses English number words up to 60 or digits."""
    word = word.strip().lower()
    word_map = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    }
    if word in word_map:
        return word_map[word]
    try:
        return int(word)
    except ValueError:
        return 1


def resolve_time(
    time_constraint: QueryTimeConstraint,
    reference_now: float,
    tz_offset_hours: float = 5.5,  # Default to manifest timezone offset (+05:30)
) -> tuple[float | None, float | None]:
    """Resolves relative or absolute time constraint into [t_start, t_end] epoch UTC seconds.

    Args:
        time_constraint: Parsed time expression constraint.
        reference_now: Reference timestamp in UTC epoch seconds (default = max camera end).
        tz_offset_hours: Timezone offset in hours from UTC for local time interpretations.

    Returns:
        (t_start, t_end) in UTC epoch seconds, or (None, None) if unrestricted.
    """
    if time_constraint.kind == "none" or not time_constraint.raw:
        return None, None

    raw = time_constraint.raw.strip().lower()
    tz = timezone(timedelta(hours=tz_offset_hours))
    ref_dt = datetime.fromtimestamp(reference_now, tz=tz)

    # 1. Relative Expressions
    # "last/past hour"
    if re.search(r"\b(?:last|past)\s+hour\b", raw):
        return reference_now - 3600.0, reference_now

    # "last/past N minutes/hours"
    m_rel = re.search(r"\b(?:last|past|in the last|in the past)\s+([a-z0-9]+)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", raw)
    if m_rel:
        val = parse_word_number(m_rel.group(1))
        unit = m_rel.group(2)
        if "hour" in unit or "hr" in unit:
            duration = val * 3600.0
        elif "min" in unit:
            duration = val * 60.0
        else:
            duration = float(val)
        return reference_now - duration, reference_now

    # "N minutes/hours ago"
    m_ago = re.search(r"\b([a-z0-9]+)\s*(hours?|hrs?|minutes?|mins?)\s+ago\b", raw)
    if m_ago:
        val = parse_word_number(m_ago.group(1))
        unit = m_ago.group(2)
        offset = val * 3600.0 if "hour" in unit or "hr" in unit else val * 60.0
        target = reference_now - offset
        return target - 60.0, target + 60.0

    # "this morning" (06:00 to 12:00 local time)
    if "this morning" in raw:
        start_dt = ref_dt.replace(hour=6, minute=0, second=0, microsecond=0)
        end_dt = ref_dt.replace(hour=12, minute=0, second=0, microsecond=0)
        return start_dt.astimezone(UTC).timestamp(), end_dt.astimezone(UTC).timestamp()

    # "this afternoon" (12:00 to 18:00 local time)
    if "this afternoon" in raw:
        start_dt = ref_dt.replace(hour=12, minute=0, second=0, microsecond=0)
        end_dt = ref_dt.replace(hour=18, minute=0, second=0, microsecond=0)
        return start_dt.astimezone(UTC).timestamp(), end_dt.astimezone(UTC).timestamp()

    # "tomorrow"
    if "tomorrow" in raw:
        tomorrow_dt = ref_dt + timedelta(days=1)
        if "noon" in raw:
            dt = tomorrow_dt.replace(hour=12, minute=0, second=0, microsecond=0)
            ts = dt.astimezone(UTC).timestamp()
            return ts - 1800.0, ts + 1800.0
        start_dt = tomorrow_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = tomorrow_dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return start_dt.astimezone(UTC).timestamp(), end_dt.astimezone(UTC).timestamp()

    # "yesterday"
    if "yesterday" in raw:
        yesterday_dt = ref_dt - timedelta(days=1)
        start_dt = yesterday_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = yesterday_dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return start_dt.astimezone(UTC).timestamp(), end_dt.astimezone(UTC).timestamp()

    # "today" (00:00 to 23:59:59 local time)
    if "today" in raw:
        start_dt = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = ref_dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return start_dt.astimezone(UTC).timestamp(), end_dt.astimezone(UTC).timestamp()

    # "at noon" / "noon"
    if re.search(r"\b(?:at\s+)?noon\b", raw):
        dt = ref_dt.replace(hour=12, minute=0, second=0, microsecond=0)
        ts = dt.astimezone(UTC).timestamp()
        return ts - 1800.0, ts + 1800.0

    # 2. Absolute Expressions
    # Full ISO format check
    try:
        if "t" in raw:
            dt = datetime.fromisoformat(time_constraint.raw.strip())
            ts = dt.timestamp()
            return ts - 30.0, ts + 30.0
    except ValueError:
        pass

    # "between HH:MM and HH:MM" or "from HH:MM to HH:MM"
    m_between = re.search(
        r"(?:between|from)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:and|to)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?",
        raw
    )
    if m_between:
        h1, m1, s1 = int(m_between.group(1)), int(m_between.group(2)), int(m_between.group(3) or 0)
        h2, m2, s2 = int(m_between.group(4)), int(m_between.group(5)), int(m_between.group(6) or 0)
        dt1 = ref_dt.replace(hour=h1, minute=m1, second=s1, microsecond=0)
        dt2 = ref_dt.replace(hour=h2, minute=m2, second=s2, microsecond=0)
        return dt1.astimezone(UTC).timestamp(), dt2.astimezone(UTC).timestamp()

    # "at HH:MM" / "around HH:MM"
    m_at = re.search(r"(?:at|around)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", raw)
    if m_at:
        h, m, s = int(m_at.group(1)), int(m_at.group(2)), int(m_at.group(3) or 0)
        dt = ref_dt.replace(hour=h, minute=m, second=s, microsecond=0)
        ts = dt.astimezone(UTC).timestamp()
        return ts - 30.0, ts + 30.0

    return None, None


def resolve_referents(
    location_phrases: list[str],
    camera_ids: list[str],
    site_id: str,
    conn: sqlite3.Connection,
) -> ReferentResolution:
    """Resolves location phrases into camera targets or identifies unknown referents."""
    resolved: list[ResolvedTarget] = []
    unresolved: list[str] = []
    seen_cams: set[str] = set()

    # 1. Resolve explicit camera IDs
    for cam_id in camera_ids:
        cur = conn.cursor()
        cur.execute("SELECT camera_id FROM cameras WHERE site_id = ? AND camera_id = ?", (site_id, cam_id))
        row = cur.fetchone()
        if row and cam_id not in seen_cams:
            resolved.append(ResolvedTarget(camera_id=cam_id))
            seen_cams.add(cam_id)

    # Fetch camera labels for name matching
    cur = conn.cursor()
    cur.execute("SELECT camera_id, label FROM cameras WHERE site_id = ?", (site_id,))
    camera_labels = cur.fetchall()

    for phrase in location_phrases:
        p_norm = normalize_phrase(phrase)
        if not p_norm:
            continue

        matched = False

        # 2. Check referents table in Knowledge Base (scoped to site_id or global)
        cur.execute(
            "SELECT targets_json FROM referents WHERE (site_id = ? OR site_id IS NULL) AND phrase_norm = ? ORDER BY (site_id = ?) DESC LIMIT 1",
            (site_id, p_norm, site_id),
        )
        row = cur.fetchone()
        if row:
            try:
                targets_data = json.loads(row[0])
                referent_resolved = False
                for t in targets_data:
                    c_id = t.get("camera_id")
                    roi = tuple(t.get("roi")) if t.get("roi") else None
                    cur.execute(
                        "SELECT 1 FROM cameras WHERE site_id = ? AND camera_id = ?",
                        (site_id, c_id),
                    )
                    if c_id and cur.fetchone():
                        referent_resolved = True
                        if c_id not in seen_cams:
                            resolved.append(ResolvedTarget(camera_id=c_id, roi=roi))
                            seen_cams.add(c_id)
                matched = referent_resolved
            except Exception:
                matched = False

        # 3. Check camera IDs and manifest labels
        if not matched:
            for c_id, label in camera_labels:
                id_norm = normalize_phrase(c_id.replace("_", " "))
                label_norm = normalize_phrase(label or "")
                if p_norm == id_norm or (
                    label_norm and (p_norm in label_norm or label_norm in p_norm)
                ):
                    if c_id not in seen_cams:
                        resolved.append(ResolvedTarget(camera_id=c_id))
                        seen_cams.add(c_id)
                    matched = True
                    break

        if not matched:
            unresolved.append(phrase)

    return ReferentResolution(resolved_targets=resolved, unresolved_phrases=unresolved)
