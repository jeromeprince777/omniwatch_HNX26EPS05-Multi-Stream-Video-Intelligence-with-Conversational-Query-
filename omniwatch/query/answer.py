"""Deterministic Answer Generation Module.

Responsibility:
- Enforces Non-negotiable Anti-hallucination Rule 3 (AGENTS.md):
  Answers are strictly built by interpolating database evidence rows (camera, timestamp, bbox, clip).
- If no candidate passes the calibrated threshold, returns grounded "no match" with the searched window.
- Generates human-readable, deterministic answer summaries without free-form model hallucination.
- Complies strictly with docs/00_PRD.md FR-QRY-5.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from omniwatch.query.retrieve import CandidateResult


def format_timestamp(epoch_ts: float | None, tz_offset_hours: float = 5.5) -> str:
    """Formats epoch timestamp into readable ISO format with manifest local offset."""
    if epoch_ts is None:
        return "N/A"
    tz = timezone(timedelta(hours=tz_offset_hours))
    dt = datetime.fromtimestamp(epoch_ts, tz=tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_answer_summary(
    intent: str,
    results: list[CandidateResult],
    searched_window: dict[str, Any],
    object_phrase: str = "target",
    tz_offset_hours: float = 5.5,
) -> str:
    """Builds templated deterministic response string grounded strictly in result rows.

    This module contains zero external model calls, guaranteeing grounded anti-hallucination.
    """
    cams = searched_window.get("cameras", [])
    cams_str = ", ".join(cams) if cams else "all cameras"

    t_start = searched_window.get("t_start")
    t_end = searched_window.get("t_end")
    if t_start is not None and t_end is not None:
        window_str = f"{format_timestamp(t_start, tz_offset_hours)} to {format_timestamp(t_end, tz_offset_hours)}"
    elif t_start is not None:
        window_str = f"after {format_timestamp(t_start, tz_offset_hours)}"
    elif t_end is not None:
        window_str = f"before {format_timestamp(t_end, tz_offset_hours)}"
    else:
        window_str = "entire recorded footage timeline"

    # Negative No-Match Responses
    if not results:
        if intent == "exists":
            return (
                f"No occurrence of '{object_phrase}' was observed across searched cameras "
                f"[{cams_str}] during {window_str}."
            )
        elif intent == "find":
            return (
                f"No matching observations found for '{object_phrase}' across searched cameras "
                f"[{cams_str}] during {window_str}."
            )
        elif intent == "when":
            return (
                f"Could not determine when '{object_phrase}' appeared; no detections found in "
                f"[{cams_str}] during {window_str}."
            )
        elif intent == "count":
            return (
                f"0 occurrences of '{object_phrase}' detected across searched cameras "
                f"[{cams_str}] during {window_str}."
            )
        elif intent == "path":
            return (
                f"No movement trajectory found for '{object_phrase}' across cameras "
                f"[{cams_str}] during {window_str}."
            )
        return f"No match found for '{object_phrase}' in the searched window."

    # Grounded Positive Responses
    top = results[0]
    top_cam = top.camera_label or top.camera_id
    t_start_str = format_timestamp(top.t_start, tz_offset_hours)
    t_end_str = format_timestamp(top.t_end, tz_offset_hours)
    t_peak_str = format_timestamp(top.t_peak, tz_offset_hours)

    if intent == "exists":
        return (
            f"Yes, '{object_phrase}' was observed at {top_cam} ({top.camera_id}) "
            f"between {t_start_str} and {t_end_str} (match score: {top.score:.2f})."
        )
    elif intent == "find":
        return (
            f"Found {len(results)} observation(s) matching '{object_phrase}'. "
            f"Most prominent match is at {top_cam} ({top.camera_id}) from {t_start_str} to {t_end_str} "
            f"(score: {top.score:.2f})."
        )
    elif intent == "when":
        return (
            f"'{object_phrase}' was detected at {t_peak_str} "
            f"(active interval: {t_start_str} to {t_end_str}) on {top_cam} ({top.camera_id})."
        )
    elif intent == "count":
        return (
            f"Counted {len(results)} unique occurrence(s) of '{object_phrase}' "
            f"across cameras [{cams_str}] during {window_str}."
        )
    elif intent == "path":
        chronological = sorted(results, key=lambda c: c.t_start)
        hops = [
            f"{c.camera_label or c.camera_id} at {format_timestamp(c.t_start, tz_offset_hours)}"
            for c in chronological
        ]
        return f"Trajectory for '{object_phrase}': " + " -> ".join(hops)

    return (
        f"Detected '{object_phrase}' at {top_cam} ({top.camera_id}) "
        f"from {t_start_str} to {t_end_str}."
    )
