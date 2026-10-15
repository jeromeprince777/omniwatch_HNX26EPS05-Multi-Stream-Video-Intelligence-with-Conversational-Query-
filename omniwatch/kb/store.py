"""Persistent Knowledge Base CRUD and Clarification Log.

Responsibility:
- Stores location referent mappings (phrase_norm, site_id, targets_json).
- Maintains referent_history for edits and auditability.
- Logs clarification requests and resolutions to clarification_log.
- Ensures all writes are durably committed (fsync / synchronous=FULL) before HTTP responses.

Exposed Interfaces:
- def get_referent(site_id: str, phrase_norm: str, conn) -> dict | None
- def save_referent(site_id: str, phrase_norm: str, phrase_display: str, targets: list, conn, source="user") -> int
- def update_referent(referent_id: int, new_targets: list, conn, actor="user") -> None
- def delete_referent(referent_id: int, conn) -> None
- def log_clarification_asked(site_id: str, phrase_norm: str, session_id: str, conn) -> int
- def log_clarification_resolved(clarification_id: int, referent_id: int, conn) -> None
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any


def get_referent(site_id: str, phrase_norm: str, conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Fetch stored referent by site_id and normalized phrase."""
    row = conn.execute(
        "SELECT * FROM referents WHERE site_id = ? AND phrase_norm = ? AND kind = 'location'",
        (site_id, phrase_norm),
    ).fetchone()
    if not row:
        return None
    res = dict(row)
    res["targets"] = json.loads(res["targets_json"])
    return res


def save_referent(
    site_id: str,
    phrase_norm: str,
    phrase_display: str,
    targets: list[dict[str, Any]],
    conn: sqlite3.Connection,
    source: str = "user",
) -> int:
    """Durably writes a new referent and returns its referent_id."""
    now = time.time()
    targets_json = json.dumps(targets)
    cursor = conn.execute(
        """
        INSERT INTO referents (site_id, phrase_norm, phrase_display, kind, targets_json, source, created_at, updated_at)
        VALUES (?, ?, ?, 'location', ?, ?, ?, ?)
        """,
        (site_id, phrase_norm, phrase_display, targets_json, source, now, now),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def update_referent(
    referent_id: int,
    new_targets: list[dict[str, Any]],
    conn: sqlite3.Connection,
    actor: str = "user",
) -> None:
    """Updates referent targets and appends to referent_history."""
    raise NotImplementedError("update_referent will be implemented in Stage 06")


def delete_referent(referent_id: int, conn: sqlite3.Connection) -> None:
    """Deletes referent from KB to allow future re-clarification."""
    raise NotImplementedError("delete_referent will be implemented in Stage 06")


def log_clarification_asked(
    site_id: str,
    phrase_norm: str,
    session_id: str,
    conn: sqlite3.Connection,
) -> int:
    """Records that a clarification card was prompted to the user."""
    raise NotImplementedError("log_clarification_asked will be implemented in Stage 06")


def log_clarification_resolved(
    clarification_id: int,
    referent_id: int,
    conn: sqlite3.Connection,
) -> None:
    """Marks clarification as resolved."""
    raise NotImplementedError("log_clarification_resolved will be implemented in Stage 06")
