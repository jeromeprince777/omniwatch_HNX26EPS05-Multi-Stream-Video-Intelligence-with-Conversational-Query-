"""Tests for OmniWatch SQLite connection helper and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from omniwatch.db import get_connection, init_db

EXPECTED_TABLES = {
    "sites",
    "cameras",
    "observations",
    "tracklets",
    "referents",
    "referent_history",
    "clarification_log",
    "pending_queries",
    "query_log",
    "standing_queries",
    "alerts",
}


def test_init_db_creates_all_tables(tmp_path: Path):
    db_file = tmp_path / "test.db"
    init_db(db_file)

    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row["name"] for row in cursor.fetchall()}
    conn.close()

    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing expected tables: {missing}"


def test_pragmas_enforced(tmp_path: Path):
    db_file = tmp_path / "pragma_test.db"
    conn = get_connection(db_file)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.upper() == "WAL"

    cursor.execute("PRAGMA synchronous;")
    synchronous = cursor.fetchone()[0]
    # In SQLite, synchronous=FULL corresponds to numeric value 2
    assert synchronous == 2

    cursor.execute("PRAGMA foreign_keys;")
    foreign_keys = cursor.fetchone()[0]
    assert foreign_keys == 1

    conn.close()


def test_referents_unique_constraint(tmp_path: Path):
    db_file = tmp_path / "referents_test.db"
    init_db(db_file)
    conn = get_connection(db_file)

    conn.execute(
        """
        INSERT INTO referents (site_id, phrase_norm, phrase_display, targets_json, source, created_at, updated_at)
        VALUES ('site1', 'main gate', 'Main Gate', '[{"camera_id":"cam1"}]', 'user', 1000.0, 1000.0)
        """
    )
    conn.commit()

    # Inserting duplicate (site_id, phrase_norm, kind) should violate UNIQUE
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO referents (site_id, phrase_norm, phrase_display, targets_json, source, created_at, updated_at)
            VALUES ('site1', 'main gate', 'The Main Gate', '[{"camera_id":"cam2"}]', 'user', 1001.0, 1001.0)
            """
        )
        conn.commit()

    conn.close()
