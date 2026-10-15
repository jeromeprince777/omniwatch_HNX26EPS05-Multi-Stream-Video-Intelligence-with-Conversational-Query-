"""SQLite Connection and Schema Management for OmniWatch.

Enforces WAL mode, synchronous=FULL, and foreign keys ON.
Single source of truth for metadata and persistent knowledge base.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sites(
    site_id TEXT PRIMARY KEY,
    reference_now REAL,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS cameras(
    site_id TEXT,
    camera_id TEXT,
    label TEXT,
    video_path TEXT,
    start_time REAL NOT NULL,      -- epoch seconds UTC
    native_fps REAL,
    width INT,
    height INT,
    duration REAL,
    thumb_path TEXT,
    PRIMARY KEY(site_id, camera_id)
);

CREATE TABLE IF NOT EXISTS observations(
    obs_id INTEGER PRIMARY KEY,
    site_id TEXT,
    camera_id TEXT,
    tracklet_id INTEGER,
    frame_idx INT,
    ts REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,                       -- normalized 0..1
    px1 INT,
    py1 INT,
    px2 INT,
    py2 INT,                       -- absolute pixel coordinates
    frame_w INT,
    frame_h INT,                   -- native frame dimensions
    cls TEXT,
    det_conf REAL,
    quality REAL,
    color_primary TEXT,
    color_secondary TEXT,
    carries TEXT,                  -- carries: JSON list
    carrying_bag INT DEFAULT 0,    -- 1 if carrying backpack/handbag/suitcase
    bag_type TEXT,                 -- specific bag class
    bag_conf REAL DEFAULT 0.0,     -- detection confidence of associated bag
    thumb_path TEXT,               -- keyframe thumbnail cached at index time
    emb_row INT                    -- row in observation vector index
);
CREATE INDEX IF NOT EXISTS obs_cam_ts ON observations(site_id, camera_id, ts);

CREATE TABLE IF NOT EXISTS tracklets(
    tracklet_id INTEGER PRIMARY KEY,
    site_id TEXT,
    camera_id TEXT,
    local_track_id INT,
    cls TEXT,
    t_start REAL,
    t_end REAL,
    n_obs INT,
    best_obs_id INT,
    topk_obs_json TEXT,
    attrs_json TEXT,
    emb_row INT,
    global_id INT                  -- global_id: re-ID (stretch)
);
CREATE INDEX IF NOT EXISTS trk_cam_t ON tracklets(site_id, camera_id, t_start, t_end);

CREATE TABLE IF NOT EXISTS referents(
    referent_id INTEGER PRIMARY KEY,
    site_id TEXT,
    phrase_norm TEXT,
    phrase_display TEXT,
    kind TEXT DEFAULT 'location',
    targets_json TEXT NOT NULL,    -- [{"camera_id":"cam2","roi":[x1,y1,x2,y2]|null}]
    source TEXT,
    created_at REAL,
    updated_at REAL,
    UNIQUE(site_id, phrase_norm, kind)
);

CREATE TABLE IF NOT EXISTS referent_history(
    id INTEGER PRIMARY KEY,
    referent_id INT,
    old_targets_json TEXT,
    new_targets_json TEXT,
    changed_at REAL,
    actor TEXT
);

CREATE TABLE IF NOT EXISTS clarification_log(
    id INTEGER PRIMARY KEY,
    site_id TEXT,
    phrase_norm TEXT,
    session_id TEXT,
    asked_at REAL,
    resolved_at REAL,
    referent_id INT
);

CREATE TABLE IF NOT EXISTS pending_queries(
    session_id TEXT PRIMARY KEY,
    message TEXT,
    parsed_json TEXT,
    unresolved_json TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS query_log(
    id INTEGER PRIMARY KEY,
    ts REAL,
    session_id TEXT,
    message TEXT,
    parsed_json TEXT,
    mode TEXT,
    latency_json TEXT,
    result_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS standing_queries(
    id INTEGER PRIMARY KEY,
    site_id TEXT,
    message TEXT,
    parsed_json TEXT,
    active INT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS alerts(
    id INTEGER PRIMARY KEY,
    standing_id INT,
    tracklet_id INT,
    fired_at REAL,
    payload_json TEXT
);
"""


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Migrates existing database tables to include new columns if missing."""
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(observations)")
        cols = {row["name"] for row in cur.fetchall()}
        new_cols = [
            ("px1", "INT"),
            ("py1", "INT"),
            ("px2", "INT"),
            ("py2", "INT"),
            ("frame_w", "INT"),
            ("frame_h", "INT"),
            ("carrying_bag", "INT DEFAULT 0"),
            ("bag_type", "TEXT"),
            ("bag_conf", "REAL DEFAULT 0.0"),
            ("thumb_path", "TEXT"),
        ]
        for col_name, col_type in new_cols:
            if col_name not in cols:
                cur.execute(f"ALTER TABLE observations ADD COLUMN {col_name} {col_type}")
        conn.commit()
    except Exception:
        pass


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Creates a configured connection to the SQLite database.

    Enforces WAL journal mode, FULL synchrony, and foreign key constraints.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    _apply_migrations(conn)
    return conn


def init_db(db_path: str | Path) -> None:
    """Initializes the database schema with all required tables and indexes."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniWatch SQLite Database Manager")
    parser.add_argument(
        "--init",
        type=str,
        help="Initialize database schema at specified path (e.g. data/test.db)",
    )
    args = parser.parse_args()

    if args.init:
        init_db(args.init)
        print(f"Database successfully initialized at {args.init}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
