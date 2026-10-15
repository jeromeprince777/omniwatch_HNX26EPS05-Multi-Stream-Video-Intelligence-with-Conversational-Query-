"""Cross-Camera Re-Identification and Multi-Camera Timeline Linking (Prompt 10).

Complies with:
- docs/00_PRD (stretch), docs/01 (3: tracklets.global_id), prompts/10_cross_camera_reid_stretch.md.
- Links tracklets across distinct camera feeds into unified global identities.
- Enforces strict spatio-temporal causality:
    - Transit interval > 0 (candidate starts after source ends).
    - Transit interval <= max_transit_time_s.
    - Never links overlapping tracklets on the same camera.
    - Visual appearance similarity using normalized OpenCLIP tracklet embeddings.
- Produces deterministic, grounded multi-hop timeline paths:
    "Camera 1 (09:14:00) -> Camera 2 (09:14:12)".
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

logger = logging.getLogger("omniwatch.reid")


def compute_appearance_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two normalized feature vectors."""
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(vec1 / n1, (vec2 / n2).T))


def link_cross_camera_tracklets(
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    max_transit_time_s: float = 300.0,
    min_transit_time_s: float = 0.0,
    similarity_thresh: float = 0.65,
) -> list[list[dict[str, Any]]]:
    """Groups tracklets across cameras into chronological global identity paths.

    Args:
        tracklets: List of tracklet dicts (must contain tracklet_id, camera_id, cls, t_start, t_end).
        embeddings: Dict mapping tracklet_id to normalized appearance vector.
        max_transit_time_s: Maximum transit interval allowed between camera hops.
        min_transit_time_s: Minimum transit interval (default 0, causality check).
        similarity_thresh: Minimum OpenCLIP cosine similarity required to link.

    Returns:
        List of chronological paths, each being a list of tracklet dicts.
    """
    if not tracklets:
        return []

    # Sort tracklets strictly by t_start
    sorted_trks = sorted(tracklets, key=lambda t: t.get("t_start", 0.0))
    visited: set[int] = set()
    paths: list[list[dict[str, Any]]] = []

    for _i, src in enumerate(sorted_trks):
        src_id = src["tracklet_id"]
        if src_id in visited:
            continue

        current_path = [src]
        visited.add(src_id)
        current_trk = src

        # Greedily search for the best subsequent hop on a different camera
        while True:
            best_candidate = None
            best_sim = -1.0
            src_vec = embeddings.get(current_trk["tracklet_id"])

            for cand in sorted_trks:
                cand_id = cand["tracklet_id"]
                if cand_id in visited:
                    continue

                # Must be different camera
                if cand["camera_id"] == current_trk["camera_id"]:
                    continue

                # Class must match
                if cand.get("cls") != current_trk.get("cls"):
                    continue

                # Spatio-temporal transit constraints
                dt = cand["t_start"] - current_trk["t_end"]
                if dt < min_transit_time_s or dt > max_transit_time_s:
                    continue

                # Appearance similarity
                cand_vec = embeddings.get(cand_id)
                if src_vec is not None and cand_vec is not None:
                    sim = compute_appearance_similarity(src_vec, cand_vec)
                else:
                    sim = 0.50

                if sim >= similarity_thresh and sim > best_sim:
                    best_sim = sim
                    best_candidate = cand

            if best_candidate is not None:
                visited.add(best_candidate["tracklet_id"])
                current_path.append(best_candidate)
                current_trk = best_candidate
            else:
                break

        paths.append(current_path)

    return paths


def trace_object_path(
    source_tracklet_id: int,
    conn: sqlite3.Connection,
    max_transit_time_s: float = 300.0,
    similarity_thresh: float = 0.60,
) -> dict[str, Any]:
    """Finds the cross-camera path starting from or including source_tracklet_id."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.tracklet_id, t.camera_id, t.cls, t.t_start, t.t_end, t.emb_row, c.label
        FROM tracklets t
        LEFT JOIN cameras c ON t.camera_id = c.camera_id
        WHERE t.tracklet_id = ?
        """,
        (source_tracklet_id,),
    )
    src_row = cur.fetchone()
    if not src_row:
        return {"error": "Tracklet not found", "hops": []}

    target_cls = src_row[2]

    # Fetch all candidate tracklets with same class
    cur.execute(
        """
        SELECT t.tracklet_id, t.camera_id, t.cls, t.t_start, t.t_end, t.emb_row, c.label
        FROM tracklets t
        LEFT JOIN cameras c ON t.camera_id = c.camera_id
        WHERE t.cls = ?
        ORDER BY t.t_start ASC
        """,
        (target_cls,),
    )
    all_rows = cur.fetchall()

    # Load vector index for appearance embeddings
    from pathlib import Path

    from omniwatch.config import load_config
    from omniwatch.index.vector import load_vector_index

    cfg = load_config()
    trk_index = load_vector_index(Path(cfg.storage.vector_dir), "trk")
    all_embs = trk_index.get_all_embeddings()

    tracklet_dicts = []
    emb_dict: dict[int, np.ndarray] = {}

    for r in all_rows:
        t_id, cam_id, cls_name, t0, t1, emb_row, label = r
        tracklet_dicts.append({
            "tracklet_id": t_id,
            "camera_id": cam_id,
            "cls": cls_name,
            "t_start": t0,
            "t_end": t1,
            "label": label or cam_id,
        })
        if emb_row is not None and emb_row < len(all_embs):
            emb_dict[t_id] = all_embs[emb_row]

    linked_paths = link_cross_camera_tracklets(
        tracklet_dicts,
        emb_dict,
        max_transit_time_s=max_transit_time_s,
        similarity_thresh=similarity_thresh,
    )

    # Find the path containing source_tracklet_id
    matched_path = []
    for p in linked_paths:
        if any(t["tracklet_id"] == source_tracklet_id for t in p):
            matched_path = p
            break

    if not matched_path:
        matched_path = [{
            "tracklet_id": src_row[0],
            "camera_id": src_row[1],
            "cls": src_row[2],
            "t_start": src_row[3],
            "t_end": src_row[4],
            "label": src_row[6] or src_row[1],
        }]

    # Format human-readable timeline
    from omniwatch.api.main import format_timestamp
    hops = []
    for h in matched_path:
        hops.append({
            "tracklet_id": h["tracklet_id"],
            "camera_id": h["camera_id"],
            "camera_label": h.get("label", h["camera_id"]),
            "t_start": h["t_start"],
            "t_end": h["t_end"],
            "time_display": format_timestamp(h["t_start"]),
            "thumb_url": f"/api/results/{h['tracklet_id']}/thumb",
            "clip_url": f"/api/results/{h['tracklet_id']}/clip",
        })

    # Summary text
    timeline_str = " -> ".join([f"{h['camera_label']} ({h['time_display']})" for h in hops])
    return {
        "source_tracklet_id": source_tracklet_id,
        "object_class": target_cls,
        "hop_count": len(hops),
        "timeline_summary": timeline_str,
        "hops": hops,
    }
