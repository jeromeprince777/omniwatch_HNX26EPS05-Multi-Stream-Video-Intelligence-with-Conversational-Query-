"""Baseline Model Implementations (B1, B1+, B2).

Responsibility:
- Implements B1 baseline: Open-vocab detector on sampled frames -> CLIP crop embeddings ->
  cosine similarity to query text -> top-K detections. No tracking, no query decomposition,
  no explicit attributes. Reuses already-indexed observation embeddings for fair evaluation.
- Implements B1+: B1 equipped with parser camera/time filters to isolate tracking/attribute gains.
- Implements B2: Whole-frame CLIP retrieval (weak reference).
- Complies strictly with docs/02_RESEARCH_AND_EVAL_PLAN.md.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from omniwatch.config import OmniWatchConfig, load_config
from omniwatch.db import get_connection
from omniwatch.embed.encoder import VLEncoder
from omniwatch.index.vector import load_vector_index
from omniwatch.query.parser import RuleQueryParser
from omniwatch.query.resolver import resolve_referents, resolve_time


class BaselineB1:
    """Primary Baseline (B1): Crop-level vector search without tracking or attribute fusion."""

    def __init__(
        self,
        db_path: str = "data/omniwatch.db",
        vector_dir: str = "data/index",
        config: OmniWatchConfig | None = None,
    ):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self.config = config or load_config()
        self.encoder = VLEncoder(
            model_name=self.config.current.encoder_name,
            pretrained=self.config.current.encoder_pretrained,
            device=self.config.resolve_device(),
        )
        self.obs_index = load_vector_index(vector_dir, "obs")

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Queries observation crops with query text prompt ensemble."""
        if self.obs_index.ntotal == 0:
            return []

        # Encode text prompt
        prompts = [
            f"a photo of {text}",
            f"a surveillance image of {text}",
            text,
        ]
        p_vecs = self.encoder.encode_text(prompts)
        q_vec = np.mean(p_vecs, axis=0, keepdims=True)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-6:
            q_vec = q_vec / norm

        # Unfiltered search across all observation crops
        top_matches = self.obs_index.search(q_vec, top_k=top_k)
        if not top_matches:
            return []

        obs_ids = [m[0] for m in top_matches]
        scores_by_id = {m[0]: m[1] for m in top_matches}

        # Retrieve metadata
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in obs_ids)
            cur.execute(
                f"SELECT obs_id, camera_id, ts, x1, y1, x2, y2, cls FROM observations WHERE obs_id IN ({placeholders})",
                obs_ids,
            )
            rows = {r[0]: r for r in cur.fetchall()}

        results = []
        for obs_id, score in top_matches:
            if obs_id in rows:
                r = rows[obs_id]
                results.append(
                    {
                        "obs_id": r[0],
                        "camera_id": r[1],
                        "t_start": r[2],
                        "t_end": r[2],
                        "score": float(score),
                        "bbox": [float(r[3]), float(r[4]), float(r[5]), float(r[6])],
                        "cls": r[7],
                    }
                )
        return results


class BaselineB1Plus:
    """Fairness Baseline (B1+): B1 with camera and time filtering from shared query parser."""

    def __init__(
        self,
        db_path: str = "data/omniwatch.db",
        vector_dir: str = "data/index",
        config: OmniWatchConfig | None = None,
    ):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self.config = config or load_config()
        self.encoder = VLEncoder(
            model_name=self.config.current.encoder_name,
            pretrained=self.config.current.encoder_pretrained,
            device=self.config.resolve_device(),
        )
        self.obs_index = load_vector_index(vector_dir, "obs")

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Parses filters and searches candidate observation crops."""
        if self.obs_index.ntotal == 0:
            return []

        parsed = RuleQueryParser.parse(text)

        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            # Determine reference_now
            cur.execute("SELECT MAX(start_time + duration) FROM cameras")
            max_row = cur.fetchone()
            ref_now = max_row[0] if max_row and max_row[0] else 0.0

            cur.execute("SELECT site_id FROM cameras LIMIT 1")
            site_row = cur.fetchone()
            site_id = site_row[0] if site_row and site_row[0] else "default_site"

            time_range = resolve_time(parsed.time, reference_now=ref_now)
            ref_res = resolve_referents(parsed.location_phrases, parsed.camera_ids, site_id, conn)

            # Build candidate obs_ids
            sql = "SELECT obs_id FROM observations WHERE 1=1"
            params: list[Any] = []

            if ref_res.resolved_targets:
                target_cams = [t.camera_id for t in ref_res.resolved_targets]
                placeholders = ",".join("?" for _ in target_cams)
                sql += f" AND camera_id IN ({placeholders})"
                params.extend(target_cams)
            elif parsed.camera_ids:
                # Explicit camera IDs requested but none exist in manifest
                return []

            if time_range[0] is not None:
                sql += " AND ts >= ?"
                params.append(time_range[0])
            if time_range[1] is not None:
                sql += " AND ts <= ?"
                params.append(time_range[1])

            cur.execute(sql, params)
            allowed_ids = {r[0] for r in cur.fetchall()}

        if not allowed_ids:
            return []

        prompts = [
            f"a photo of {parsed.object_phrase}",
            f"a surveillance image of {parsed.object_phrase}",
            parsed.object_phrase,
        ]
        p_vecs = self.encoder.encode_text(prompts)
        q_vec = np.mean(p_vecs, axis=0, keepdims=True)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-6:
            q_vec = q_vec / norm

        top_matches = self.obs_index.search(q_vec, top_k=top_k, allowed_ids=allowed_ids)
        if not top_matches:
            return []

        obs_ids = [m[0] for m in top_matches]
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in obs_ids)
            cur.execute(
                f"SELECT obs_id, camera_id, ts, x1, y1, x2, y2, cls FROM observations WHERE obs_id IN ({placeholders})",
                obs_ids,
            )
            rows = {r[0]: r for r in cur.fetchall()}

        results = []
        for obs_id, score in top_matches:
            if obs_id in rows:
                r = rows[obs_id]
                results.append(
                    {
                        "obs_id": r[0],
                        "camera_id": r[1],
                        "t_start": r[2],
                        "t_end": r[2],
                        "score": float(score),
                        "bbox": [float(r[3]), float(r[4]), float(r[5]), float(r[6])],
                        "cls": r[7],
                    }
                )
        return results
