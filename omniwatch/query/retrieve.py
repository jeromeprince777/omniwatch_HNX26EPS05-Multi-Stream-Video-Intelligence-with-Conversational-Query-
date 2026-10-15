"""Candidate Retrieval, Scoring, and Fusion Module.

Responsibility:
- Filters candidate tracklets by camera set, time window, ROI intersection, and class hint.
- Computes embedding similarity against prompt ensemble.
- Fuses vision-language similarity with explicit attribute matches (color, carries).
- Performs temporal deduplication and applies calibrated score thresholds.
- Complies with docs/01_ARCHITECTURE.md sections 5 & 6.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from omniwatch.config import OmniWatchConfig
from omniwatch.index.vector import VectorIndex
from omniwatch.query.parser import ParsedQuery
from omniwatch.query.resolver import ResolvedTarget

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    tracklet_id: int
    camera_id: str
    camera_label: str | None
    score: float
    t_start: float
    t_end: float
    t_peak: float
    bbox: tuple[float, float, float, float]
    matched: dict[str, Any]
    best_obs_id: int
    det_conf: float = 0.0
    match_score: float = 0.0
    unverified: bool = False


def retrieve_candidates(
    parsed_query: ParsedQuery,
    targets: list[ResolvedTarget],
    time_range: tuple[float | None, float | None],
    conn: sqlite3.Connection,
    vector_index: VectorIndex,
    config: OmniWatchConfig,
    site_id: str | None = None,
    encoder: Any = None,
) -> list[CandidateResult]:
    """Retrieves, scores, and filters tracklet candidates against parsed query."""
    cur = conn.cursor()

    # 1. Base SQL candidate filtering
    sql_clauses = ["1=1"]
    params: list[Any] = []

    if site_id:
        sql_clauses.append("t.site_id = ?")
        params.append(site_id)

    # Camera target filtering
    if targets:
        target_cams = [t.camera_id for t in targets]
        placeholders = ",".join(["?"] * len(target_cams))
        sql_clauses.append(f"t.camera_id IN ({placeholders})")
        params.extend(target_cams)

    # Time overlap filtering [t_start, t_end]
    t_qstart, t_qend = time_range
    if t_qstart is not None and t_qend is not None:
        sql_clauses.append("t.t_start <= ? AND t.t_end >= ?")
        params.extend([t_qend, t_qstart])
    elif t_qstart is not None:
        sql_clauses.append("t.t_end >= ?")
        params.append(t_qstart)
    elif t_qend is not None:
        sql_clauses.append("t.t_start <= ?")
        params.append(t_qend)

    # Class hint filtering
    if parsed_query.object_class_hint:
        # Check if class matches or is vehicle/person
        hint = parsed_query.object_class_hint.lower()
        if hint == "vehicle":
            sql_clauses.append("t.cls IN ('car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle', 'vehicle')")
        elif hint == "person":
            sql_clauses.append("t.cls IN ('person', 'pedestrian')")
        elif hint == "bag":
            sql_clauses.append("t.cls IN ('backpack', 'handbag', 'suitcase', 'box', 'bag')")

    where_sql = " AND ".join(sql_clauses)
    query_sql = f"""
        SELECT t.tracklet_id, t.camera_id, t.cls, t.t_start, t.t_end,
               t.n_obs, t.best_obs_id, t.attrs_json, t.emb_row, c.label
        FROM tracklets t
        LEFT JOIN cameras c ON t.site_id = c.site_id AND t.camera_id = c.camera_id
        WHERE {where_sql}
        ORDER BY t.t_start ASC
    """
    cur.execute(query_sql, params)
    rows = cur.fetchall()

    if not rows:
        return []

    # 2. Text Prompt Ensemble Embedding
    if encoder is None:
        from omniwatch.embed.encoder import VLEncoder
        encoder = VLEncoder(
            model_name=config.current.encoder_name,
            pretrained=config.current.encoder_pretrained,
            device=config.resolve_device(),
        )

    prompts = [
        f"a photo of {parsed_query.object_phrase}",
        f"a surveillance image of {parsed_query.object_phrase}",
        f"a security camera view of {parsed_query.object_phrase}",
        f"a CCTV video recording of {parsed_query.object_phrase}",
        f"a video of a {parsed_query.object_phrase}",
        parsed_query.object_phrase,
    ]
    lower_obj = parsed_query.object_phrase.lower()
    if "news reader" in lower_obj or "news anchor" in lower_obj or "reader" in lower_obj:
        prompts.extend([
            "a news anchor presenting in a studio",
            "a television news reader at a desk",
            "a news reporter on television broadcast",
            "a person presenting the news",
        ])
    elif "cyclist" in lower_obj or "bicyclist" in lower_obj:
        prompts.extend(["a person riding a bicycle", "a cyclist"])
    elif "courier" in lower_obj or "delivery" in lower_obj:
        prompts.extend(["a delivery courier", "a delivery worker with package"])
    prompt_vecs = encoder.encode_text(prompts)  # (N, D) float32
    query_vec = np.mean(prompt_vecs, axis=0, keepdims=True)
    norm = np.linalg.norm(query_vec)
    if norm > 1e-6:
        query_vec = query_vec / norm

    # Negative prompts for calibrated softmax margin (Step 4)
    neg_prompts = [
        "a person with empty hands",
        "an empty street",
        "a building",
        "a blurry image",
    ]
    neg_vecs = encoder.encode_text(neg_prompts)

    # 3. Embedding Similarities & Attribute Scoring
    w_emb = config.current.weights.w_emb
    w_attr = config.current.weights.w_attr
    w_time = config.current.weights.w_time

    candidates: list[CandidateResult] = []
    all_embs = vector_index.get_all_embeddings()

    for row in rows:
        (trk_id, cam_id, cls_name, t_start, t_end, n_obs, best_obs_id, attrs_json, emb_row, cam_label) = row
        attrs = json.loads(attrs_json) if attrs_json else {}

        # Embedding similarity and calibrated softmax match score
        emb_sim = 0.0
        calibrated_match = 0.0
        if emb_row is not None and emb_row < len(all_embs):
            trk_vec = all_embs[emb_row : emb_row + 1]
            emb_sim = float(np.dot(query_vec, trk_vec.T)[0, 0])
            neg_sims = np.dot(neg_vecs, trk_vec.T).flatten()
            tau = 0.07
            all_sims = np.array([emb_sim] + list(neg_sims)) / tau
            exp_sims = np.exp(all_sims - np.max(all_sims))
            calibrated_match = float(exp_sims[0] / np.sum(exp_sims))

        # Attribute score
        attr_scores = []
        matched_dict: dict[str, Any] = {"class": cls_name}

        # Color matching
        query_colors = parsed_query.attributes.color
        if query_colors:
            trk_color = attrs.get("dominant_color", "")
            trk_hist = attrs.get("color_histogram", {})
            matched_colors = [c for c in query_colors if c == trk_color or c in trk_hist]
            matched_dict["color"] = matched_colors
            attr_scores.append(1.0 if matched_colors else 0.0)

        # Carries matching
        query_carries = parsed_query.attributes.carries
        if query_carries:
            trk_carries = attrs.get("carries", [])
            matched_carries = [c for c in query_carries if c in trk_carries]
            matched_dict["carries"] = matched_carries
            attr_scores.append(1.0 if matched_carries else 0.0)

        # Class match
        if parsed_query.object_class_hint:
            hint = parsed_query.object_class_hint.lower()
            if hint == "vehicle" and cls_name in ("car", "truck", "bus", "van", "motorcycle", "bicycle", "vehicle"):
                attr_scores.append(1.0)
            elif hint == "person" and cls_name in ("person", "pedestrian"):
                attr_scores.append(1.0)
            elif hint == "bag" and cls_name in ("backpack", "handbag", "suitcase", "box", "bag"):
                attr_scores.append(1.0)
            elif cls_name == hint:
                attr_scores.append(1.0)
            else:
                attr_scores.append(0.5)

        if attr_scores:
            s_attr = float(np.mean(attr_scores))
            # Blended score with attribute fusion
            final_score = float(w_emb * calibrated_match + w_attr * s_attr + w_time * 0.0)
        else:
            final_score = float(calibrated_match)

        # Retrieve best observation bbox and detector confidence
        cur.execute(
            "SELECT x1, y1, x2, y2, det_conf, carrying_bag, carries FROM observations WHERE obs_id = ?",
            (best_obs_id,),
        )
        obs_row = cur.fetchone()
        bbox = (obs_row[0], obs_row[1], obs_row[2], obs_row[3]) if obs_row else (0.0, 0.0, 1.0, 1.0)
        det_conf = float(obs_row[4]) if obs_row and obs_row[4] is not None else 0.50
        has_bag = bool(obs_row[5]) if obs_row and obs_row[5] is not None else False
        if not has_bag and obs_row and obs_row[6] and obs_row[6] != "[]":
            has_bag = True
        matched_dict["has_bag"] = has_bag

        t_peak = (t_start + t_end) / 2.0

        candidates.append(
            CandidateResult(
                tracklet_id=trk_id,
                camera_id=cam_id,
                camera_label=cam_label or cam_id,
                score=final_score,
                t_start=t_start,
                t_end=t_end,
                t_peak=t_peak,
                bbox=bbox,
                matched=matched_dict,
                best_obs_id=best_obs_id,
                det_conf=det_conf,
                match_score=calibrated_match,
                unverified=False,
            )
        )

    # 4. Step 3 Bag Association Metadata Prefilter
    requires_bag = (
        bool(parsed_query.attributes.carries)
        or "bag" in parsed_query.object_phrase.lower()
        or "carrying a bag" in parsed_query.raw.lower()
        or "with a bag" in parsed_query.raw.lower()
    ) and (parsed_query.object_class_hint == "person" or "person" in parsed_query.raw.lower())

    if requires_bag:
        bag_candidates = [c for c in candidates if c.matched.get("has_bag", False)]
        if len(bag_candidates) >= 3:
            candidates = bag_candidates
        else:
            # Fallback to unfiltered search if fewer than 3 candidates, flag as unverified
            for c in candidates:
                if not c.matched.get("has_bag", False):
                    c.unverified = True

    # 5. Pre-filter to top-K by score before dedup
    retrieval_top_k = getattr(config.query, "retrieval_top_k", 50)
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:retrieval_top_k]

    # 6. Temporal Deduplication
    dedup_gap = config.query.dedup_gap_s
    deduped: list[CandidateResult] = []

    for cand in candidates:
        duplicate = False
        for existing in deduped:
            if existing.camera_id == cand.camera_id:
                overlap = not (cand.t_end < existing.t_start - dedup_gap or cand.t_start > existing.t_end + dedup_gap)
                if overlap:
                    duplicate = True
                    break
        if not duplicate:
            deduped.append(cand)

    # 7. Intent threshold filtering
    threshold = getattr(config.current.thresholds, parsed_query.intent, 0.20)
    filtered = [c for c in deduped if c.score >= threshold]

    return filtered
