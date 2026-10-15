"""Dev Set Threshold and Weight Calibration Script (Prompt 09).

Grid-searches retrieval fusion weights and intent score thresholds
strictly on the dev split. Writes optimal calibrated values to config.yaml
and logs the metric surface to eval/results/tuning_log.json.

Strict Protocol:
- Asserts split is NOT test.
- Calibrates exclusively on eval/queries/dev.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval.metrics import compute_existence_metrics, compute_hit_at_k, compute_mrr
from omniwatch.config import load_config
from omniwatch.db import get_connection
from omniwatch.embed.encoder import VLEncoder
from omniwatch.index.vector import load_vector_index
from omniwatch.query.parser import RuleQueryParser
from omniwatch.query.resolver import resolve_referents, resolve_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_dev")


def run_dev_tuning(
    dev_path: str | Path = "eval/queries/dev.jsonl",
    output_log: str | Path = "eval/results/tuning_log.json",
    config_path: str | Path = "config.yaml",
) -> dict[str, Any]:
    dev_path = Path(dev_path)
    # Anti-hallucination / strict split assertion
    assert "test" not in str(dev_path).lower(), (
        f"CRITICAL ERROR: tune_dev.py is strictly forbidden from accessing the test set! Given: {dev_path}"
    )
    assert dev_path.exists(), f"Dev queries file not found: {dev_path}"

    config = load_config()
    conn = get_connection(config.storage.db_path)

    # 1. Load dev queries
    dev_queries = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dev_queries.append(json.loads(line))

    logger.info("Loaded %d dev queries for calibration from %s", len(dev_queries), dev_path)

    # 2. Pre-extract candidates & features for instant in-memory grid evaluation
    cur = conn.cursor()
    cur.execute("SELECT MAX(start_time + duration) FROM cameras")
    max_row = cur.fetchone()
    ref_now = max_row[0] if max_row and max_row[0] else 0.0
    cur.execute("SELECT site_id FROM cameras LIMIT 1")
    site_row = cur.fetchone()
    site_id = site_row[0] if site_row and site_row[0] else "default_site"

    # Pre-load OpenCLIP encoder and vector index
    vector_dir = Path(config.storage.vector_dir)
    trk_index = load_vector_index(vector_dir, "trk")
    all_trk_embs = trk_index.get_all_embeddings()

    encoder = VLEncoder(
        model_name=config.current.encoder_name,
        pretrained=config.current.encoder_pretrained,
        device=config.resolve_device(),
    )

    query_precomputed: list[dict[str, Any]] = []

    for q in dev_queries:
        q_text = q["text"]
        parsed = RuleQueryParser.parse(q_text)
        time_range = resolve_time(parsed.time, reference_now=ref_now)
        ref_res = resolve_referents(parsed.location_phrases, parsed.camera_ids, site_id, conn)

        # Check explicit camera mismatch
        if parsed.camera_ids and not ref_res.resolved_targets:
            query_precomputed.append({
                "query": q,
                "parsed": parsed,
                "candidates": [],
            })
            continue

        # SQL filtering
        sql_clauses = ["1=1"]
        params: list[Any] = []
        if site_id:
            sql_clauses.append("t.site_id = ?")
            params.append(site_id)

        if ref_res.resolved_targets:
            target_cams = [t.camera_id for t in ref_res.resolved_targets]
            placeholders = ",".join(["?"] * len(target_cams))
            sql_clauses.append(f"t.camera_id IN ({placeholders})")
            params.extend(target_cams)

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

        where_sql = " AND ".join(sql_clauses)
        cur.execute(
            f"""
            SELECT t.tracklet_id, t.camera_id, t.cls, t.t_start, t.t_end,
                   t.best_obs_id, t.attrs_json, t.emb_row
            FROM tracklets t
            WHERE {where_sql}
            ORDER BY t.t_start ASC
            """,
            params,
        )
        rows = cur.fetchall()

        # Query vector
        prompts = [
            f"a photo of {parsed.object_phrase}",
            f"a surveillance image of {parsed.object_phrase}",
            parsed.object_phrase,
        ]
        prompt_vecs = encoder.encode_text(prompts)
        query_vec = np.mean(prompt_vecs, axis=0, keepdims=True)
        norm = np.linalg.norm(query_vec)
        if norm > 1e-6:
            query_vec = query_vec / norm

        cands = []
        for r in rows:
            trk_id, cam_id, cls_name, t0, t1, best_obs_id, attrs_json, emb_row = r
            attrs = json.loads(attrs_json) if attrs_json else {}

            emb_sim = 0.0
            if emb_row is not None and emb_row < len(all_trk_embs):
                trk_vec = all_trk_embs[emb_row : emb_row + 1]
                emb_sim = float(np.dot(query_vec, trk_vec.T)[0, 0])

            attr_scores = []
            if parsed.attributes.color:
                trk_col = attrs.get("dominant_color", "")
                trk_hist = attrs.get("color_histogram", {})
                matched_col = [c for c in parsed.attributes.color if c == trk_col or c in trk_hist]
                attr_scores.append(1.0 if matched_col else 0.0)

            if parsed.attributes.carries:
                trk_car = attrs.get("carries", [])
                matched_car = [c for c in parsed.attributes.carries if c in trk_car]
                attr_scores.append(1.0 if matched_car else 0.0)

            if parsed.object_class_hint:
                attr_scores.append(1.0 if cls_name == parsed.object_class_hint else 0.5)

            s_attr = float(np.mean(attr_scores)) if attr_scores else None

            cands.append({
                "tracklet_id": trk_id,
                "camera_id": cam_id,
                "t_start": t0,
                "t_end": t1,
                "emb_sim": emb_sim,
                "s_attr": s_attr,
                "has_attrs": bool(attr_scores),
            })

        query_precomputed.append({
            "query": q,
            "parsed": parsed,
            "candidates": cands,
        })

    # 3. Grid Search Surface
    w_pairs = [
        (0.70, 0.30),
        (0.65, 0.35),
        (0.60, 0.40),
        (0.55, 0.45),
        (0.50, 0.50),
    ]
    thresholds = [0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32, 0.35]

    surface_results: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_score = -1.0

    logger.info("Evaluating grid search surface (%d combinations)...", len(w_pairs) * len(thresholds))

    for w_emb, w_attr in w_pairs:
        for thresh in thresholds:
            all_preds: list[list[dict[str, Any]]] = []
            all_gts: list[list[dict[str, Any]]] = []
            all_neg_flags: list[bool] = []

            for q_data in query_precomputed:
                q = q_data["query"]
                is_neg = q.get("expects_no_match", False) or not q.get("gt")
                all_neg_flags.append(is_neg)
                gts = [
                    {"camera_id": gt["camera_id"], "t_start": gt["t_start"], "t_end": gt["t_end"]}
                    for gt in q.get("gt", [])
                ]
                all_gts.append(gts)

                # Score candidates
                scored = []
                for c in q_data["candidates"]:
                    if c["has_attrs"]:
                        score = w_emb * c["emb_sim"] + w_attr * c["s_attr"]
                    else:
                        score = c["emb_sim"]

                    if score >= thresh:
                        scored.append({
                            "camera_id": c["camera_id"],
                            "t_start": c["t_start"],
                            "t_end": c["t_end"],
                            "score": float(score),
                        })

                # Sort and deduplicate intervals
                scored.sort(key=lambda x: x["score"], reverse=True)
                # Deduplicate temporally (gap = 2.0s)
                deduped: list[dict[str, Any]] = []
                for s in scored:
                    dup = False
                    for existing in deduped:
                        if existing["camera_id"] == s["camera_id"]:
                            if not (s["t_end"] < existing["t_start"] - 2.0 or s["t_start"] > existing["t_end"] + 2.0):
                                dup = True
                                break
                    if not dup:
                        deduped.append(s)

                all_preds.append(deduped[:5])

            # Compute metrics across queries
            pos_indices = [i for i, g in enumerate(all_gts) if len(g) > 0]
            hits1 = [compute_hit_at_k(all_preds[i], all_gts[i], k=1, tau=15.0) for i in pos_indices]
            hit1 = float(np.mean(hits1)) if hits1 else 0.0

            hits5 = [compute_hit_at_k(all_preds[i], all_gts[i], k=5, tau=15.0) for i in pos_indices]
            hit5 = float(np.mean(hits5)) if hits5 else 0.0

            mrrs = [compute_mrr(all_preds[i], all_gts[i], tau=15.0) for i in pos_indices]
            mrr = float(np.mean(mrrs)) if mrrs else 0.0

            exist_res = [compute_existence_metrics(all_preds[i], all_gts[i], score_threshold=thresh) for i in range(len(all_preds))]
            exist_acc = float(np.mean([r["correct"] for r in exist_res]))
            neg_items = [r for r in exist_res if r["is_negative"] == 1.0]
            neg_fpr = float(np.mean([r["is_false_positive"] for r in neg_items])) if neg_items else 0.0

            row = {
                "w_emb": w_emb,
                "w_attr": w_attr,
                "threshold": thresh,
                "hit1_tau15": round(hit1, 4),
                "hit5_tau15": round(hit5, 4),
                "mrr": round(mrr, 4),
                "existence_acc": round(exist_acc, 4),
                "negative_fpr": round(neg_fpr, 4),
            }
            surface_results.append(row)

            # Optimization criterion:
            # 1. Strict priority: zero false positives on negatives (neg_fpr == 0.0)
            # 2. Maximum Hit@1 and MRR
            if neg_fpr == 0.0:
                composite = hit1 * 1.0 + mrr * 0.5 + exist_acc * 0.5
                if composite > best_score:
                    best_score = composite
                    best_config = row

    # Sort surface for display
    surface_results.sort(key=lambda r: (r["negative_fpr"], -r["hit1_tau15"], -r["mrr"]))

    # 4. Save tuning log
    output_log = Path(output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": "dev",
                "num_queries": len(dev_queries),
                "best_calibrated": best_config,
                "surface": surface_results,
            },
            f,
            indent=2,
        )

    logger.info("Saved tuning log to %s", output_log)
    logger.info("Best calibrated parameters: %s", best_config)

    # 5. Update config.yaml with calibrated values
    if best_config and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f)

        # Update balanced preset
        if "presets" in cfg_yaml and "balanced" in cfg_yaml["presets"]:
            b_preset = cfg_yaml["presets"]["balanced"]
            b_preset["weights"]["w_emb"] = best_config["w_emb"]
            b_preset["weights"]["w_attr"] = best_config["w_attr"]
            thresh_val = best_config["threshold"]
            b_preset["thresholds"]["exists"] = thresh_val
            b_preset["thresholds"]["find"] = thresh_val
            b_preset["thresholds"]["when"] = thresh_val
            b_preset["thresholds"]["count"] = round(thresh_val + 0.03, 3)
            b_preset["thresholds"]["path"] = thresh_val

        # Update fast preset
        if "presets" in cfg_yaml and "fast" in cfg_yaml["presets"]:
            f_preset = cfg_yaml["presets"]["fast"]
            f_preset["weights"]["w_emb"] = best_config["w_emb"]
            f_preset["weights"]["w_attr"] = best_config["w_attr"]
            thresh_val = best_config["threshold"]
            f_preset["thresholds"]["exists"] = thresh_val
            f_preset["thresholds"]["find"] = thresh_val
            f_preset["thresholds"]["when"] = thresh_val
            f_preset["thresholds"]["count"] = round(thresh_val + 0.03, 3)
            f_preset["thresholds"]["path"] = thresh_val

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_yaml, f, sort_keys=False)

        logger.info("Updated %s with calibrated thresholds (threshold=%.2f, w_emb=%.2f, w_attr=%.2f)",
                    config_path, best_config["threshold"], best_config["w_emb"], best_config["w_attr"])

    return {
        "best_config": best_config,
        "total_combinations": len(surface_results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate OmniWatch weights and thresholds on dev set.")
    parser.add_argument("--dev", default="eval/queries/dev.jsonl", help="Path to dev queries JSONL")
    parser.add_argument("--out", default="eval/results/tuning_log.json", help="Path to output tuning log")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_dev_tuning(dev_path=args.dev, output_log=args.out, config_path=args.config)
