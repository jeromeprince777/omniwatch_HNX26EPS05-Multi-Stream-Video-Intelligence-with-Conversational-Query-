"""Automated Evaluation and Ablation Benchmark Runner for OmniWatch.

Responsibility:
- Single command execution to evaluate baseline vs OmniWatch modes.
- Generates the full ablation matrix defined in docs/02_RESEARCH_AND_EVAL_PLAN.md:
  B1, B1+, +tracklets, +attributes (OW-fast), +verify (OW-full).
- Measures warm-process latencies with 3 discarded warm-ups.
- Enforces test-set lock file protocol (eval/results/test_lock).
- Writes per_query.jsonl, summary.json, and config_used.json to eval/results/<run_id>/.
- Strictly adheres to AGENTS.md rule 2: No invented numbers; write-up numbers come from here.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.baseline import BaselineB1, BaselineB1Plus
from eval.metrics import (
    compute_existence_metrics,
    compute_hit_at_k,
    compute_latency_stats,
    compute_localization_error,
    compute_mrr,
)
from omniwatch.config import OmniWatchConfig, load_config
from omniwatch.db import get_connection
from omniwatch.index.vector import load_vector_index
from omniwatch.query.parser import RuleQueryParser
from omniwatch.query.resolver import resolve_referents, resolve_time
from omniwatch.query.retrieve import retrieve_candidates

logger = logging.getLogger("omniwatch.eval")


def get_git_revision_hash() -> str:
    """Returns current git commit hash if available, or 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:
        return "development_uncommitted"


def get_camera_start_times(db_path: str) -> dict[str, float]:
    """Fetches camera start_time mapping from SQLite database."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT camera_id, start_time FROM cameras")
        return {r[0]: float(r[1]) for r in cur.fetchall()}


def normalize_ground_truths(
    gts: list[dict],
    cam_start_times: dict[str, float],
) -> list[dict]:
    """Converts relative ground truth timestamps (< 100000s) to epoch seconds."""
    norm_gts = []
    for gt in gts:
        c_id = str(gt.get("camera_id", "")).strip()
        t0 = float(gt.get("t_start", 0.0))
        t1 = float(gt.get("t_end", t0))
        if t0 < 100000.0 and c_id in cam_start_times:
            c_base = cam_start_times[c_id]
            norm_gts.append({
                "camera_id": c_id,
                "t_start": c_base + t0,
                "t_end": c_base + t1,
            })
        else:
            norm_gts.append({
                "camera_id": c_id,
                "t_start": t0,
                "t_end": t1,
            })
    return norm_gts


def run_system_query(
    system_name: str,
    query_text: str,
    config: OmniWatchConfig,
    b1: BaselineB1 | None,
    b1_plus: BaselineB1Plus | None,
    encoder: Any,
    verifier: Any,
    conn: Any,
    trk_index: Any,
    cam_start_times: dict[str, float],
) -> list[dict[str, Any]]:
    """Runs a single query on the specified system or ablation row."""
    if system_name == "B1":
        if b1 is None:
            b1 = BaselineB1(config.storage.db_path, config.storage.vector_dir, config)
            b1.encoder = encoder
        return b1.query(query_text, top_k=5)

    if system_name == "B1+":
        if b1_plus is None:
            b1_plus = BaselineB1Plus(config.storage.db_path, config.storage.vector_dir, config)
            b1_plus.encoder = encoder
        return b1_plus.query(query_text, top_k=5)

    # OmniWatch modes / ablation rows
    parsed = RuleQueryParser.parse(query_text)

    cur = conn.cursor()
    cur.execute("SELECT MAX(start_time + duration) FROM cameras")
    max_row = cur.fetchone()
    ref_now = max_row[0] if max_row and max_row[0] else 0.0
    site_id: str | None = None
    if parsed.camera_ids:
        placeholders = ",".join("?" for _ in parsed.camera_ids)
        cur.execute(
            f"SELECT DISTINCT site_id FROM cameras WHERE camera_id IN ({placeholders}) ORDER BY site_id",
            parsed.camera_ids,
        )
        sites = [row[0] for row in cur.fetchall()]
        if len(sites) == 1:
            site_id = sites[0]
    elif parsed.location_phrases:
        cur.execute("SELECT site_id, camera_id, label FROM cameras")
        camera_rows = cur.fetchall()
        matched_sites: set[str] = set()
        for phrase in parsed.location_phrases:
            normalized = phrase.lower().replace("_", " ").strip()
            for candidate_site, camera_id, label in camera_rows:
                names = (
                    candidate_site.lower().replace("_", " "),
                    camera_id.lower().replace("_", " "),
                    (label or "").lower(),
                )
                if any(normalized == name or normalized in name or name in normalized for name in names if name):
                    matched_sites.add(candidate_site)
        if len(matched_sites) == 1:
            site_id = next(iter(matched_sites))

    time_range = resolve_time(parsed.time, reference_now=ref_now)
    ref_res = resolve_referents(parsed.location_phrases, parsed.camera_ids, site_id, conn)

    if parsed.camera_ids and not ref_res.resolved_targets:
        return []

    if system_name == "+tracklets":
        # Ablation: Tracklets with parser filter, but NO attribute fusion (w_attr=0, w_emb=1.0)
        ablate_cfg = load_config()
        ablate_cfg.current.weights.w_emb = 1.0
        ablate_cfg.current.weights.w_attr = 0.0
        # Strip attributes from parsed query
        parsed_no_attrs = parsed.model_copy(deep=True)
        parsed_no_attrs.attributes.color = []
        parsed_no_attrs.attributes.carries = []
        parsed_no_attrs.object_class_hint = None
        cands = retrieve_candidates(
            parsed_query=parsed_no_attrs,
            targets=ref_res.resolved_targets,
            time_range=time_range,
            conn=conn,
            vector_index=trk_index,
            config=ablate_cfg,
            site_id=site_id,
            encoder=encoder,
        )
    elif system_name in ("+attributes", "OW-fast"):
        # OW-fast: Tracklets + parser filter + attribute fusion + temporal dedup, no VLM
        cands = retrieve_candidates(
            parsed_query=parsed,
            targets=ref_res.resolved_targets,
            time_range=time_range,
            conn=conn,
            vector_index=trk_index,
            config=config,
            site_id=site_id,
            encoder=encoder,
        )
    elif system_name in ("+verify", "OW-full"):
        # OW-full: OW-fast + verifier pass
        cands = retrieve_candidates(
            parsed_query=parsed,
            targets=ref_res.resolved_targets,
            time_range=time_range,
            conn=conn,
            vector_index=trk_index,
            config=config,
            site_id=site_id,
            encoder=encoder,
        )
        if cands and config.vlm.enabled and verifier is not None:
            cands = verifier.rerank(
                cands,
                text_hypothesis=parsed.object_phrase,
                alpha=config.vlm.alpha,
                beta=config.vlm.beta,
                max_k=min(config.current.verify_top_k, len(cands)),
                conn=conn,
            )
    else:
        raise ValueError(f"Unknown system name: {system_name}")

    return [
        {
            "tracklet_id": c.tracklet_id,
            "camera_id": c.camera_id,
            "t_start": c.t_start,
            "t_end": c.t_end,
            "score": float(c.score),
            "bbox": list(c.bbox) if c.bbox is not None else [0.0, 0.0, 1.0, 1.0],
        }
        for c in cands
    ]


def run_evaluation(
    queries_path: str | Path,
    output_dir: str | Path = "eval/results",
    split: str = "dev",
    systems: list[str] | None = None,
    ablate: bool = False,
    override_test_lock: bool = False,
    config: OmniWatchConfig | None = None,
) -> dict[str, Any]:
    """Executes benchmark evaluation across all systems under test."""
    queries_path = Path(queries_path)
    output_dir = Path(output_dir)
    config = config or load_config()

    # Enforce test-set lock file
    test_lock_file = output_dir / "test_lock"
    if split == "test":
        if test_lock_file.exists() and not override_test_lock:
            raise RuntimeError(
                f"Test set is locked ({test_lock_file} exists)! "
                "Per docs/02 section 6, the test set is touched only once. "
                "To override, pass --i-know-this-burns-the-test-set."
            )

    # Determine systems list
    if ablate:
        systems = ["B1", "B1+", "+tracklets", "+attributes", "+verify"]
    elif not systems:
        systems = ["B1", "B1+", "OW-fast", "OW-full"]

    # Load queries
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("example") is True:
                continue  # Ignore fake template examples
            queries.append(item)

    if not queries:
        raise ValueError(f"No valid evaluation queries found in {queries_path}")

    # Prepare output directory
    run_id = f"eval_{split}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cam_start_times = get_camera_start_times(config.storage.db_path)
    trk_index = load_vector_index(config.storage.vector_dir, "trk")

    print("[OmniWatch Benchmark] Pre-loading models and warm-up...", flush=True)
    from omniwatch.embed.encoder import VLEncoder
    shared_encoder = VLEncoder(
        model_name=config.current.encoder_name,
        pretrained=config.current.encoder_pretrained,
        device=config.resolve_device(),
    )
    b1_instance: BaselineB1 | None = None
    if "B1" in systems:
        b1_instance = BaselineB1(config.storage.db_path, config.storage.vector_dir, config)
        b1_instance.encoder = shared_encoder

    b1_plus_instance: BaselineB1Plus | None = None
    if "B1+" in systems:
        b1_plus_instance = BaselineB1Plus(config.storage.db_path, config.storage.vector_dir, config)
        b1_plus_instance.encoder = shared_encoder

    verifier_instance: Any = None
    if any("+verify" in s or "OW-full" in s for s in systems) and config.vlm.enabled:
        from omniwatch.query.rerank import get_vlm_verifier
        verifier_instance = get_vlm_verifier(config)

    # Warm-up pass (3 warm-up queries discarded)
    warmup_texts = ["person", "car", "bus"]

    with get_connection(config.storage.db_path) as conn:
        for sys_name in systems:
            for w_text in warmup_texts:
                try:
                    run_system_query(
                        sys_name, w_text, config, b1_instance, b1_plus_instance, shared_encoder, verifier_instance, conn, trk_index, cam_start_times
                    )
                except Exception:
                    pass

        # Execution and timing loop
        results_by_system: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
        latencies_by_system: dict[str, list[float]] = {s: [] for s in systems}

        for idx, q in enumerate(queries, start=1):
            if idx % 5 == 0 or idx == len(queries):
                print(f"[OmniWatch Benchmark] Processing query {idx}/{len(queries)}...", flush=True)
            q_id = q.get("id", "q")
            q_text = q.get("text", "")
            raw_gt = q.get("gt", [])
            norm_gt = normalize_ground_truths(raw_gt, cam_start_times)

            for sys_name in systems:
                t0 = time.perf_counter()
                preds = run_system_query(
                    sys_name, q_text, config, b1_instance, b1_plus_instance, shared_encoder, verifier_instance, conn, trk_index, cam_start_times
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0

                h1_5 = compute_hit_at_k(preds, norm_gt, k=1, tau=5.0)
                h1_15 = compute_hit_at_k(preds, norm_gt, k=1, tau=15.0)
                h5_5 = compute_hit_at_k(preds, norm_gt, k=5, tau=5.0)
                h5_15 = compute_hit_at_k(preds, norm_gt, k=5, tau=15.0)
                mrr = compute_mrr(preds, norm_gt, tau=5.0)
                ex_metrics = compute_existence_metrics(preds, norm_gt, score_threshold=config.current.thresholds.find)
                loc_err = compute_localization_error(preds, norm_gt)

                rec = {
                    "query_id": q_id,
                    "query_text": q_text,
                    "system": sys_name,
                    "predictions": preds[:5],
                    "ground_truth": norm_gt,
                    "latency_ms": round(lat_ms, 2),
                    "hit1_tau5": h1_5,
                    "hit1_tau15": h1_15,
                    "hit5_tau5": h5_5,
                    "hit5_tau15": h5_15,
                    "mrr": round(mrr, 4),
                    "existence_correct": ex_metrics["correct"],
                    "is_negative": ex_metrics["is_negative"],
                    "is_false_positive": ex_metrics["is_false_positive"],
                    "loc_error_s": round(loc_err, 2) if loc_err is not None else None,
                }
                results_by_system[sys_name].append(rec)
                latencies_by_system[sys_name].append(lat_ms)

    # Save per_query.jsonl
    per_query_file = run_dir / "per_query.jsonl"
    with open(per_query_file, "w", encoding="utf-8") as f:
        for sys_name in systems:
            for rec in results_by_system[sys_name]:
                f.write(json.dumps(rec) + "\n")

    # Aggregate metrics
    summary: dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "n_queries": len(queries),
        "systems": {},
    }

    for sys_name in systems:
        recs = results_by_system[sys_name]
        lats = latencies_by_system[sys_name]
        pos_recs = [r for r in recs if r["is_negative"] == 0.0]
        neg_recs = [r for r in recs if r["is_negative"] == 1.0]

        h1_5_avg = float(np.mean([r["hit1_tau5"] for r in pos_recs])) if pos_recs else 0.0
        h1_15_avg = float(np.mean([r["hit1_tau15"] for r in pos_recs])) if pos_recs else 0.0
        h5_5_avg = float(np.mean([r["hit5_tau5"] for r in pos_recs])) if pos_recs else 0.0
        h5_15_avg = float(np.mean([r["hit5_tau15"] for r in pos_recs])) if pos_recs else 0.0
        mrr_avg = float(np.mean([r["mrr"] for r in pos_recs])) if pos_recs else 0.0
        ex_acc = float(np.mean([r["existence_correct"] for r in recs])) if recs else 0.0
        neg_fpr = float(np.mean([r["is_false_positive"] for r in neg_recs])) if neg_recs else 0.0

        loc_errors = [r["loc_error_s"] for r in pos_recs if r["loc_error_s"] is not None]
        med_loc_err = float(np.median(loc_errors)) if loc_errors else 0.0
        p90_loc_err = float(np.percentile(loc_errors, 90)) if loc_errors else 0.0

        lat_stats = compute_latency_stats(lats)

        summary["systems"][sys_name] = {
            "hit1_tau5": round(h1_5_avg, 4),
            "hit1_tau15": round(h1_15_avg, 4),
            "hit5_tau5": round(h5_5_avg, 4),
            "hit5_tau15": round(h5_15_avg, 4),
            "mrr": round(mrr_avg, 4),
            "existence_acc": round(ex_acc, 4),
            "negative_fpr": round(neg_fpr, 4),
            "loc_err_median_s": round(med_loc_err, 2),
            "loc_err_p90_s": round(p90_loc_err, 2),
            "latency_p50_ms": round(lat_stats["p50"], 1),
            "latency_p95_ms": round(lat_stats["p95"], 1),
            "latency_mean_ms": round(lat_stats["mean"], 1),
        }

    # Save summary.json
    summary_file = run_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save config_used.json
    config_used = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_hash": get_git_revision_hash(),
        "split": split,
        "preset": config.active_preset,
        "sample_fps": config.current.sample_fps,
        "detector_model": config.current.detector_model,
        "encoder_name": config.current.encoder_name,
        "encoder_pretrained": config.current.encoder_pretrained,
        "weights": config.current.weights.model_dump(),
        "thresholds": config.current.thresholds.model_dump(),
        "device": config.resolve_device(),
    }
    config_file = run_dir / "config_used.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_used, f, indent=2)

    # Lock test split
    if split == "test":
        with open(test_lock_file, "w", encoding="utf-8") as f:
            f.write(f"LOCKED: {datetime.now(UTC).isoformat()} git={config_used['git_hash']}\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniWatch Evaluation Benchmark Runner")
    parser.add_argument("--queries", type=str, default=None, help="Path to queries JSONL")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Dataset split")
    parser.add_argument("--out", type=str, default="eval/results", help="Output directory")
    parser.add_argument("--systems", type=str, default=None, help="Comma-separated system names")
    parser.add_argument("--ablate", action="store_true", help="Run full ablation matrix")
    parser.add_argument(
        "--i-know-this-burns-the-test-set",
        action="store_true",
        help="Override test set lock protection",
    )
    args = parser.parse_args()

    queries_path = args.queries or f"eval/queries/{args.split}.jsonl"
    sys_list = [s.strip() for s in args.systems.split(",")] if args.systems else None

    print(f"\n[OmniWatch Benchmark] Starting evaluation on split='{args.split}' (queries='{queries_path}')")
    summary = run_evaluation(
        queries_path=queries_path,
        output_dir=args.out,
        split=args.split,
        systems=sys_list,
        ablate=args.ablate,
        override_test_lock=args.i_know_this_burns_the_test_set,
    )
    print(f"\n[OmniWatch Benchmark] Evaluation Complete! Summary saved to {args.out}/{summary['run_id']}/summary.json")
    print(json.dumps(summary["systems"], indent=2))


if __name__ == "__main__":
    main()
