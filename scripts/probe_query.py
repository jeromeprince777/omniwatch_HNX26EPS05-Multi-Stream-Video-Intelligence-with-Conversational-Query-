"""Probe Query CLI for OmniWatch Vector Index Retrieval.

Tests index loading, text prompt encoding, and vector similarity search
over ingested tracklets without re-running video models.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omniwatch.config import load_config
from omniwatch.embed.encoder import VLEncoder
from omniwatch.index.vector import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniWatch Raw Vector Probe Query")
    parser.add_argument("query", type=str, nargs="?", default="red car", help="Natural language search phrase")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate tracklets to retrieve")
    parser.add_argument("--db", type=str, default="data/omniwatch.db", help="Path to SQLite database")
    parser.add_argument("--index-dir", type=str, default="data/index", help="Path to vector index directory")
    args = parser.parse_args()

    cfg = load_config()
    index_dir = Path(args.index_dir)
    db_path = Path(args.db)

    if not (index_dir / "trk.index").exists():
        print(f"Error: Vector index 'trk.index' not found in {index_dir}. Run ingestion first.")
        sys.exit(1)

    print(f"[OmniWatch Probe Query] Query: '{args.query}'")

    # 1. Load Vector Index
    t_load_start = time.perf_counter()
    trk_index = VectorIndex(dimension=512, index_name="trk")
    trk_index.load(index_dir)
    load_ms = (time.perf_counter() - t_load_start) * 1000.0

    # 2. Encode Query
    t_enc_start = time.perf_counter()
    encoder = VLEncoder(
        model_name=cfg.current.encoder_name,
        pretrained=cfg.current.encoder_pretrained,
        device=cfg.resolve_device(),
    )
    query_vector = encoder.encode_prompt_ensemble(args.query)
    enc_ms = (time.perf_counter() - t_enc_start) * 1000.0

    # 3. Search Vector Index
    t_search_start = time.perf_counter()
    search_results = trk_index.search(query_vector, top_k=args.top_k)
    search_ms = (time.perf_counter() - t_search_start) * 1000.0

    total_latency_ms = enc_ms + search_ms

    print(f"[Timing] Index load: {load_ms:.2f} ms | Text encoding: {enc_ms:.2f} ms | Vector search: {search_ms:.3f} ms | Total: {total_latency_ms:.2f} ms")
    print(f"[Results] Found {len(search_results)} candidate tracklets in index ({trk_index.ntotal} total indexed):")

    if not search_results:
        print("  (No tracklets matched the search query)")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        print("-" * 75)
        print(f"{'Rank':<5} {'Score':<8} {'Tracklet':<10} {'Camera':<10} {'Class':<10} {'Time Window (s)':<20} {'Attributes'}")
        print("-" * 75)

        for rank, (tracklet_id, score) in enumerate(search_results, 1):
            row = conn.execute(
                "SELECT * FROM tracklets WHERE tracklet_id = ?",
                (tracklet_id,),
            ).fetchone()
            if not row:
                continue

            attrs = json.loads(row["attrs_json"]) if row["attrs_json"] else {}
            color = attrs.get("color_primary", "unknown")
            carries = ", ".join(attrs.get("carries", [])) or "none"
            attr_str = f"color={color}, carries={carries}"
            time_window = f"{row['t_start']:.1f} - {row['t_end']:.1f}"

            print(
                f"{rank:<5} {score:<8.4f} #{row['tracklet_id']:<9} {row['camera_id']:<10} {row['cls']:<10} {time_window:<20} {attr_str}"
            )
        print("-" * 75)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
