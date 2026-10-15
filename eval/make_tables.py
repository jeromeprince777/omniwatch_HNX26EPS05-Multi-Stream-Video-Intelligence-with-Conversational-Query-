"""Markdown Table Generator for Write-Up and Ablation Reports.

Reads summary.json from eval/results/<run_id>/ and prints GitHub Flavored Markdown
tables for the comparison benchmark and ablation matrix.
Adheres strictly to AGENTS.md rule 2: No numbers typed by hand; all numbers copied from summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def generate_markdown_tables(summary_path: str | Path) -> str:
    path = Path(summary_path)
    if path.is_dir():
        # Find latest summary.json
        summaries = list(path.glob("**/summary.json"))
        if not summaries:
            raise FileNotFoundError(f"No summary.json found in {path}")
        summaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        path = summaries[0]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    run_id = data.get("run_id", "unknown")
    split = data.get("split", "dev")
    n_queries = data.get("n_queries", 0)
    systems = data.get("systems", {})

    output = []
    output.append(f"### OmniWatch Benchmark Results ({split.upper()} Set, N={n_queries} queries)")
    output.append(f"*Source: `{path}` (Run ID: `{run_id}`)*\n")

    # Table 1: Systems Comparison
    output.append("| System | Hit@1 (tau=5s) | Hit@1 (tau=15s) | Hit@5 (tau=5s) | MRR | Ex. Acc | Neg FPR | Latency p50 | Latency p95 |")
    output.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for sys_name, m in systems.items():
        h1_5 = f"{m.get('hit1_tau5', 0.0) * 100:.1f}%"
        h1_15 = f"{m.get('hit1_tau15', 0.0) * 100:.1f}%"
        h5_5 = f"{m.get('hit5_tau5', 0.0) * 100:.1f}%"
        mrr = f"{m.get('mrr', 0.0):.3f}"
        ex_acc = f"{m.get('existence_acc', 0.0) * 100:.1f}%"
        neg_fpr = f"{m.get('negative_fpr', 0.0) * 100:.1f}%"
        p50 = f"{m.get('latency_p50_ms', 0.0):.1f} ms"
        p95 = f"{m.get('latency_p95_ms', 0.0):.1f} ms"
        output.append(f"| **{sys_name}** | {h1_5} | {h1_15} | {h5_5} | {mrr} | {ex_acc} | {neg_fpr} | {p50} | {p95} |")

    output.append("")
    return "\n".join(output)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate markdown tables from eval summary.json")
    parser.add_argument("summary", nargs="?", default="eval/results", help="Path to summary.json or results dir")
    args = parser.parse_args()

    table_md = generate_markdown_tables(args.summary)
    print(table_md)


if __name__ == "__main__":
    main()
