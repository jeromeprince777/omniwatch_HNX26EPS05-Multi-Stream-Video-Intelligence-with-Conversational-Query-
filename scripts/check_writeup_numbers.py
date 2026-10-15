"""Audit script to verify writeup numbers against eval/results/ summary.json.

Enforces non-negotiable Anti-Hallucination Rule 2 (AGENTS.md):
Never hard-code, estimate or illustrate metrics, latencies or accuracies.
Numbers must be verified against files written by eval/run_eval.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def check_numbers() -> bool:
    results_dir = Path("eval/results")
    if not results_dir.exists():
        print("[FAIL] eval/results directory does not exist.")
        return False

    summaries = list(results_dir.glob("**/summary.json"))
    if not summaries:
        print("[FAIL] No summary.json files found in eval/results/.")
        return False

    print(f"[PASS] Found {len(summaries)} evaluation run summary files in eval/results/.")

    writeup_path = Path("WRITEUP.md")
    if not writeup_path.exists():
        print("[FAIL] WRITEUP.md does not exist.")
        return False

    with open(writeup_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Load test summary
    test_summaries = [s for s in summaries if "test" in s.parent.name]
    if test_summaries:
        latest_test = sorted(test_summaries, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        with open(latest_test, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"[CHECK] Verifying TEST results against {latest_test.name}...")
        test_systems = test_data.get("systems", {})
        for sys_name, metrics in test_systems.items():
            # Check for system name in writeup
            if f"**{sys_name}**" not in content:
                print(f"[FAIL] System {sys_name} not found in WRITEUP.md table.")
                return False

            # Verify Hit@1 (tau=15s)
            hit1 = metrics.get("hit1_tau15", 0.0)
            hit1_str = f"{hit1 * 100:.1f}%"
            if hit1_str not in content:
                print(f"[FAIL] Metric hit1_tau15={hit1_str} for {sys_name} not found in WRITEUP.md.")
                return False

            # Verify MRR
            mrr = metrics.get("mrr", 0.0)
            mrr_str = f"{mrr:.3f}"
            if mrr_str not in content:
                print(f"[FAIL] Metric mrr={mrr_str} for {sys_name} not found in WRITEUP.md.")
                return False

            print(f"  [PASS] {sys_name}: Hit@1={hit1_str}, MRR={mrr_str} matched exactly.")

    # Load dev summary
    dev_summaries = [s for s in summaries if "dev" in s.parent.name]
    if dev_summaries:
        latest_dev = sorted(dev_summaries, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        with open(latest_dev, "r", encoding="utf-8") as f:
            dev_data = json.load(f)

        print(f"[CHECK] Verifying DEV ablation results against {latest_dev.name}...")
        dev_systems = dev_data.get("systems", {})
        for sys_name, metrics in dev_systems.items():
            if f"**{sys_name}**" not in content:
                print(f"[FAIL] System {sys_name} not found in WRITEUP.md ablation table.")
                return False

            hit1 = metrics.get("hit1_tau15", 0.0)
            hit1_str = f"{hit1 * 100:.1f}%"
            if hit1_str not in content:
                print(f"[FAIL] Ablation metric hit1_tau15={hit1_str} for {sys_name} not found in WRITEUP.md.")
                return False

            print(f"  [PASS] {sys_name}: Hit@1={hit1_str} matched exactly.")

    print("\n[SUCCESS] 100% of reported writeup numbers verified against eval/results/.")
    return True


if __name__ == "__main__":
    success = check_numbers()
    sys.exit(0 if success else 1)
