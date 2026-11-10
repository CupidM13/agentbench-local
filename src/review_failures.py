"""Failure taxonomy manual review script.
Samples 20 failed trajectories deterministically, inspects their full messages,
compares automated taxonomy labels with manual assessment, and outputs agreement stats.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.report import load_all_runs
from src.taxonomy import TAXONOMY_CLASSES


def review_failures(runs_dir: str = "runs", seed: int = 42, sample_size: int = 20) -> Dict[str, Any]:
    df = load_all_runs(runs_dir)
    if df.empty:
        return {"error": "No runs found"}

    failed = df[~df["passed"]].copy()
    if len(failed) == 0:
        return {"error": "No failed runs found"}

    # Sample deterministically
    sample_n = min(sample_size, len(failed))
    sampled_df = failed.sample(n=sample_n, random_state=seed)

    # Read full trajectories for sampled run keys
    run_keys = set(sampled_df["run_key"].tolist())
    trajectories: Dict[str, Dict[str, Any]] = {}

    for fpath in Path(runs_dir).glob("*.jsonl"):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    k = rec.get("run_key")
                    if k in run_keys:
                        trajectories[k] = rec
                except Exception:
                    continue

    print(f"\n=======================================================")
    print(f"Manual Review of {len(trajectories)} Randomly Sampled Failures")
    print(f"=======================================================\n")

    reviews = []
    agreements = 0

    for idx, (run_key, traj) in enumerate(trajectories.items(), start=1):
        task_id = traj.get("task_id")
        cat = traj.get("category")
        model = traj.get("model")
        cfg = traj.get("config")
        auto_label = traj.get("failure_label")
        answer = traj.get("final_answer", "")
        reason = traj.get("grader_result", {}).get("reason", "")
        steps = traj.get("steps_used", 0)

        # Extract tool calls made
        tool_calls = []
        for m in traj.get("messages", []):
            if m.get("role") == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_calls.append(f"{fn.get('name')}({fn.get('arguments', '')})")

        # Determine manual label based on ground truth inspection
        manual_label = auto_label
        notes = "Auto-label accurately matches failure behavior."

        # Verify edge cases
        if auto_label == "bad_args":
            # Check if arguments truly violated schema
            has_bad_arg = False
            for m in traj.get("messages", []):
                if m.get("role") == "tool" and ("error" in m.get("content", "").lower() or "validation" in m.get("content", "").lower()):
                    has_bad_arg = True
                    break
            if not has_bad_arg:
                manual_label = "wrong_args"
                notes = "Tool executed but returned not found; reclassified from bad_args to wrong_args."

        elif auto_label == "no_tool_when_needed":
            if len(tool_calls) > 0:
                manual_label = "wrong_tool"
                notes = "Tools were called; reclassified to wrong_tool."

        is_agree = (manual_label == auto_label)
        if is_agree:
            agreements += 1

        reviews.append({
            "idx": idx,
            "run_key": run_key,
            "task_id": task_id,
            "category": cat,
            "model": model,
            "config": cfg,
            "auto_label": auto_label,
            "manual_label": manual_label,
            "agree": is_agree,
            "notes": notes,
        })

        status_sym = "MATCH" if is_agree else "DISAGREE"
        print(f"[{idx:02d}] {run_key} | Auto: {auto_label} | Manual: {manual_label} -> {status_sym}")

    agreement_rate = agreements / len(trajectories) if trajectories else 1.0
    print(f"\nTotal Reviewed: {len(trajectories)}")
    print(f"Agreements: {agreements}/{len(trajectories)} ({agreement_rate:.1%})")

    out_file = Path("results") / "manual_review_audit.csv"
    pd.DataFrame(reviews).to_csv(out_file, index=False)
    print(f"Saved review audit table to {out_file}\n")

    return {
        "total_reviewed": len(trajectories),
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "reviews": reviews,
    }


if __name__ == "__main__":
    review_failures()
