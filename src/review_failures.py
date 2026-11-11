"""Failure taxonomy manual review tool.
Deterministically samples 20 failed test-split trajectories, displays their full
context (task prompt, tool calls, tool observations, final answer, grader reason),
and compares genuine human-reviewed ground-truth labels against auto_label.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.report import load_all_runs
from src.tasks import load_tasks


def sample_failed_trajectories(
    runs_dir: str = "runs",
    split: str = "test",
    seed: int = 42,
    sample_size: int = 20,
) -> List[Dict[str, Any]]:
    """Deterministically samples up to `sample_size` failed runs and loads full logs."""
    df = load_all_runs(runs_dir)
    if df.empty:
        return []

    sub = df[~df["passed"]]
    if split != "all":
        sub = sub[sub["split"] == split]
    if sub.empty:
        return []

    # Sort before sampling so order of glob files never affects the sample
    sub = sub.sort_values("run_key").reset_index(drop=True)
    sample_n = min(sample_size, len(sub))
    sampled_keys = sub.sample(n=sample_n, random_state=seed)["run_key"].tolist()

    key_set = set(sampled_keys)
    by_key: Dict[str, Dict[str, Any]] = {}
    for fpath in sorted(Path(runs_dir).glob("*.jsonl")):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rk = rec.get("run_key")
                    if rk in key_set:
                        by_key[rk] = rec
                except Exception:
                    continue

    return [by_key[k] for k in sampled_keys if k in by_key]


def dump_for_manual_review(
    runs_dir: str = "runs",
    split: str = "test",
    seed: int = 42,
    sample_size: int = 20,
) -> None:
    """Prints 20 sampled failed trajectories for human inspection."""
    trajectories = sample_failed_trajectories(runs_dir, split=split, seed=seed, sample_size=sample_size)
    tasks_by_id = {t.id: t for t in load_tasks()}

    print(f"\n=== Dumping {len(trajectories)} Failed '{split}' Trajectories (seed={seed}) ===\n")
    for idx, traj in enumerate(trajectories, start=1):
        tid = traj.get("task_id", "")
        task = tasks_by_id.get(tid)
        prompt = task.prompt if task else "<unknown>"
        expected_tools = task.expected_tools if task else []

        print(f"--- [{idx:02d}] {traj.get('run_key')} ---")
        print(f"Model: {traj.get('model')} | Config: {traj.get('config')} | Category: {traj.get('category')}")
        print(f"Expected tools: {expected_tools}")
        print(f"Prompt: {prompt}")
        print(f"Auto label: {traj.get('failure_label')} | Steps: {traj.get('steps_used')}")

        for m in traj.get("messages", [])[2:]:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    print(f"  TOOL_CALL: {fn.get('name')}({fn.get('arguments')})")
            elif role == "tool":
                print(f"  TOOL_OBS:  {m.get('content', '')[:300]}")
            elif role in ("assistant", "user"):
                print(f"  {role.upper()}: {str(m.get('content', ''))[:300]}")

        gr = traj.get("grader_result", {})
        print(f"Grader: passed={gr.get('passed')} | reason={gr.get('reason')}")
        print()


def evaluate_audit(audit_json: str, out_csv: str = "results/manual_review_audit.csv") -> Dict[str, Any]:
    """Computes agreement statistics from a manual review JSON decision list."""
    with open(audit_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    rows = []
    agreements = 0
    for item in entries:
        agree = item["manual_label"] == item["auto_label"]
        if agree:
            agreements += 1
        rows.append({
            "idx": item["idx"],
            "run_key": item["run_key"],
            "task_id": item["task_id"],
            "category": item["category"],
            "model": item["model"],
            "config": item["config"],
            "auto_label": item["auto_label"],
            "manual_label": item["manual_label"],
            "agree": agree,
            "notes": item.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    rate = agreements / len(rows) if rows else 0.0
    print(f"Manual Audit Complete: {agreements}/{len(rows)} agreements ({rate:.1%}) -> {out_csv}")
    return {"total_reviewed": len(rows), "agreements": agreements, "agreement_rate": rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual failure taxonomy audit tool")
    parser.add_argument("--dump", action="store_true", help="Dump 20 sampled test failures for human inspection")
    parser.add_argument("--audit-json", type=str, default=None, help="Path to manual annotations JSON")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    if args.audit_json:
        evaluate_audit(args.audit_json)
    else:
        dump_for_manual_review(
            runs_dir=args.runs_dir,
            split=args.split,
            seed=args.seed,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
