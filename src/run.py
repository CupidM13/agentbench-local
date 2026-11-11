"""Evaluation runner CLI for AgentBench-Local with resumability,
detailed JSONL trajectory logging, and cooldown management.
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm import tqdm

from src.agent import AgentRunner
from src.llm import LLMClient
from src.tasks import Task, load_tasks


def get_hardware_info() -> Dict[str, str]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


def load_existing_run_keys(run_file: Path) -> Set[str]:
    """Reads all completed run_keys from a JSONL run file."""
    keys: Set[str] = set()
    if not run_file.exists():
        return keys
    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "run_key" in record:
                    keys.add(record["run_key"])
            except Exception:
                continue
    return keys


def append_trajectory(run_file: Path, trajectory_dict: Dict):
    run_file.parent.mkdir(parents=True, exist_ok=True)
    with open(run_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trajectory_dict) + "\n")


def run_evaluation(
    models: List[str],
    configs: List[str],
    split: str = "dev",
    repeats: int = 1,
    limit: Optional[int] = None,
    cooldown_sec: float = 0.0,
    base_url: str = "http://localhost:11434/v1",
    runs_dir: str = "runs",
):
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    client = LLMClient(base_url=base_url)
    runner = AgentRunner(client=client)
    all_tasks = load_tasks()

    if split != "all":
        tasks = [t for t in all_tasks if t.split == split]
    else:
        tasks = all_tasks

    if limit is not None:
        tasks = tasks[:limit]

    hw_info = get_hardware_info()

    print(f"==================================================")
    print(f"AgentBench-Local Evaluator")
    print(f"Models: {models}")
    print(f"Configs: {configs}")
    print(f"Split: {split} ({len(tasks)} tasks)")
    print(f"Repeats: {repeats}")
    print(f"Hardware: {hw_info['os']} ({hw_info['architecture']})")
    print(f"==================================================")

    for m_idx, model in enumerate(models):
        if m_idx > 0 and cooldown_sec > 0:
            print(f"Cooling down for {cooldown_sec} seconds before next model...")
            time.sleep(cooldown_sec)

        meta = client.get_ollama_metadata(model)
        meta.update(hw_info)
        protocol = client.probe_protocol(model)
        print(f"\n[Model: {model}] Protocol: {protocol} | Param: {meta.get('parameter_size')} | Quant: {meta.get('quantization')}")

        for config in configs:
            run_file = runs_path / f"{model.replace(':', '_')}_{config}_{split}.jsonl"
            existing_keys = load_existing_run_keys(run_file)
            print(f"\n--- Running Config: {config} on {split} (Existing runs: {len(existing_keys)}) ---")

            # Prepare job list
            jobs = []
            for rep in range(repeats):
                for task in tasks:
                    run_key = f"{task.id}_{model.replace(':', '_')}_{config}_{rep}"
                    if run_key not in existing_keys:
                        jobs.append((task, rep, run_key))

            if not jobs:
                print(f"All {len(tasks) * repeats} runs already completed in {run_file.name}. Skipping.")
                continue

            passed_count = 0
            total_count = 0

            with tqdm(total=len(jobs), desc=f"{model[:15]} {config}") as pbar:
                for task, rep, run_key in jobs:
                    try:
                        traj = runner.run_task(
                            task=task,
                            model=model,
                            config=config,
                            protocol=protocol,
                            repeat=rep,
                            model_metadata=meta,
                        )
                        traj_dict = traj.to_dict()
                        append_trajectory(run_file, traj_dict)
                        existing_keys.add(run_key)

                        if traj.grader_result.get("passed"):
                            passed_count += 1
                        total_count += 1
                        pbar.set_postfix({"PassRate": f"{passed_count / total_count:.1%}"})
                    except Exception as e:
                        print(f"\nError running task {task.id}: {e}")
                    finally:
                        pbar.update(1)

            print(f"Completed {config}: {passed_count}/{total_count} passed ({passed_count / max(1, total_count):.1%})")


def main():
    parser = argparse.ArgumentParser(description="AgentBench-Local Evaluator CLI")
    parser.add_argument("--models", nargs="+", default=["llama3.2:latest"], help="Models to benchmark")
    parser.add_argument("--configs", nargs="+", default=["A_baseline"], help="Hardening configs: A_baseline B_validate_retry C_guards")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="Task split")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--cooldown", type=float, default=0.0, help="Cooldown seconds between models")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama base URL")
    parser.add_argument("--runs-dir", default="runs", help="Output directory for JSONL logs")

    args = parser.parse_args()
    run_evaluation(
        models=args.models,
        configs=args.configs,
        split=args.split,
        repeats=args.repeats,
        limit=args.limit,
        cooldown_sec=args.cooldown,
        base_url=args.base_url,
        runs_dir=args.runs_dir,
    )


if __name__ == "__main__":
    main()
