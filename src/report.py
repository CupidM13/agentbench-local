"""Evaluation and statistical reporting suite for AgentBench-Local.
Computes Wilson 95% confidence intervals, paired-bootstrap hypothesis testing (10,000 resamples),
generates 4 publication-quality PNG figures, CSV summaries, and Markdown tables.
"""

from __future__ import annotations
import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tools import TOOL_SCHEMAS_PYDANTIC


# Set publication style for matplotlib
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculates Wilson score interval for binomial proportion.
    Returns (center, lower, upper).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% normal quantile
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))
    return p, max(0.0, center - margin), min(1.0, center + margin)


def paired_bootstrap(
    tasks_a_scores: Dict[str, float],
    tasks_b_scores: Dict[str, float],
    n_resamples: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Performs a paired bootstrap test over tasks between config A and config B.
    Returns delta, 95% CI [lower, upper], and two-tailed p-value.
    """
    common_tasks = sorted(set(tasks_a_scores.keys()).intersection(tasks_b_scores.keys()))
    n = len(common_tasks)
    if n == 0:
        return {"delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}

    a_vals = np.array([tasks_a_scores[t] for t in common_tasks])
    b_vals = np.array([tasks_b_scores[t] for t in common_tasks])
    deltas = b_vals - a_vals
    observed_delta = float(np.mean(deltas))

    rng = np.random.default_rng(random_seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_deltas = np.mean(deltas[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    # P-value against null hypothesis (delta == 0)
    # Centered bootstrap distribution under null
    centered_boot = boot_deltas - observed_delta
    p_val = float(np.mean(np.abs(centered_boot) >= np.abs(observed_delta)))
    significant = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    return {
        "delta": round(observed_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "significant": significant,
    }


def _count_tool_validity(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Counts (valid_calls, total_calls) across native and json_text message logs."""
    total_calls = 0
    valid_calls = 0
    invalid_codes = {"VALIDATION_ERROR", "TYPE_ERROR", "UNKNOWN_TOOL", "MISSING_FIELD", "INVALID_ACTION"}

    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        parsed_calls: List[Tuple[Optional[str], Any]] = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = None
                parsed_calls.append((name, args))
        elif isinstance(msg.get("content"), str) and msg["content"].strip().startswith("{"):
            try:
                obj = json.loads(msg["content"])
                if isinstance(obj, dict) and "tool" in obj:
                    parsed_calls.append((obj.get("tool"), obj.get("args", {})))
            except Exception:
                pass

        for name, args in parsed_calls:
            total_calls += 1
            if name not in TOOL_SCHEMAS_PYDANTIC or not isinstance(args, dict):
                continue
            try:
                TOOL_SCHEMAS_PYDANTIC[name](**args)
            except Exception:
                continue

            obs_err = False
            if idx + 1 < len(messages):
                nxt = messages[idx + 1]
                content = nxt.get("content", "")
                if nxt.get("role") == "tool" or (nxt.get("role") == "user" and content.startswith("Observation: ")):
                    payload = content[len("Observation: "):] if content.startswith("Observation: ") else content
                    try:
                        obs = json.loads(payload)
                        if isinstance(obs, dict) and obs.get("code") in invalid_codes:
                            obs_err = True
                    except Exception:
                        pass
            if not obs_err:
                valid_calls += 1

    return valid_calls, total_calls


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table without requiring tabulate."""
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_all_runs(runs_dir: str = "runs") -> pd.DataFrame:
    """Reads all JSONL trajectory files from runs directory into a DataFrame."""
    records: List[Dict[str, Any]] = []
    files = glob.glob(f"{runs_dir}/*.jsonl")
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    passed = rec.get("grader_result", {}).get("passed", False)
                    score = rec.get("grader_result", {}).get("score", 1.0 if passed else 0.0)
                    failure_label = rec.get("failure_label")
                    valid_tc, total_tc = _count_tool_validity(rec.get("messages", []))
                    records.append({
                        "run_key": rec.get("run_key"),
                        "model": rec.get("model"),
                        "protocol": rec.get("protocol"),
                        "config": rec.get("config"),
                        "split": rec.get("split"),
                        "task_id": rec.get("task_id"),
                        "category": rec.get("category"),
                        "repeat": rec.get("repeat", 0),
                        "passed": bool(passed),
                        "score": float(score),
                        "steps_used": rec.get("steps_used", 0),
                        "tool_calls_valid": valid_tc,
                        "tool_calls_total": total_tc,
                        "total_latency_sec": rec.get("total_latency_sec", 0.0),
                        "prompt_tokens": rec.get("prompt_tokens", 0),
                        "completion_tokens": rec.get("completion_tokens", 0),
                        "total_tokens": rec.get("prompt_tokens", 0) + rec.get("completion_tokens", 0),
                        "failure_label": failure_label if failure_label else ("pass" if passed else "unknown"),
                    })
                except Exception:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["run_key"], keep="last")
    return df


def generate_plots(df: pd.DataFrame, out_dir: Path):
    """Generates the 4 required publication figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    test_df = df[df["split"] == "test"]
    if test_df.empty:
        test_df = df  # fallback if analyzing dev

    # -------------------------------------------------------------
    # Plot 1: Success Rate by Model, Grouped by Config
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    models = sorted(test_df["model"].unique())
    configs = [c for c in ["A_baseline", "B_validate_retry", "C_guards"] if c in test_df["config"].unique()]
    x = np.arange(len(models))
    width = 0.25 if len(configs) >= 3 else 0.35

    colors = {"A_baseline": "#4A5568", "B_validate_retry": "#3182CE", "C_guards": "#38A169"}

    for i, cfg in enumerate(configs):
        cfg_df = test_df[test_df["config"] == cfg]
        rates = []
        err_low = []
        err_high = []
        for m in models:
            sub = cfg_df[cfg_df["model"] == m]
            if len(sub) > 0:
                p, lo, hi = wilson_interval(sub["passed"].sum(), len(sub))
                rates.append(p * 100)
                err_low.append((p - lo) * 100)
                err_high.append((hi - p) * 100)
            else:
                rates.append(0)
                err_low.append(0)
                err_high.append(0)

        offset = (i - (len(configs) - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            rates,
            width,
            yerr=[err_low, err_high],
            capsize=4,
            label=cfg,
            color=colors.get(cfg, "#718096"),
            edgecolor="black",
            alpha=0.9,
        )

        # Label percentages on bars
        for rect, rate in zip(rects, rates):
            if rate > 0:
                ax.annotate(
                    f"{rate:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Task Success Rate (%) (Wilson 95% CI)")
    ax.set_title("AgentBench-Local: Tool-Use Success Rate Across Hardening Layers")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(out_dir / "chart1_success_by_model_and_config.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Stacked Failure-Type Breakdown (Baseline vs Best Config)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fail_df = test_df[~test_df["passed"]]
    if not fail_df.empty:
        pivot = pd.crosstab(fail_df["config"], fail_df["failure_label"], normalize="index") * 100
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="white")
        ax.set_ylabel("Share of Failures (%)")
        ax.set_title("Failure Taxonomy Distribution (Baseline vs Hardened)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Failure Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "chart2_failure_taxonomy_distribution.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # Plot 3: Success Rate by Category
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_pivot = test_df.groupby(["category", "config"])["passed"].mean().unstack() * 100
    cat_pivot.plot(kind="bar", ax=ax, colormap="viridis", width=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Task Category")
    ax.set_title("Task Success Rate by Category Across Hardening Layers")
    ax.set_xticklabels(cat_pivot.index, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(title="Config")
    plt.tight_layout()
    plt.savefig(out_dir / "chart3_success_by_category.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 4: Success vs Average Latency per Model
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = test_df.groupby(["model", "config"]).agg(
        success_pct=("passed", lambda s: s.mean() * 100),
        avg_latency=("total_latency_sec", "mean"),
    ).reset_index()

    for cfg in configs:
        sub = grouped[grouped["config"] == cfg]
        ax.scatter(
            sub["avg_latency"],
            sub["success_pct"],
            s=120,
            label=cfg,
            color=colors.get(cfg, "#718096"),
            alpha=0.9,
            edgecolors="black",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['model'].split(':')[0]}",
                xy=(row["avg_latency"], row["success_pct"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Average Latency per Task (seconds)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Efficiency Frontier: Task Success Rate vs Latency")
    ax.set_ylim(0, 105)
    ax.legend(title="Config")
    plt.tight_layout()
    plt.savefig(out_dir / "chart4_success_vs_latency.png", dpi=300)
    plt.close()


def generate_summary_tables(df: pd.DataFrame, out_dir: Path) -> str:
    """Generates markdown tables and summary CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return "No data found."

    test_df = df[df["split"] == "test"]
    if test_df.empty:
        test_df = df

    # Summary table per model and config
    rows = []
    models = sorted(test_df["model"].unique())
    configs = sorted(test_df["config"].unique())

    for m in models:
        for c in configs:
            sub = test_df[(test_df["model"] == m) & (test_df["config"] == c)]
            if len(sub) == 0:
                continue
            total = len(sub)
            passed = int(sub["passed"].sum())
            p, lo, hi = wilson_interval(passed, total)
            avg_steps = sub["steps_used"].mean()
            avg_lat = sub["total_latency_sec"].mean()
            avg_tok = sub["total_tokens"].mean()
            tc_valid = int(sub["tool_calls_valid"].sum())
            tc_total = int(sub["tool_calls_total"].sum())
            tc_rate_str = f"{(tc_valid / tc_total) * 100:.1f}%" if tc_total > 0 else "N/A"
            rows.append({
                "Model": m,
                "Config": c,
                "Runs": total,
                "Passed": passed,
                "Success Rate": f"{p * 100:.1f}%",
                "95% CI (Wilson)": f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
                "Tool Validity": tc_rate_str,
                "Avg Steps": f"{avg_steps:.2f}",
                "Avg Latency (s)": f"{avg_lat:.2f}",
                "Avg Tokens": f"{avg_tok:.0f}",
                "_raw_p": p,
                "_raw_lo": lo,
                "_raw_hi": hi,
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    # Paired Bootstrap comparisons (A vs B, A vs C)
    bootstrap_rows = []
    for m in models:
        sub_m = test_df[test_df["model"] == m]
        a_scores = sub_m[sub_m["config"] == "A_baseline"].groupby("task_id")["passed"].mean().to_dict()
        b_scores = sub_m[sub_m["config"] == "B_validate_retry"].groupby("task_id")["passed"].mean().to_dict()
        c_scores = sub_m[sub_m["config"] == "C_guards"].groupby("task_id")["passed"].mean().to_dict()

        if a_scores and b_scores:
            res_ab = paired_bootstrap(a_scores, b_scores)
            bootstrap_rows.append({
                "Model": m,
                "Comparison": "A vs B (Validation)",
                "Delta (% points)": f"{res_ab['delta'] * 100:+.1f}%",
                "95% CI (Bootstrap)": f"[{res_ab['ci_lower'] * 100:+.1f}%, {res_ab['ci_upper'] * 100:+.1f}%]",
                "p-value": f"{res_ab['p_value']:.4f}",
                "Significant": "Yes" if res_ab["significant"] else "No",
            })

        if a_scores and c_scores:
            res_ac = paired_bootstrap(a_scores, c_scores)
            bootstrap_rows.append({
                "Model": m,
                "Comparison": "A vs C (Hardened)",
                "Delta (% points)": f"{res_ac['delta'] * 100:+.1f}%",
                "95% CI (Bootstrap)": f"[{res_ac['ci_lower'] * 100:+.1f}%, {res_ac['ci_upper'] * 100:+.1f}%]",
                "p-value": f"{res_ac['p_value']:.4f}",
                "Significant": "Yes" if res_ac["significant"] else "No",
            })

    boot_df = pd.DataFrame(bootstrap_rows)
    if not boot_df.empty:
        boot_df.to_csv(out_dir / "bootstrap_deltas.csv", index=False)

    # Per-category table
    cat_pivot = test_df.groupby(["category", "config"])["passed"].agg(
        success_rate=lambda s: f"{s.mean() * 100:.1f}%",
        count="count",
    ).unstack()
    cat_pivot.to_csv(out_dir / "category_metrics.csv")

    # Failure taxonomy summary table
    fail_only = test_df[~test_df["passed"]]
    if not fail_only.empty:
        tax_pivot = pd.crosstab([fail_only["model"], fail_only["config"]], fail_only["failure_label"])
        tax_pivot.to_csv(out_dir / "failure_taxonomy.csv")

    # Format Markdown Output
    clean_summary = summary_df.drop(columns=["_raw_p", "_raw_lo", "_raw_hi"])
    md_output = "### Evaluation Results: Primary Benchmark Summary\n\n"
    md_output += _df_to_markdown(clean_summary)
    md_output += "\n\n### Paired-Bootstrap Hypothesis Tests (10,000 resamples)\n\n"
    if not boot_df.empty:
        md_output += _df_to_markdown(boot_df)
    else:
        md_output += "_No comparisons available yet._"

    return md_output


def main():
    parser = argparse.ArgumentParser(description="AgentBench-Local Report Generator")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing JSONL trajectory logs")
    parser.add_argument("--out-dir", default="results", help="Directory to save CSVs and PNG charts")
    args = parser.parse_args()

    df = load_all_runs(args.runs_dir)
    if df.empty:
        print("No runs found to process.")
        return

    out_dir = Path(args.out_dir)
    generate_plots(df, out_dir)
    md_table = generate_summary_tables(df, out_dir)
    print("\n" + md_table)
    print(f"\nGenerated 4 figures and summary CSVs in {out_dir}/")


if __name__ == "__main__":
    main()
