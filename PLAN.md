# AgentBench-Local: Engineering Plan

A reliability benchmark and hardening layer for tool-calling agents running on small local LLMs.

---

## 1. System Architecture & Objectives

- **Target:** Small local LLMs (1B to 14B) running via Ollama OpenAI-compatible endpoint (`http://localhost:11434/v1`).
- **Core Dilemma:** Small models fail at tool use due to schema non-compliance, hallucinated tool names, repetition loops, syntax errors, and context degradation.
- **Solution:** 
  1. Dual-protocol execution layer (`native` function calling vs strict `json_text` with think-block stripping).
  2. Multi-tier hardening pipeline:
     - `A_baseline`: Naive execution, step limit 8, unguided error strings.
     - `B_validate_retry`: Pre-execution Pydantic schema validation, targeted error feedback, up to 2 retries per step.
     - `C_guards`: Loop detection (2x identical call threshold + nudge), repeated-error cutoff, forced-answer final step prompt.
     - `D_selfcheck` (optional exploratory layer).
  3. Deterministic evaluation harness:
     - 60 auto-graded tasks (20 Dev / 40 Test frozen) across 7 categories.
     - Programmatic ground-truth verification of all 60 tasks.
     - Automated failure taxonomy (12 classes with priority ranking) + manual review calibration.
     - Paired-bootstrap 95% confidence intervals and Wilson score intervals.

---

## 2. Directory Layout

```
agentbench-local/
  README.md
  PLAN.md
  pyproject.toml
  .gitignore
  src/
    __init__.py
    agent.py            # Agent loop (A_baseline, B_validate_retry, C_guards)
    tools.py            # 4 tools: calculator, search_files/read_file, calendar, run_python
    llm.py              # LLM client, probe, native/json_text protocols, think-block stripper
    tasks.py            # Task loader, fixture manager, deterministic graders
    taxonomy.py         # 12-class failure classifier with strict priority
    run.py              # Resumable evaluation runner CLI
    report.py           # Statistical evaluation, bootstrap CI, Wilson intervals, charts
  data/
    sandbox/            # 15 synthetic sandbox files (csv, md, txt, json)
    tasks.yaml          # 60 task specs (20 dev, 40 test)
  runs/                 # JSONL trajectory logs (one per model+config)
  results/              # Metrics CSVs, summary tables, generated PNG charts
  notebooks/
    kaggle_run.ipynb    # Kaggle execution notebook for 9B/14B models
  tests/
    test_tools.py       # Unit tests for 4 deterministic tools
    test_graders.py     # Unit tests verifying graders pass good answers & fail bad ones
    test_tasks.py       # Verification script solving all 60 tasks programmatically
    test_agent.py       # Unit tests for agent loop and hardening guards
```

---

## 3. Phased Execution Roadmap

### Phase 1: Setup & Environment Validation (Checkpoint 1)
- Verify Python 3.13 `.venv` and dependencies (`openai`, `pydantic`, `pyyaml`, `pandas`, `matplotlib`, `pytest`, `tqdm`).
- Create `pyproject.toml` and `.gitignore`.
- Probe Ollama locally and list models. Recommend `ollama pull` commands for tool-capable models (e.g., `llama3.2:1b`, `qwen2.5:3b`, `qwen2.5:7b`).
- Implement `src/llm.py` with model probe and protocol dispatch (`native` vs `json_text`).
- Commit Phase 1.

### Phase 2: Tools & Synthetic Sandbox (Checkpoint 2)
- Implement `src/tools.py`:
  - `calculator(expression: str)`: safe AST-based arithmetic (no `eval`).
  - `search_files(query: str, max_results: int = 5)` and `read_file(path: str)` against `data/sandbox/`.
  - `calendar(action: "create"|"list"|"delete", ...)`: in-memory stateful calendar with fixture resets.
  - `run_python(code: str)`: subprocess runner with 5s timeout, restricted sandbox dir, captured output.
- Generate realistic, diverse sandbox files in `data/sandbox/` (15 files: financials, roster, sprint logs, system metrics, specs).
- Write comprehensive `tests/test_tools.py`.
- Commit Phase 2.

### Phase 3: Task Set & Deterministic Graders (Checkpoint 3)
- Author `data/tasks.yaml` with exactly 60 tasks (20 dev, 40 test):
  - 7 categories: `single_tool`, `multi_step`, `independent_multi`, `arg_precision`, `no_tool_needed`, `unanswerable`, `stateful`.
- Implement `src/tasks.py` with deterministic graders (numeric tolerance, normalized string, regex, set match, calendar state check).
- Write `tests/test_graders.py` proving each grader passes known-good and rejects known-bad answers.
- Write `tests/test_tasks.py` programmatically solving every task to prove ground truth correctness.
- Commit Phase 3.

### Phase 4: Agent Loop, Hardening Layers & Logging (Checkpoint 4)
- Implement `src/agent.py`:
  - `A_baseline`: standard prompt, step limit 8, raw tool output.
  - `B_validate_retry`: Pydantic schema validation before invocation, structured feedback on validation failure, 2 retries per step.
  - `C_guards`: loop detector (identical calls >= 2), error cutoff, forced answer on final step.
- Implement JSONL logging with exact fields (`run_key`, latency, tokens, messages, failure labels).
- Test resumability. Smoke test on 5 dev tasks.
- Commit Phase 4.

### Phase 5: Failure Taxonomy & Dev Calibration (Checkpoint 5)
- Implement `src/taxonomy.py`:
  - Strict priority order: `format_error`, `hallucinated_tool`, `bad_args`, `wrong_tool`, `wrong_args`, `no_tool_needed`, `unnecessary_tool`, `loop`, `premature_stop`, `synthesis_error`, `fabricated_unanswerable`, `timeout`.
- Run Dev split on `llama3.2:latest` (and others available).
- Inspect dev failures to tune layer B and C parameters. Freeze B and C.
- Commit Phase 5.

### Phase 6: Full Evaluation Runs (Laptop & Kaggle) (Checkpoint 6)
- Laptop: run small models (`llama3.2:3b`, `llama3.2:1b`, etc.) across configs A, B, C on test split (3 repeats). Add cooldown flag.
- Notebook: build `notebooks/kaggle_run.ipynb` for heavy models (`qwen3.5:9b`, `deepseek-r1:14b`).
- Ingest and merge all logs into `runs/`.
- Commit Phase 6.

### Phase 7: Statistical Analysis & Reporting (Checkpoint 7)
- Implement `src/report.py`:
  - Success rates with Wilson 95% score intervals.
  - Paired bootstrap (10,000 resamples) for deltas (A vs C, A vs B) with 95% CI.
  - Latency, token usage, tool validity rate, per-category breakdown.
  - 4 publication-quality PNG charts:
    1. Success by model grouped by config.
    2. Stacked failure taxonomy distribution.
    3. Success by category.
    4. Success vs Latency scatter.
  - Generate CSVs and Markdown tables.
  - Conduct manual review of 20 random failures and report agreement rate.
- Commit Phase 7.

### Phase 8: Final Polish & Portfolio README (Checkpoint 8)
- Complete comprehensive `README.md` containing all real numbers, architecture diagrams, CV claims, failure analysis, and reproduction commands.
- Final test verification and git commit.
