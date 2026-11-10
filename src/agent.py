"""Agent execution loop with multi-layer hardening (A_baseline, B_validate_retry, C_guards, D_selfcheck),
dual protocols (native / json_text), and detailed trajectory logging.
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError

from src.llm import LLMClient, LLMResponse, ToolCallAction
from src.tools import (
    OPENAI_TOOL_DEFINITIONS,
    TOOL_SCHEMAS_PYDANTIC,
    TOOLS_REGISTRY,
    calendar_reset,
    execute_tool_call,
    get_calendar_state,
)
from src.tasks import Task, GradeResult, grade_task
from src.taxonomy import classify_failure


BASE_SYSTEM_PROMPT = """You are an accurate, disciplined AI assistant equipped with specialized tools.
Always solve user requests thoroughly using the available tools when needed.
If a task does not require tools, answer directly.
If required information is not available in the tools or files, clearly state that it is unavailable rather than guessing.
"""

JSON_TEXT_INSTRUCTIONS = """
To interact with tools, you MUST reply ONLY with a valid JSON object in one of two formats:

1. To call a tool:
{"tool": "<tool_name>", "args": {<arguments>}}

2. When you have the final answer:
{"final_answer": "<your final response text>"}

Available tools:
- calculator(expression: str): Safely evaluates math expressions (supports +, -, *, /, **, %, sqrt, round, min, max, abs).
- search_files(query: str, max_results: int = 5): Searches files in sandbox for keywords.
- read_file(path: str): Reads full text of a file in the sandbox.
- calendar(action: "create"|"list"|"delete", title: str, date: "YYYY-MM-DD", start: "HH:MM", end: "HH:MM", event_id: str): Manages calendar.
- run_python(code: str): Executes Python 3 code in subprocess and returns stdout.

Do not output any explanation outside the JSON object.
"""


def compute_prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.strip().encode("utf-8")).hexdigest()[:12]


@dataclass
class AgentTrajectory:
    run_key: str
    timestamp: str
    model: str
    protocol: str
    config: str
    task_id: str
    category: str
    split: str
    repeat: int
    messages: List[Dict[str, Any]]
    steps_used: int
    step_latencies: List[float]
    prompt_tokens: int
    completion_tokens: int
    total_latency_sec: float
    final_answer: str
    calendar_final_state: List[Dict[str, Any]]
    grader_result: Dict[str, Any]
    failure_label: Optional[str]
    system_prompt_hash: str
    model_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentRunner:
    def __init__(
        self,
        client: LLMClient,
        max_steps: int = 8,
    ):
        self.client = client
        self.max_steps = max_steps

    def _build_system_prompt(self, protocol: str) -> str:
        prompt = BASE_SYSTEM_PROMPT
        if protocol == "json_text":
            prompt += "\n" + JSON_TEXT_INSTRUCTIONS
        return prompt.strip()

    def _validate_tool_args_pydantic(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Validates tool arguments using Pydantic schema."""
        if tool_name not in TOOL_SCHEMAS_PYDANTIC:
            return False, None, f"Tool '{tool_name}' does not exist in registry."
        schema_cls = TOOL_SCHEMAS_PYDANTIC[tool_name]
        try:
            validated = schema_cls(**args)
            return True, validated.model_dump(), None
        except ValidationError as e:
            err_details = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"])
                err_details.append(f"field '{loc}': {err['msg']}")
            return False, None, f"Schema validation error: {'; '.join(err_details)}"
        except Exception as e:
            return False, None, f"Invalid arguments: {e}"

    def run_task(
        self,
        task: Task,
        model: str,
        config: str = "A_baseline",
        protocol: Optional[str] = None,
        repeat: int = 0,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentTrajectory:
        """Executes a single task run with the specified hardening layer and protocol."""
        # 1. Reset fixture state
        calendar_reset(task.fixture)

        # 2. Determine protocol
        active_protocol = protocol or self.client.probe_protocol(model)
        system_prompt = self._build_system_prompt(active_protocol)
        prompt_hash = compute_prompt_hash(system_prompt)

        # 3. Initialize message history
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.prompt},
        ]

        step_latencies: List[float] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_observations: List[Dict[str, Any]] = []
        raw_responses: List[str] = []
        final_answer = ""
        timed_out = False
        start_wall_time = time.time()
        consecutive_errors = 0

        # Step loop
        step = 0
        while step < self.max_steps:
            step += 1

            # Layer C Guard: Forced final answer on last step
            if config in ("C_guards", "D_selfcheck") and step == self.max_steps:
                messages.append({
                    "role": "user",
                    "content": "Step limit reached. You must synthesize and provide your final answer now."
                })

            # Call LLM
            tools_schema = OPENAI_TOOL_DEFINITIONS if active_protocol == "native" else None
            try:
                resp = self.client.chat(
                    model=model,
                    messages=messages,
                    tools_schema=tools_schema,
                    protocol=active_protocol,
                )
            except Exception as e:
                # LLM call failure / timeout
                timed_out = True
                final_answer = f"LLM error: {e}"
                break

            step_latencies.append(resp.latency_sec)
            total_prompt_tokens += resp.prompt_tokens
            total_completion_tokens += resp.completion_tokens
            if resp.content:
                raw_responses.append(resp.content)

            # If model provided a final answer and no tool calls, terminate loop
            if resp.final_answer and not resp.tool_calls:
                final_answer = resp.final_answer
                messages.append({"role": "assistant", "content": resp.final_answer})
                break

            # If no tool calls were generated and no final answer
            if not resp.tool_calls:
                final_answer = resp.content or ""
                messages.append({"role": "assistant", "content": final_answer})
                break

            # Handle Tool Calls
            # In our setup, process the primary tool call generated in this step
            tc = resp.tool_calls[0]
            tool_name = tc.tool
            tool_args = tc.args
            call_id = tc.call_id or f"call_{step}"

            # Layer C Guard: Loop Detection (identical call repeated)
            if config in ("C_guards", "D_selfcheck") and len(all_tool_calls) >= 1:
                prev_call = all_tool_calls[-1]
                if prev_call.get("tool") == tool_name and prev_call.get("args") == tool_args:
                    loop_msg = f"Loop detected: You already called '{tool_name}' with arguments {tool_args}. Do not repeat identical calls. Use the observation already received or state your final answer."
                    if active_protocol == "native":
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                            }],
                        })
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": loop_msg})
                    else:
                        messages.append({"role": "assistant", "content": json.dumps({"tool": tool_name, "args": tool_args})})
                        messages.append({"role": "user", "content": f"Observation: {loop_msg}"})
                    all_tool_calls.append({"tool": tool_name, "args": tool_args})
                    all_tool_observations.append({"error": loop_msg, "code": "LOOP_PREVENTED"})
                    consecutive_errors += 1
                    continue

            # Layer B & C: Pre-execution Pydantic Validation & Retry
            is_valid = True
            val_error_msg = None
            if config in ("B_validate_retry", "C_guards", "D_selfcheck"):
                is_valid, clean_args, val_error_msg = self._validate_tool_args_pydantic(tool_name, tool_args)
                if not is_valid:
                    # Retry loop up to 2 attempts for layer B/C
                    retry_success = False
                    for retry_idx in range(2):
                        retry_prompt = (
                            f"Tool call failed validation: {val_error_msg}. "
                            f"Please correct the arguments and provide the valid JSON tool call or answer."
                        )
                        retry_messages = list(messages)
                        retry_messages.append({"role": "user", "content": retry_prompt})
                        try:
                            retry_resp = self.client.chat(
                                model=model,
                                messages=retry_messages,
                                tools_schema=tools_schema,
                                protocol=active_protocol,
                            )
                            step_latencies.append(retry_resp.latency_sec)
                            total_prompt_tokens += retry_resp.prompt_tokens
                            total_completion_tokens += retry_resp.completion_tokens

                            if retry_resp.tool_calls:
                                r_tc = retry_resp.tool_calls[0]
                                r_tool, r_args = r_tc.tool, r_tc.args
                                r_valid, r_clean, r_err = self._validate_tool_args_pydantic(r_tool, r_args)
                                if r_valid:
                                    tool_name = r_tool
                                    tool_args = r_clean
                                    is_valid = True
                                    retry_success = True
                                    break
                            elif retry_resp.final_answer:
                                final_answer = retry_resp.final_answer
                                retry_success = True
                                break
                        except Exception:
                            pass
                    if not retry_success and not final_answer:
                        tool_obs = {"error": val_error_msg, "code": "VALIDATION_ERROR"}
                        consecutive_errors += 1
                    else:
                        tool_obs = None
                else:
                    tool_args = clean_args
                    tool_obs = None
            else:
                # Layer A_baseline: no validation, naive execution
                tool_obs = None

            # Execute tool if not already failed by validation
            if tool_obs is None and not final_answer:
                tool_obs = execute_tool_call(tool_name, tool_args)
                if "error" in tool_obs:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

            all_tool_calls.append({"tool": tool_name, "args": tool_args})
            all_tool_observations.append(tool_obs or {})

            # Append messages based on protocol
            obs_str = json.dumps(tool_obs) if isinstance(tool_obs, dict) else str(tool_obs)
            if active_protocol == "native":
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                    }],
                })
                messages.append({"role": "tool", "tool_call_id": call_id, "content": obs_str})
            else:
                messages.append({"role": "assistant", "content": json.dumps({"tool": tool_name, "args": tool_args})})
                messages.append({"role": "user", "content": f"Observation: {obs_str}"})

            # Layer C Guard: Consecutive error cutoff
            if config in ("C_guards", "D_selfcheck") and consecutive_errors >= 3:
                messages.append({
                    "role": "user",
                    "content": "Multiple tool errors encountered. Please provide your best final answer immediately without further tool calls."
                })

        # End of step loop

        # If answer is still empty, fallback to last assistant message
        if not final_answer:
            for m in reversed(messages):
                if m.get("role") == "assistant" and m.get("content"):
                    final_answer = m["content"]
                    break

        total_latency = time.time() - start_wall_time
        final_calendar_state = get_calendar_state()

        # Grade task
        grade_res = grade_task(
            task=task,
            answer=final_answer,
            calendar_state=final_calendar_state,
            tool_calls_made=all_tool_calls,
        )

        # Classify failure taxonomy
        failure_label = classify_failure(
            task=asdict(task),
            passed=grade_res.passed,
            steps_used=step,
            max_steps=self.max_steps,
            tool_calls=all_tool_calls,
            tool_observations=all_tool_observations,
            raw_responses=raw_responses,
            timed_out=timed_out,
        )

        run_key = f"{task.id}_{model.replace(':', '_')}_{config}_{repeat}"
        iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return AgentTrajectory(
            run_key=run_key,
            timestamp=iso_timestamp,
            model=model,
            protocol=active_protocol,
            config=config,
            task_id=task.id,
            category=task.category,
            split=task.split,
            repeat=repeat,
            messages=messages,
            steps_used=step,
            step_latencies=step_latencies,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_latency_sec=round(total_latency, 3),
            final_answer=final_answer,
            calendar_final_state=final_calendar_state,
            grader_result=asdict(grade_res),
            failure_label=failure_label,
            system_prompt_hash=prompt_hash,
            model_metadata=model_metadata or {},
        )
