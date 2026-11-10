"""Failure taxonomy classifier for AgentBench-Local with strict priority ordering."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.tools import TOOLS_REGISTRY


TAXONOMY_CLASSES = [
    "format_error",
    "hallucinated_tool",
    "bad_args",
    "wrong_tool",
    "wrong_args",
    "no_tool_when_needed",
    "unnecessary_tool",
    "loop",
    "premature_stop",
    "synthesis_error",
    "fabricated_unanswerable",
    "timeout",
]


def classify_failure(
    task: Dict[str, Any],
    passed: bool,
    steps_used: int,
    max_steps: int,
    tool_calls: List[Dict[str, Any]],
    tool_observations: List[Dict[str, Any]],
    raw_responses: List[str],
    timed_out: bool = False,
) -> Optional[str]:
    """Classifies a failed agent trajectory according to the 12-class hierarchy.
    Returns None if the task passed.
    """
    if passed:
        return None

    if timed_out:
        return "timeout"

    category = task.get("category", "")
    expected_tools = task.get("expected_tools", [])

    # 1. format_error: unparseable output or invalid JSON
    for obs in tool_observations:
        if obs.get("code") in ("UNPARSEABLE_JSON", "MALFORMED_OUTPUT", "JSON_DECODE_ERROR"):
            return "format_error"
        if "error" in obs and ("json" in str(obs["error"]).lower() or "unparseable" in str(obs["error"]).lower()):
            return "format_error"

    # 2. hallucinated_tool: called a tool that does not exist in registry
    for tc in tool_calls:
        tool_name = tc.get("tool")
        if tool_name not in TOOLS_REGISTRY:
            return "hallucinated_tool"

    # 3. bad_args: schema-invalid arguments (pydantic or type error)
    for obs in tool_observations:
        code = obs.get("code")
        if code in ("VALIDATION_ERROR", "TYPE_ERROR", "MISSING_FIELD", "INVALID_ACTION"):
            return "bad_args"

    # 4. unnecessary_tool: used a tool on a no-tool-needed task
    if category == "no_tool_needed" and len(tool_calls) > 0:
        return "unnecessary_tool"

    # 5. no_tool_when_needed: answered from imagination when tool was required
    if category != "no_tool_needed" and len(tool_calls) == 0:
        return "no_tool_when_needed"

    # 6. fabricated_unanswerable: invented an answer on an unanswerable task
    if category == "unanswerable":
        return "fabricated_unanswerable"

    # 7. wrong_tool: valid call, but wrong tool for the task (e.g. used calendar for math)
    if expected_tools:
        tools_called = [tc.get("tool") for tc in tool_calls]
        if tools_called and not any(t in expected_tools for t in tools_called):
            return "wrong_tool"

    # 8. wrong_args: valid schema, wrong values (e.g. wrong file name or wrong date)
    for obs in tool_observations:
        code = obs.get("code")
        if code in ("NOT_FOUND", "SYNTAX_ERROR", "ZERO_DIVISION"):
            return "wrong_args"

    # 9. loop: repeated identical calls or hit max steps limit
    if steps_used >= max_steps:
        return "loop"
    if len(tool_calls) >= 2:
        for i in range(len(tool_calls) - 1):
            curr = tool_calls[i]
            nxt = tool_calls[i + 1]
            if curr.get("tool") == nxt.get("tool") and curr.get("args") == nxt.get("args"):
                return "loop"

    # 10. premature_stop: stopped before executing expected multi-step dependencies
    if category in ("multi_step", "independent_multi") and len(tool_calls) < len(expected_tools):
        return "premature_stop"

    # 11. synthesis_error: tool executed without error, but final answer was graded incorrect
    has_valid_tool_results = any(
        isinstance(obs, dict) and "error" not in obs for obs in tool_observations
    )
    if has_valid_tool_results:
        return "synthesis_error"

    # Default fallback
    return "synthesis_error"
