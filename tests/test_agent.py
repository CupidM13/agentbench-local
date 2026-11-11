"""Unit tests for agent loop, hardening layers, protocol parsing, and failure taxonomy."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.agent import AgentRunner
from src.llm import LLMResponse, ToolCallAction, extract_json_object, strip_think_blocks
from src.tasks import Task
from src.taxonomy import classify_failure


class _MockLLMClient:
    """Deterministic scripted LLM client for testing agent hardening configs."""

    def __init__(self, responses: List[LLMResponse], protocol: str = "native") -> None:
        self.responses = list(responses)
        self.protocol = protocol
        self.calls: List[List[Dict[str, Any]]] = []

    def probe_protocol(self, model: str, force: Optional[str] = None) -> str:
        return force or self.protocol

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        protocol: str = "native",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            return LLMResponse(final_answer="Done", latency_sec=0.01)
        return self.responses.pop(0)


def _math_task() -> Task:
    return Task(
        id="mock_math",
        split="dev",
        category="single_tool",
        prompt="Compute 6 * 7.",
        fixture=[],
        grader={"type": "numeric", "value": 42, "tolerance": 0.01},
        expected_tools=["calculator"],
    )


def test_think_block_stripping_and_json_extraction() -> None:
    raw = "<think>First think step.\nCompute 2+2.</think>\n```json\n{\"tool\": \"calculator\", \"args\": {\"expression\": \"2+2\"}}\n```"
    cleaned, thoughts = strip_think_blocks(raw)
    assert "First think step." in (thoughts or "")
    assert "<think>" not in cleaned
    parsed = extract_json_object(cleaned)
    assert parsed == {"tool": "calculator", "args": {"expression": "2+2"}}


def test_validation_retry_recovers_from_bad_args() -> None:
    # Step 1: invalid call (missing expression), Retry 1: valid call, Step 2: final answer
    responses = [
        LLMResponse(
            tool_calls=[ToolCallAction(tool="calculator", args={"expr": "6 * 7"}, call_id="c1")],
            prompt_tokens=10,
            completion_tokens=5,
            latency_sec=0.05,
        ),
        LLMResponse(
            tool_calls=[ToolCallAction(tool="calculator", args={"expression": "6 * 7"}, call_id="c1_retry")],
            prompt_tokens=12,
            completion_tokens=6,
            latency_sec=0.05,
        ),
        LLMResponse(
            final_answer="The answer is 42.",
            prompt_tokens=15,
            completion_tokens=5,
            latency_sec=0.05,
        ),
    ]
    client = _MockLLMClient(responses)
    runner = AgentRunner(client=client, max_steps=8)
    traj = runner.run_task(_math_task(), model="mock-model", config="B_validate_retry")

    assert traj.grader_result["passed"] is True
    assert traj.failure_label is None
    assert len(client.calls) == 3
    # Verify structured validation error was fed back on retry call
    retry_prompt = client.calls[1][-1]["content"]
    assert "validation" in retry_prompt.lower()
    assert "expression" in retry_prompt


def test_baseline_does_not_retry_bad_args() -> None:
    responses = [
        LLMResponse(
            tool_calls=[ToolCallAction(tool="calculator", args={"expr": "6 * 7"}, call_id="c1")],
            latency_sec=0.05,
        ),
        LLMResponse(
            final_answer="I could not compute it.",
            latency_sec=0.05,
        ),
    ]
    client = _MockLLMClient(responses)
    runner = AgentRunner(client=client, max_steps=8)
    traj = runner.run_task(_math_task(), model="mock-model", config="A_baseline")

    assert traj.grader_result["passed"] is False
    assert traj.failure_label == "bad_args"
    assert len(client.calls) == 2


def test_loop_guard_injects_nudge_on_identical_repeat() -> None:
    responses = [
        LLMResponse(tool_calls=[ToolCallAction(tool="calculator", args={"expression": "6 * 7"}, call_id="c1")]),
        LLMResponse(tool_calls=[ToolCallAction(tool="calculator", args={"expression": "6 * 7"}, call_id="c2")]),
        LLMResponse(final_answer="The result is 42."),
    ]
    client = _MockLLMClient(responses)
    runner = AgentRunner(client=client, max_steps=8)
    traj = runner.run_task(_math_task(), model="mock-model", config="C_guards")

    assert traj.grader_result["passed"] is True
    tool_msgs = [m["content"] for m in traj.messages if m.get("role") == "tool"]
    assert any("Loop detected" in msg for msg in tool_msgs)


def test_system_prompt_hash_consistent_across_configs() -> None:
    runner = AgentRunner(client=_MockLLMClient([]), max_steps=8)
    hashes = {
        runner.run_task(_math_task(), model="mock", config=cfg).system_prompt_hash
        for cfg in ("A_baseline", "B_validate_retry", "C_guards")
    }
    assert len(hashes) == 1


def test_taxonomy_priority_order() -> None:
    task = {"category": "multi_step", "expected_tools": ["search_files", "calculator"]}
    assert classify_failure(task, False, 1, 8, [], [], [], timed_out=True) == "timeout"
    assert classify_failure(task, False, 1, 8, [{"tool": "nonexistent", "args": {}}], [{}], []) == "hallucinated_tool"
    assert classify_failure(task, False, 1, 8, [{"tool": "calculator", "args": {}}], [{"code": "VALIDATION_ERROR"}], []) == "bad_args"
    assert classify_failure({"category": "no_tool_needed", "expected_tools": []}, False, 1, 8, [{"tool": "calculator", "args": {}}], [{}], []) == "unnecessary_tool"
    assert classify_failure(task, False, 1, 8, [], [], []) == "no_tool_when_needed"
    assert classify_failure({"category": "unanswerable", "expected_tools": ["search_files"]}, False, 2, 8, [{"tool": "search_files", "args": {"query": "x"}}], [{"results": []}], []) == "fabricated_unanswerable"
    assert classify_failure(task, False, 2, 8, [{"tool": "calendar", "args": {"action": "list"}}], [{"events": []}], []) == "wrong_tool"
    assert classify_failure(task, False, 8, 8, [{"tool": "calculator", "args": {"expression": "1+1"}}], [{"result": 2}], []) == "loop"
    assert classify_failure(task, False, 2, 8, [{"tool": "calculator", "args": {"expression": "1+1"}}], [{"result": 2}], []) == "premature_stop"
    assert classify_failure({"category": "single_tool", "expected_tools": ["calculator"]}, False, 2, 8, [{"tool": "calculator", "args": {"expression": "1+1"}}], [{"result": 2}], []) == "synthesis_error"


def test_json_text_format_retry_and_thought_logging() -> None:
    responses = [
        LLMResponse(
            content='{"calculator": "6 * 7"}',
            thought="Let me call calculator.",
            final_answer='{"calculator": "6 * 7"}',
            parse_error="JSON object must contain either 'tool' and 'args', or 'final_answer'.",
        ),
        LLMResponse(
            thought="Fixed schema.",
            tool_calls=[ToolCallAction(tool="calculator", args={"expression": "6 * 7"})],
        ),
        LLMResponse(
            thought="Got result.",
            final_answer="The result is 42.",
        ),
    ]
    client = _MockLLMClient(responses, protocol="json_text")
    runner = AgentRunner(client=client, max_steps=8)
    traj = runner.run_task(_math_task(), model="mock-r1", config="B_validate_retry")

    assert traj.grader_result["passed"] is True
    assert traj.thoughts == ["Let me call calculator.", "Fixed schema.", "Got result."]
    assert len(client.calls) == 3

    # Under A_baseline, format error is not retried and is labeled format_error
    baseline_client = _MockLLMClient([
        LLMResponse(
            content='{"calculator": "6 * 7"}',
            thought="Thinking...",
            final_answer='{"calculator": "6 * 7"}',
            parse_error="Invalid JSON schema",
        )
    ], protocol="json_text")
    traj_a = AgentRunner(client=baseline_client, max_steps=8).run_task(_math_task(), model="mock-r1", config="A_baseline")
    assert traj_a.grader_result["passed"] is False
    assert traj_a.failure_label == "format_error"
    assert traj_a.thoughts == ["Thinking..."]
