"""Task definitions, loaders, and deterministic graders for AgentBench-Local."""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml


@dataclass
class Task:
    id: str
    split: str  # 'dev' or 'test'
    category: str
    prompt: str
    fixture: List[Dict[str, Any]] = field(default_factory=list)
    grader: Dict[str, Any] = field(default_factory=dict)
    expected_tools: List[str] = field(default_factory=list)


@dataclass
class GradeResult:
    passed: bool
    reason: str
    score: float = 0.0


def load_tasks(tasks_path: Optional[str] = None) -> List[Task]:
    """Loads tasks from data/tasks.yaml."""
    if tasks_path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "tasks.yaml"
    else:
        path = Path(tasks_path)

    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_list = yaml.safe_load(f)

    tasks: List[Task] = []
    for item in raw_list:
        tasks.append(
            Task(
                id=item["id"],
                split=item["split"],
                category=item["category"],
                prompt=item["prompt"],
                fixture=item.get("fixture", []),
                grader=item.get("grader", {}),
                expected_tools=item.get("expected_tools", []),
            )
        )
    return tasks


# =====================================================================
# Deterministic Graders
# =====================================================================

NUMERIC_EXTRACT_REGEX = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    """Finds all numerical values in text string, stripping comma thousands separators."""
    if not text:
        return []
    matches = NUMERIC_EXTRACT_REGEX.findall(text)
    numbers = []
    for m in matches:
        cleaned = m.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def grade_numeric(expected: float, tolerance: float, answer: str) -> GradeResult:
    """Checks if any extracted number in answer is within tolerance of expected."""
    numbers = _extract_numbers(answer)
    if not numbers:
        return GradeResult(
            passed=False,
            reason=f"No numbers found in answer. Expected {expected} (tol {tolerance})",
            score=0.0,
        )
    for n in numbers:
        if math.isclose(n, expected, abs_tol=tolerance, rel_tol=1e-3):
            return GradeResult(
                passed=True,
                reason=f"Found {n} matching expected {expected} within tolerance {tolerance}",
                score=1.0,
            )
    return GradeResult(
        passed=False,
        reason=f"None of extracted numbers {numbers[:5]} matched expected {expected} within tolerance {tolerance}",
        score=0.0,
    )


def grade_string_contains(
    keywords: List[str],
    answer: str,
    all_required: bool = True,
    case_sensitive: bool = False,
    forbidden: Optional[List[str]] = None,
) -> GradeResult:
    """Checks presence of required keywords and absence of forbidden strings."""
    norm_answer = answer if case_sensitive else answer.lower()

    if forbidden:
        for f in forbidden:
            target = f if case_sensitive else f.lower()
            if target in norm_answer:
                return GradeResult(
                    passed=False,
                    reason=f"Forbidden string '{f}' found in answer",
                    score=0.0,
                )

    if all_required:
        missing = []
        for kw in keywords:
            target = kw if case_sensitive else kw.lower()
            if target not in norm_answer:
                missing.append(kw)
        if missing:
            return GradeResult(
                passed=False,
                reason=f"Missing required keywords: {missing}",
                score=0.0,
            )
        return GradeResult(passed=True, reason="All required keywords present", score=1.0)
    else:
        # At least one required
        for kw in keywords:
            target = kw if case_sensitive else kw.lower()
            if target in norm_answer:
                return GradeResult(
                    passed=True,
                    reason=f"Found matching keyword: '{kw}'",
                    score=1.0,
                )
        return GradeResult(
            passed=False,
            reason=f"None of the alternative keywords {keywords} were found",
            score=0.0,
        )


def grade_regex(pattern: str, answer: str) -> GradeResult:
    """Matches regular expression pattern against answer."""
    match = re.search(pattern, answer, re.IGNORECASE)
    if match:
        return GradeResult(
            passed=True,
            reason=f"Answer matched regex pattern: '{pattern}'",
            score=1.0,
        )
    return GradeResult(
        passed=False,
        reason=f"Answer failed to match regex pattern: '{pattern}'",
        score=0.0,
    )


def grade_unanswerable(answer: str) -> GradeResult:
    """Validates that model recognized information is missing/unavailable rather than fabricating."""
    norm = answer.lower()
    positive_indicators = [
        "not found",
        "cannot be found",
        "unavailable",
        "no information",
        "does not mention",
        "not mentioned",
        "not available",
        "could not find",
        "unable to find",
        "not present",
        "no data",
        "no record",
        "cannot determine",
    ]
    found = any(ind in norm for ind in positive_indicators)
    if found:
        return GradeResult(
            passed=True,
            reason="Correctly recognized missing information in sandbox",
            score=1.0,
        )
    return GradeResult(
        passed=False,
        reason="Failed to state information was unavailable; likely fabricated an answer",
        score=0.0,
    )


def grade_no_tool(
    answer: str,
    tool_calls_made: List[Dict[str, Any]],
    expected_keywords: Optional[List[str]] = None,
) -> GradeResult:
    """Distractor task grader: calling any tool is an automatic failure."""
    if tool_calls_made and len(tool_calls_made) > 0:
        return GradeResult(
            passed=False,
            reason=f"Unnecessary tool calls made: {[tc.get('tool') for tc in tool_calls_made]}",
            score=0.0,
        )
    if expected_keywords:
        return grade_string_contains(expected_keywords, answer, all_required=False)
    return GradeResult(passed=True, reason="No tools called and answer provided", score=1.0)


def grade_calendar_state(
    expected_events: List[Dict[str, Any]],
    actual_state: List[Dict[str, Any]],
    expected_count: Optional[int] = None,
) -> GradeResult:
    """Checks final state of calendar events against expected set."""
    if expected_count is not None and len(actual_state) != expected_count:
        return GradeResult(
            passed=False,
            reason=f"Expected {expected_count} calendar events, found {len(actual_state)}",
            score=0.0,
        )

    # Check each required event specification
    for exp in expected_events:
        matched = False
        for act in actual_state:
            title_ok = (
                exp["title"].strip().lower() == act.get("title", "").strip().lower()
                if "title" in exp
                else True
            )
            date_ok = (
                exp["date"].strip() == act.get("date", "").strip()
                if "date" in exp
                else True
            )
            start_ok = (
                exp["start"].strip() == act.get("start", "").strip()
                if "start" in exp
                else True
            )
            end_ok = (
                exp["end"].strip() == act.get("end", "").strip()
                if "end" in exp
                else True
            )
            if title_ok and date_ok and start_ok and end_ok:
                matched = True
                break
        if not matched:
            return GradeResult(
                passed=False,
                reason=f"Expected event not found in final calendar state: {exp}",
                score=0.0,
            )

    return GradeResult(
        passed=True,
        reason="Final calendar state matches expected events exactly",
        score=1.0,
    )


def grade_task(
    task: Task,
    answer: str,
    calendar_state: Optional[List[Dict[str, Any]]] = None,
    tool_calls_made: Optional[List[Dict[str, Any]]] = None,
) -> GradeResult:
    """Dispatches task grading based on task.grader configuration."""
    answer = answer or ""
    cal_state = calendar_state or []
    calls = tool_calls_made or []
    grader_spec = task.grader
    grader_type = grader_spec.get("type")

    if grader_type == "numeric":
        expected = float(grader_spec["value"])
        tolerance = float(grader_spec.get("tolerance", 0.05))
        return grade_numeric(expected, tolerance, answer)

    elif grader_type == "string_contains":
        keywords = grader_spec.get("keywords", [])
        all_req = grader_spec.get("all_required", True)
        case_sens = grader_spec.get("case_sensitive", False)
        forbidden = grader_spec.get("forbidden", [])
        return grade_string_contains(keywords, answer, all_req, case_sens, forbidden)

    elif grader_type == "regex":
        pattern = grader_spec.get("pattern", "")
        return grade_regex(pattern, answer)

    elif grader_type == "unanswerable":
        return grade_unanswerable(answer)

    elif grader_type == "no_tool":
        keywords = grader_spec.get("keywords", [])
        return grade_no_tool(answer, calls, keywords)

    elif grader_type == "calendar_state":
        expected_events = grader_spec.get("expected_events", [])
        expected_count = grader_spec.get("expected_count")
        return grade_calendar_state(expected_events, cal_state, expected_count)

    elif grader_type == "composite":
        # Supports sub-graders where all must pass (e.g. numeric AND string)
        sub_graders = grader_spec.get("graders", [])
        for sub in sub_graders:
            sub_task = Task(
                id=task.id,
                split=task.split,
                category=task.category,
                prompt=task.prompt,
                grader=sub,
            )
            sub_res = grade_task(sub_task, answer, cal_state, calls)
            if not sub_res.passed:
                return sub_res
        return GradeResult(passed=True, reason="All composite checks passed", score=1.0)

    else:
        return GradeResult(
            passed=False,
            reason=f"Unknown grader type: '{grader_type}'",
            score=0.0,
        )
