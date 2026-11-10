"""Unit tests verifying that every deterministic grader accepts known-good inputs
and rejects known-bad inputs across all grading categories.
"""

import pytest
from src.tasks import (
    GradeResult,
    Task,
    grade_numeric,
    grade_string_contains,
    grade_regex,
    grade_unanswerable,
    grade_no_tool,
    grade_calendar_state,
    grade_task,
)


def test_grade_numeric_good_and_bad():
    # Good
    res_good = grade_numeric(expected=4060.0, tolerance=0.1, answer="The final calculated value is 4,060.00.")
    assert res_good.passed is True
    assert res_good.score == 1.0

    # Bad (wrong number)
    res_bad = grade_numeric(expected=4060.0, tolerance=0.1, answer="The result is 3999.")
    assert res_bad.passed is False
    assert res_bad.score == 0.0

    # Bad (no numbers)
    res_no_num = grade_numeric(expected=4060.0, tolerance=0.1, answer="There are no values here.")
    assert res_no_num.passed is False


def test_grade_string_contains_good_and_bad():
    # Good: all required
    res_good = grade_string_contains(
        keywords=["clara.oswald", "company.internal"],
        answer="The email is Clara.Oswald@Company.Internal.",
        all_required=True,
    )
    assert res_good.passed is True

    # Bad: missing one keyword
    res_bad = grade_string_contains(
        keywords=["clara.oswald", "nonexistent_domain"],
        answer="The email is clara.oswald@company.internal.",
        all_required=True,
    )
    assert res_bad.passed is False

    # Forbidden keyword check
    res_forbidden = grade_string_contains(
        keywords=["Paris"],
        answer="Paris is the capital, but London is better.",
        forbidden=["London"],
    )
    assert res_forbidden.passed is False


def test_grade_regex_good_and_bad():
    pattern = r"v\d+\.\d+\.\d+"
    res_good = grade_regex(pattern, "Upgraded to v2.4.0 yesterday.")
    assert res_good.passed is True

    res_bad = grade_regex(pattern, "No version specified here.")
    assert res_bad.passed is False


def test_grade_unanswerable_good_and_bad():
    res_good_1 = grade_unanswerable("The requested marketing budget cannot be found in the sandbox files.")
    assert res_good_1.passed is True

    res_good_2 = grade_unanswerable("This information is unavailable in our documentation.")
    assert res_good_2.passed is True

    # Bad: model fabricates a specific answer
    res_bad = grade_unanswerable("The TikTok marketing budget is $250,000 for 2024.")
    assert res_bad.passed is False


def test_grade_no_tool_good_and_bad():
    # Good: no tools called, contains expected keyword
    res_good = grade_no_tool(
        answer="The capital of France is Paris.",
        tool_calls_made=[],
        expected_keywords=["Paris"],
    )
    assert res_good.passed is True

    # Bad: tool was called on distractor task
    res_bad_tool = grade_no_tool(
        answer="Paris",
        tool_calls_made=[{"tool": "search_files", "args": {"query": "capital"}}],
        expected_keywords=["Paris"],
    )
    assert res_bad_tool.passed is False
    assert "Unnecessary tool calls" in res_bad_tool.reason

    # Bad: no tools, but wrong answer
    res_bad_ans = grade_no_tool(
        answer="The capital is Berlin.",
        tool_calls_made=[],
        expected_keywords=["Paris"],
    )
    assert res_bad_ans.passed is False


def test_grade_calendar_state_good_and_bad():
    expected_events = [
        {"title": "Design Review", "date": "2024-11-04", "start": "15:00", "end": "16:00"}
    ]
    actual_state_good = [
        {"id": "evt_001", "title": "Design Review", "date": "2024-11-04", "start": "15:00", "end": "16:00"}
    ]
    res_good = grade_calendar_state(expected_events, actual_state_good, expected_count=1)
    assert res_good.passed is True

    # Bad: wrong count
    res_bad_count = grade_calendar_state(expected_events, [], expected_count=1)
    assert res_bad_count.passed is False

    # Bad: wrong time
    actual_state_wrong_time = [
        {"id": "evt_001", "title": "Design Review", "date": "2024-11-04", "start": "14:00", "end": "15:00"}
    ]
    res_bad_time = grade_calendar_state(expected_events, actual_state_wrong_time, expected_count=1)
    assert res_bad_time.passed is False


def test_grade_task_composite():
    task = Task(
        id="comp_01",
        split="dev",
        category="independent_multi",
        prompt="Test composite",
        grader={
            "type": "composite",
            "graders": [
                {"type": "string_contains", "keywords": ["Snowflake"]},
                {"type": "numeric", "value": 3300.0, "tolerance": 1.0},
            ],
        },
    )
    # Good
    res_good = grade_task(task, "Snowflake has a higher cost by 3,300 USD.")
    assert res_good.passed is True

    # Bad: missing keyword
    res_bad_kw = grade_task(task, "The cost difference is 3,300 USD.")
    assert res_bad_kw.passed is False

    # Bad: wrong number
    res_bad_num = grade_task(task, "Snowflake has a higher cost by 1,200 USD.")
    assert res_bad_num.passed is False
