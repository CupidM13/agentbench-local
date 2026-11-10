"""Unit tests for the 4 deterministic tools:
calculator, search_files/read_file, calendar, and run_python.
"""

import pytest
from src.tools import (
    calculator,
    search_files,
    read_file,
    calendar,
    calendar_reset,
    get_calendar_state,
    run_python,
    execute_tool_call,
)


# =====================================================================
# Calculator Tests
# =====================================================================

def test_calculator_basic_arithmetic():
    assert calculator("2 + 2")["result"] == 4
    assert calculator("10 - 3")["result"] == 7
    assert calculator("6 * 7")["result"] == 42
    assert calculator("15 / 3")["result"] == 5
    assert calculator("2 ** 5")["result"] == 32
    assert calculator("17 % 5")["result"] == 2


def test_calculator_functions_and_precedence():
    res = calculator("round(sqrt(144) * 1.5, 2)")
    assert res["result"] == 18.0
    assert calculator("min(10, 4, 25)")["result"] == 4
    assert calculator("max(10, 4, 25)")["result"] == 25
    assert calculator("abs(-42)")["result"] == 42


def test_calculator_security_and_errors():
    # Division by zero
    res = calculator("10 / 0")
    assert "error" in res and res["code"] == "ZERO_DIVISION"

    # Syntax error
    res = calculator("2 + * 3")
    assert "error" in res and res["code"] == "SYNTAX_ERROR"

    # Arbitrary code execution / import attempt
    res = calculator("__import__('os').system('ls')")
    assert "error" in res

    # Empty expression
    res = calculator("  ")
    assert "error" in res and res["code"] == "EMPTY_EXPRESSION"


# =====================================================================
# Search & Read Files Tests
# =====================================================================

def test_search_files_keyword():
    res = search_files(query="Mechanical Keyboard RGB")
    assert res["count"] > 0
    assert any("prices.csv" in match["file"] for match in res["results"])


def test_search_files_limits_and_empty():
    res = search_files(query="Engineering", max_results=2)
    assert len(res["results"]) <= 2

    # Query with no matches
    res_none = search_files(query="NONEXISTENT_KEYWORD_XYZ999")
    assert res_none["count"] == 0

    # Empty query
    res_empty = search_files(query="")
    assert "error" in res_empty


def test_read_file_success_and_traversal_prevention():
    res = read_file("prices.csv")
    assert "content" in res
    assert "SKU-1001" in res["content"]

    # Traversal attempt
    res_forbidden = read_file("../../../etc/passwd")
    assert "error" in res_forbidden
    assert res_forbidden["code"] in ("FORBIDDEN_PATH", "NOT_FOUND")

    # Non-existent file
    res_missing = read_file("does_not_exist_file.txt")
    assert "error" in res_missing
    assert res_missing["code"] == "NOT_FOUND"


# =====================================================================
# Calendar Tests
# =====================================================================

def test_calendar_crud_and_fixture_reset():
    fixture = [
        {"id": "evt_001", "title": "Team Standup", "date": "2024-11-04", "start": "09:30", "end": "10:00"}
    ]
    calendar_reset(fixture)
    events = get_calendar_state()
    assert len(events) == 1
    assert events[0]["title"] == "Team Standup"

    # List events
    res_list = calendar(action="list", date="2024-11-04")
    assert res_list["count"] == 1
    assert res_list["events"][0]["id"] == "evt_001"

    # List on empty date
    res_empty_date = calendar(action="list", date="2024-12-25")
    assert res_empty_date["count"] == 0

    # Create event
    res_create = calendar(
        action="create",
        title="Product Review",
        date="2024-11-05",
        start="14:00",
        end="15:00",
    )
    assert res_create["status"] == "success"
    created_id = res_create["event"]["id"]
    assert len(get_calendar_state()) == 2

    # Delete event
    res_del = calendar(action="delete", event_id=created_id)
    assert res_del["status"] == "success"
    assert len(get_calendar_state()) == 1

    # Delete non-existent event
    res_del_fail = calendar(action="delete", event_id="evt_nonexistent")
    assert "error" in res_del_fail


def test_calendar_validation_errors():
    calendar_reset([])
    # Missing required fields
    res = calendar(action="create", title="Incomplete Event")
    assert "error" in res
    assert res["code"] == "MISSING_FIELD"

    # Invalid action
    res_invalid = calendar(action="dance")
    assert "error" in res_invalid
    assert res_invalid["code"] == "INVALID_ACTION"


# =====================================================================
# Subprocess Python Runner Tests
# =====================================================================

def test_run_python_success():
    code = "import json; print(json.dumps({'status': 'ok', 'sum': sum([1, 2, 3, 4])}))"
    res = run_python(code)
    assert res["success"] is True
    assert '"sum": 10' in res["stdout"]


def test_run_python_syntax_error():
    res = run_python("def bad_syntax(")
    assert res["success"] is False
    assert res["returncode"] != 0
    assert "SyntaxError" in res["stderr"]


def test_run_python_timeout():
    res = run_python("import time; time.sleep(10)", timeout_sec=0.5)
    assert "error" in res
    assert res["code"] == "TIMEOUT"


# =====================================================================
# Tool Dispatch Registry Tests
# =====================================================================

def test_execute_tool_call_dispatch():
    res = execute_tool_call("calculator", {"expression": "100 / 4"})
    assert res["result"] == 25

    res_unknown = execute_tool_call("magic_wand", {"spell": "lumos"})
    assert "error" in res_unknown
    assert res_unknown["code"] == "UNKNOWN_TOOL"
