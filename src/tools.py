"""Deterministic tools for AgentBench-Local with Pydantic validation,
safe AST evaluation, sandbox search/read, stateful calendar, and subprocess python execution.
"""

from __future__ import annotations
import ast
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


# =====================================================================
# Pydantic Schemas
# =====================================================================

class CalculatorArgs(BaseModel):
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate safely (e.g., '12 * 45 + 18' or 'round(sqrt(144) * 1.05, 2)').",
    )


class SearchFilesArgs(BaseModel):
    query: str = Field(..., description="Keyword or phrase to search for within the sandbox files.")
    max_results: int = Field(5, description="Maximum number of file matches to return.")


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Relative path of the sandbox file to read (e.g., 'prices.csv').")


class CalendarArgs(BaseModel):
    action: Literal["create", "list", "delete"] = Field(
        ...,
        description="Calendar action: 'create' to add an event, 'list' to view events, or 'delete' to remove an event.",
    )
    title: Optional[str] = Field(None, description="Title of the event (required for 'create').")
    date: Optional[str] = Field(
        None,
        description="ISO date 'YYYY-MM-DD' (required for 'create', optional filter for 'list').",
    )
    start: Optional[str] = Field(None, description="Start time in 24-hr format 'HH:MM' (required for 'create').")
    end: Optional[str] = Field(None, description="End time in 24-hr format 'HH:MM' (required for 'create').")
    event_id: Optional[str] = Field(None, description="Unique ID of the event to delete (required for 'delete').")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            parts = v.strip().split("-")
            if len(parts) != 3 or not (len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2):
                raise ValueError("Date must be in ISO format 'YYYY-MM-DD'.")
        return v

    @field_validator("start", "end")
    @classmethod
    def validate_time(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            parts = v.strip().split(":")
            if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
                raise ValueError("Time must be in 'HH:MM' 24-hour format.")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Time must be valid 24-hour time between 00:00 and 23:59.")
        return v


class RunPythonArgs(BaseModel):
    code: str = Field(..., description="Python 3 code to execute in an isolated environment.")


# =====================================================================
# Tool 1: AST Calculator
# =====================================================================

ALLOWED_MATH_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}

ALLOWED_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}


def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return ALLOWED_OPERATORS[op_type](operand)
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return ALLOWED_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only standard math function calls are allowed")
        fn_name = node.func.id
        if fn_name not in ALLOWED_MATH_FUNCS:
            raise ValueError(f"Function '{fn_name}' is not permitted")
        args = [_eval_ast_node(arg) for arg in node.args]
        return ALLOWED_MATH_FUNCS[fn_name](*args)
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluates an arithmetic expression using AST evaluation without eval()."""
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "Expression must be a non-empty string", "code": "EMPTY_EXPRESSION"}
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast_node(parsed)
        # Convert float to int if exact integer
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        return {"result": result, "expression": expression.strip()}
    except ZeroDivisionError:
        return {"error": "Division by zero", "code": "ZERO_DIVISION"}
    except SyntaxError as e:
        return {"error": f"Syntax error in expression: {e}", "code": "SYNTAX_ERROR"}
    except Exception as e:
        return {"error": f"Evaluation error: {e}", "code": "INVALID_EXPRESSION"}


# =====================================================================
# Tool 2: Sandbox File Search and Read
# =====================================================================

SANDBOX_DIR = Path(__file__).resolve().parent.parent / "data" / "sandbox"


def set_sandbox_dir(path: Union[str, Path]) -> None:
    global SANDBOX_DIR
    SANDBOX_DIR = Path(path).resolve()


def get_sandbox_dir() -> Path:
    return SANDBOX_DIR


def _sanitize_path(relative_path: str) -> Optional[Path]:
    """Ensures path stays strictly within the sandbox directory."""
    clean = relative_path.strip().lstrip("/")
    target = (SANDBOX_DIR / clean).resolve()
    try:
        target.relative_to(SANDBOX_DIR)
        return target
    except ValueError:
        return None


def search_files(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Keyword search across all files in the sandbox directory."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "Search query must be a non-empty string", "code": "EMPTY_QUERY"}
    if not SANDBOX_DIR.exists():
        return {"results": [], "query": query, "count": 0}

    query_lower = query.strip().lower()
    results: List[Dict[str, Any]] = []

    for root, _, files in os.walk(SANDBOX_DIR):
        for file in sorted(files):
            if file.startswith("."):
                continue
            file_path = Path(root) / file
            rel_name = file_path.relative_to(SANDBOX_DIR).as_posix()
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            matching_lines: List[str] = []
            for line_no, line in enumerate(content.splitlines(), start=1):
                if query_lower in line.lower():
                    matching_lines.append(f"L{line_no}: {line.strip()}")
                    if len(matching_lines) >= 3:
                        break

            if matching_lines:
                results.append({"file": rel_name, "snippets": matching_lines})
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    return {"results": results, "query": query, "count": len(results)}


def read_file(path: str) -> Dict[str, Any]:
    """Reads the full content of a file in the sandbox directory."""
    if not isinstance(path, str) or not path.strip():
        return {"error": "Path must be a non-empty string", "code": "EMPTY_PATH"}
    target = _sanitize_path(path)
    if target is None:
        return {"error": "Access denied: path traverses outside sandbox", "code": "FORBIDDEN_PATH"}
    if not target.exists() or not target.is_file():
        return {"error": f"File not found: '{path}'", "code": "NOT_FOUND"}

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
        return {"path": path, "content": content, "size_bytes": len(content)}
    except Exception as e:
        return {"error": f"Failed to read file: {e}", "code": "READ_ERROR"}


# =====================================================================
# Tool 3: Stateful Calendar with Fixture Reset
# =====================================================================

class CalendarStore:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._next_id: int = 1

    def reset(self, fixtures: Optional[List[Dict[str, Any]]] = None) -> None:
        self.events = []
        self._next_id = 1
        if fixtures:
            for item in fixtures:
                evt = deepcopy(item)
                if "id" not in evt:
                    evt["id"] = f"evt_{self._next_id:03d}"
                    self._next_id += 1
                else:
                    # Parse existing numeric ID if present to avoid collisions
                    id_str = str(evt["id"])
                    if id_str.startswith("evt_") and id_str[4:].isdigit():
                        val = int(id_str[4:])
                        if val >= self._next_id:
                            self._next_id = val + 1
                self.events.append(evt)

    def list_events(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        if date:
            target_date = date.strip()
            return [deepcopy(e) for e in self.events if e.get("date") == target_date]
        return [deepcopy(e) for e in self.events]

    def create_event(self, title: str, date: str, start: str, end: str) -> Dict[str, Any]:
        evt_id = f"evt_{self._next_id:03d}"
        self._next_id += 1
        new_event = {
            "id": evt_id,
            "title": title.strip(),
            "date": date.strip(),
            "start": start.strip(),
            "end": end.strip(),
        }
        self.events.append(new_event)
        return deepcopy(new_event)

    def delete_event(self, event_id: str) -> bool:
        clean_id = event_id.strip()
        before_len = len(self.events)
        self.events = [e for e in self.events if e.get("id") != clean_id]
        return len(self.events) < before_len


_GLOBAL_CALENDAR = CalendarStore()


def calendar_reset(fixtures: Optional[List[Dict[str, Any]]] = None) -> None:
    _GLOBAL_CALENDAR.reset(fixtures)


def get_calendar_state() -> List[Dict[str, Any]]:
    return _GLOBAL_CALENDAR.list_events()


def calendar(
    action: str,
    title: Optional[str] = None,
    date: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """In-memory mock calendar supporting 'create', 'list', and 'delete'."""
    act = (action or "").strip().lower()
    if act == "list":
        events = _GLOBAL_CALENDAR.list_events(date=date)
        return {"action": "list", "count": len(events), "events": events}

    elif act == "create":
        if not title:
            return {"error": "Missing required field 'title' for calendar create", "code": "MISSING_FIELD"}
        if not date:
            return {"error": "Missing required field 'date' for calendar create", "code": "MISSING_FIELD"}
        if not start or not end:
            return {"error": "Missing required fields 'start' and 'end' for calendar create", "code": "MISSING_FIELD"}
        created = _GLOBAL_CALENDAR.create_event(title=title, date=date, start=start, end=end)
        return {"action": "create", "status": "success", "event": created}

    elif act == "delete":
        if not event_id:
            return {"error": "Missing required field 'event_id' for calendar delete", "code": "MISSING_FIELD"}
        deleted = _GLOBAL_CALENDAR.delete_event(event_id)
        if not deleted:
            return {"error": f"Event '{event_id}' not found", "code": "NOT_FOUND"}
        return {"action": "delete", "status": "success", "event_id": event_id}

    else:
        return {"error": f"Invalid action '{action}'. Must be 'create', 'list', or 'delete'.", "code": "INVALID_ACTION"}


# =====================================================================
# Tool 4: Subprocess Python Runner
# =====================================================================

def run_python(code: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    """Executes Python code in a secure subprocess with a timeout and captured stdout."""
    if not isinstance(code, str) or not code.strip():
        return {"error": "Code must be a non-empty string", "code": "EMPTY_CODE"}

    tmp_dir = SANDBOX_DIR / ".py_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / f"run_{os.getpid()}_{int(1000 * os.times().elapsed)}.py"

    try:
        script_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SANDBOX_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {timeout_sec} seconds", "code": "TIMEOUT"}
    except Exception as e:
        return {"error": f"Subprocess execution error: {e}", "code": "EXEC_ERROR"}
    finally:
        if script_path.exists():
            try:
                script_path.unlink()
            except Exception:
                pass


# =====================================================================
# Tool Registry & JSON Schemas
# =====================================================================

TOOLS_REGISTRY = {
    "calculator": calculator,
    "search_files": search_files,
    "read_file": read_file,
    "calendar": calendar,
    "run_python": run_python,
}

TOOL_SCHEMAS_PYDANTIC = {
    "calculator": CalculatorArgs,
    "search_files": SearchFilesArgs,
    "read_file": ReadFileArgs,
    "calendar": CalendarArgs,
    "run_python": RunPythonArgs,
}

OPENAI_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluates an arithmetic expression (supports +, -, *, /, **, %, parentheses, sqrt, round, abs, min, max, pow). Returns numerical result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression string, e.g. '14 * 25 + 7' or 'round(sqrt(225), 2)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Searches for keywords across files in the sandbox repository. Returns matching file names and line snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or search term to look for.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of file matches to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the entire text content of a specified file inside data/sandbox/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the sandbox file (e.g. 'prices.csv', 'notes.md').",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar",
            "description": "Manage calendar events: create new events, list events by date, or delete existing events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "delete"],
                        "description": "Action to perform: 'create', 'list', or 'delete'.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the event (required for 'create').",
                    },
                    "date": {
                        "type": "string",
                        "description": "ISO date 'YYYY-MM-DD' (required for 'create', optional for 'list').",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start time in 24-hr format 'HH:MM' (required for 'create').",
                    },
                    "end": {
                        "type": "string",
                        "description": "End time in 24-hr format 'HH:MM' (required for 'create').",
                    },
                    "event_id": {
                        "type": "string",
                        "description": "Event ID to delete (e.g. 'evt_001', required for 'delete').",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Runs Python 3 code in an isolated subprocess with a 5-second timeout and returns stdout. Use when complex data transformation or logic is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python 3 code string to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


def execute_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatches tool call to registered function, returning result dictionary."""
    if tool_name not in TOOLS_REGISTRY:
        return {
            "error": f"Tool '{tool_name}' not found. Available tools: {list(TOOLS_REGISTRY.keys())}",
            "code": "UNKNOWN_TOOL",
        }
    fn = TOOLS_REGISTRY[tool_name]
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Argument signature mismatch: {e}", "code": "TYPE_ERROR"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}", "code": "TOOL_ERROR"}
