"""Programmatic ground-truth verification of all 60 tasks.
Proves that every task in data/tasks.yaml is solvable, unambiguous, and passes its grader.
"""

import pytest
from src.tasks import load_tasks, grade_task
from src.tools import (
    calendar_reset,
    calendar,
    get_calendar_state,
    calculator,
    search_files,
    read_file,
)


# Dictionary mapping task_id to programmatic solution: (answer_str, calendar_actions, tool_calls_made)
def solve_task(task_id: str):
    if task_id == "dev_01":
        # 245 * 18 - 350
        val = 245 * 18 - 350
        return str(val), [], [{"tool": "calculator"}]

    elif task_id == "dev_02":
        return "Clara Oswald's email is clara.oswald@company.internal.", [], [{"tool": "search_files"}]

    elif task_id == "dev_03":
        return "The statement_timeout_ms is 10000.", [], [{"tool": "read_file"}]

    elif task_id == "dev_04":
        # 349 * 3 = 1047
        return "The unit price is $349.00 and 3 chairs cost $1,047.00.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "dev_05":
        # 432500 - 225000 = 207500
        return "Engineering actual spend was 432,500 and Marketing was 225,000, difference is 207500.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "dev_06":
        # 49.99 + 129.50 = 179.49
        return "Combined unit price is 179.49 dollars.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "dev_07":
        # Snowflake (6500) vs DataDog (3200) -> 3300
        return "Snowflake has a higher monthly cost by 3300 USD.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "dev_08":
        calendar_reset([])
        calendar(action="create", title="Sprint Demo", date="2024-11-12", start="14:00", end="15:30")
        return "Created Sprint Demo.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "dev_09":
        # 59.90 * 0.8 = 47.92
        return "Discounted price is 47.92.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "dev_10":
        return "The capital of France is Paris.", [], []

    elif task_id == "dev_11":
        return "Hello! I am ready to assist you.", [], []

    elif task_id == "dev_12":
        return "The marketing budget for TikTok campaigns cannot be found in the provided files.", [], [{"tool": "search_files"}]

    elif task_id == "dev_13":
        return "Employee ID for Michael Scott is not available in the company roster.", [], [{"tool": "search_files"}]

    elif task_id == "dev_14":
        calendar_reset([{"id": "evt_001", "title": "Team Retro", "date": "2024-11-04", "start": "10:00", "end": "11:00"}])
        calendar(action="delete", event_id="evt_001")
        return "Deleted evt_001.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "dev_15":
        calendar_reset([])
        calendar(action="create", title="1-on-1 with Manager", date="2024-11-08", start="11:00", end="11:30")
        return "Scheduled 1-on-1.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "dev_16":
        # A-12 (148) + A-15 (85) + A-08 (95) = 328
        return "Total physical count for bin prefix A- is 328.", [], [{"tool": "search_files"}]

    elif task_id == "dev_17":
        return "OpenSSL was upgraded to 3.0.13.", [], [{"tool": "read_file"}]

    elif task_id == "dev_18":
        # sqrt(289) + 3**4 = 17 + 81 = 98
        return "Result is 98.0.", [], [{"tool": "calculator"}]

    elif task_id == "dev_19":
        return "Mars is known as the Red Planet.", [], []

    elif task_id == "dev_20":
        calendar_reset([
            {"id": "evt_001", "title": "Standup", "date": "2024-11-04", "start": "09:00", "end": "09:30"},
            {"id": "evt_002", "title": "Sync", "date": "2024-11-04", "start": "10:00", "end": "10:30"},
        ])
        calendar(action="delete", event_id="evt_001")
        calendar(action="create", title="Design Review", date="2024-11-04", start="15:00", end="16:00")
        return "Updated calendar.", get_calendar_state(), [{"tool": "calendar"}]

    # Test set tasks (test_01 to test_40)
    elif task_id == "test_01":
        # (1584 / 12) + (37 * 14) = 132 + 518 = 650
        return "Calculated result is 650.", [], [{"tool": "calculator"}]

    elif task_id == "test_02":
        return "Stock quantity is 210.", [], [{"tool": "search_files"}]

    elif task_id == "test_03":
        return "The home office stipend is 1200 dollars.", [], [{"tool": "read_file"}]

    elif task_id == "test_04":
        return "Elena Rostova is an Information Security Analyst.", [], [{"tool": "search_files"}]

    elif task_id == "test_05":
        return "Max idle session timeout is 15 minutes.", [], [{"tool": "read_file"}]

    elif task_id == "test_06":
        # 8**4 - 2**10 = 4096 - 1024 = 3072
        return "Result is 3072.", [], [{"tool": "calculator"}]

    elif task_id == "test_07":
        # 45 * 6 * 0.9 = 243
        return "Total cost for 6 lamps with 10% discount is 243.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_08":
        # 2.4 + 1.8 = 4.2
        return "Total resolution time is 4.2 hours.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_09":
        # 14500 * 12 = 174000
        return "Annual cost for AWS is 174000.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_10":
        # -3 * 59.90 = -179.7
        return "Discrepancy value is -179.7 dollars.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_11":
        # 91.2 - 45.0 = 46.2
        return "Difference in memory percentage is 46.2.", [], [{"tool": "read_file"}, {"tool": "calculator"}]

    elif task_id == "test_12":
        return "Clara Oswald is benchmarking OpenSearch 2.11 for release on November 15, 2024.", [], [{"tool": "read_file"}]

    elif task_id == "test_13":
        # 8.50 + 1.25 * 8 = 18.50
        return "Shipping cost to ZONE_1 is 18.50.", [], [{"tool": "read_file"}, {"tool": "calculator"}]

    elif task_id == "test_14":
        # 85000 + 120000 = 205000
        return "Sum of budget allocated is 205000.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_15":
        return "Alice Chen joined in 2021-03-15, which is before Bob Martinez.", [], [{"tool": "search_files"}]

    elif task_id == "test_16":
        # 18 + 95 + 130 = 243
        return "Total stock quantity across SKU-1006, 1007, 1008 is 243.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_17":
        # 1100 + 2400 = 3500
        return "Combined monthly cost is 3500.", [], [{"tool": "search_files"}, {"tool": "calculator"}]

    elif task_id == "test_18":
        return "The lead engineers for Q4 milestones are Clara Oswald and David Kim.", [], [{"tool": "read_file"}]

    elif task_id == "test_19":
        # 1420 + 2100 = 3520
        return "Total active connections is 3520.", [], [{"tool": "read_file"}, {"tool": "calculator"}]

    elif task_id == "test_20":
        # 349.00 * 0.88 + 49.99 * 0.90 = 307.12 + 44.991 = 352.111 -> 352.11
        return "The rounded total is 352.11.", [], [{"tool": "calculator"}]

    elif task_id == "test_21":
        calendar_reset([])
        calendar(action="create", title="Architecture Sync", date="2024-12-05", start="09:30", end="10:45")
        return "Created Architecture Sync.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "test_22":
        # 17500 / 450000 * 100 = 3.8888... -> 3.89
        return "Engineering budget surplus percentage is 3.89%.", [], [{"tool": "calculator"}]

    elif task_id == "test_23":
        calendar_reset([])
        calendar(action="create", title="Quarterly All-Hands", date="2025-01-15", start="13:00", end="14:00")
        return "Scheduled All-Hands.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "test_24":
        # 1250 % 37 = 29
        return "Value is 29.", [], [{"tool": "calculator"}]

    elif task_id == "test_25":
        # 18.00 * 1.75 = 31.50
        return "Express rate is 31.50.", [], [{"tool": "read_file"}, {"tool": "calculator"}]

    elif task_id == "test_26":
        return "Water boils at 100 degrees Celsius at sea level.", [], []

    elif task_id == "test_27":
        return "William Shakespeare wrote Romeo and Juliet.", [], []

    elif task_id == "test_28":
        return "Good morning! Yes, I am fully ready to assist you.", [], []

    elif task_id == "test_29":
        return "Mixing yellow and blue produces green.", [], []

    elif task_id == "test_30":
        return "There are 366 days in a leap year.", [], []

    elif task_id == "test_31":
        return "Tokyo offsite expenses cannot be found in the documentation.", [], [{"tool": "search_files"}]

    elif task_id == "test_32":
        return "Personal address and phone for Henry Zhang are unavailable in roster data.", [], [{"tool": "search_files"}]

    elif task_id == "test_33":
        return "Secret API credentials for Stripe are not present in the files.", [], [{"tool": "search_files"}]

    elif task_id == "test_34":
        return "Return policy for Australia is not mentioned in shipping_rates.json.", [], [{"tool": "read_file"}]

    elif task_id == "test_35":
        return "Information regarding Employee of the Month is unavailable.", [], [{"tool": "read_file"}]

    elif task_id == "test_36":
        calendar_reset([
            {"id": "evt_101", "title": "Old Standup", "date": "2024-11-01", "start": "09:00", "end": "09:15"},
            {"id": "evt_102", "title": "Project Review", "date": "2024-11-01", "start": "14:00", "end": "15:00"},
        ])
        calendar(action="delete", event_id="evt_101")
        return "Deleted evt_101.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "test_37":
        calendar_reset([
            {"id": "evt_201", "title": "Morning Coffee", "date": "2024-11-18", "start": "08:30", "end": "09:00"}
        ])
        calendar(action="create", title="Client Demo", date="2024-11-18", start="10:00", end="11:00")
        return "Scheduled Client Demo.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "test_38":
        calendar_reset([
            {"id": "evt_301", "title": "Standup", "date": "2024-11-20", "start": "09:00", "end": "09:30"},
            {"id": "evt_302", "title": "Sync", "date": "2024-11-20", "start": "13:00", "end": "13:30"},
        ])
        calendar(action="delete", event_id="evt_301")
        calendar(action="delete", event_id="evt_302")
        return "Cleared events.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "test_39":
        calendar_reset([])
        calendar(action="create", title="Team Lunch", date="2024-11-22", start="12:00", end="13:00")
        calendar(action="create", title="Sprint Retro", date="2024-11-22", start="16:00", end="17:00")
        return "Scheduled both events.", get_calendar_state(), [{"tool": "calendar"}]

    elif task_id == "test_40":
        calendar_reset([
            {"id": "evt_501", "title": "Tentative Sync", "date": "2024-11-25", "start": "11:00", "end": "11:30"},
            {"id": "evt_502", "title": "Engineering Review", "date": "2024-11-25", "start": "14:00", "end": "15:00"},
        ])
        calendar(action="delete", event_id="evt_501")
        calendar(action="create", title="Executive Briefing", date="2024-11-25", start="11:00", end="12:00")
        return "Updated calendar.", get_calendar_state(), [{"tool": "calendar"}]

    else:
        raise ValueError(f"Unknown task: {task_id}")


def test_all_60_tasks_verified_solvable():
    tasks = load_tasks()
    assert len(tasks) == 60, f"Expected 60 tasks, got {len(tasks)}"

    dev_count = sum(1 for t in tasks if t.split == "dev")
    test_count = sum(1 for t in tasks if t.split == "test")
    assert dev_count == 20, f"Expected 20 dev tasks, got {dev_count}"
    assert test_count == 40, f"Expected 40 test tasks, got {test_count}"

    for task in tasks:
        ans, cal_state, tools_made = solve_task(task.id)
        res = grade_task(task, answer=ans, calendar_state=cal_state, tool_calls_made=tools_made)
        assert res.passed is True, f"Task {task.id} failed verification: {res.reason}"
