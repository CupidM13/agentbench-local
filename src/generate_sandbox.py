"""Populates data/sandbox with 15 synthetic, deterministic data files for file search and multi-step reasoning."""

from pathlib import Path
import json

SANDBOX_DIR = Path(__file__).resolve().parent.parent / "data" / "sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "prices.csv": """sku,product_name,category,unit_price,stock_quantity,discount_pct
SKU-1001,Wireless Ergonomic Mouse,Electronics,49.99,150,10
SKU-1002,Mechanical Keyboard RGB,Electronics,129.50,85,15
SKU-1003,Noise-Cancelling Headphones,Electronics,199.00,42,5
SKU-1004,Standing Desk Mat,Office Supplies,39.95,210,0
SKU-1005,USB-C Dual Hub 4K,Electronics,59.90,75,20
SKU-1006,Ergonomic Mesh Chair,Furniture,349.00,18,0
SKU-1007,LED Desk Lamp with Qi,Electronics,45.00,95,10
SKU-1008,Aluminium Laptop Stand,Office Supplies,32.50,130,5
SKU-1009,4K Ultra-Wide Monitor 34in,Electronics,549.99,22,12
SKU-1010,Cable Management Box,Office Supplies,19.99,300,0
""",

    "roster.csv": """emp_id,full_name,department,role,email,start_date
EMP-01,Alice Chen,Engineering,Staff Infrastructure Engineer,alice.chen@company.internal,2021-03-15
EMP-02,Bob Martinez,Product,Senior Product Manager,bob.martinez@company.internal,2022-06-01
EMP-03,Clara Oswald,Engineering,Backend Software Engineer,clara.oswald@company.internal,2023-01-10
EMP-04,David Kim,Design,Lead Product Designer,david.kim@company.internal,2020-11-20
EMP-05,Elena Rostova,Security,Information Security Analyst,elena.rostova@company.internal,2022-09-15
EMP-06,Farhan Siddiqui,Data,Data Science Lead,farhan.siddiqui@company.internal,2021-08-01
EMP-07,Grace Hopper,Engineering,Principal Systems Architect,grace.hopper@company.internal,2019-05-01
EMP-08,Henry Zhang,Sales,Account Executive,henry.zhang@company.internal,2023-04-18
EMP-09,Isabella Rossi,Marketing,Content Marketing Manager,isabella.rossi@company.internal,2022-02-14
EMP-10,Jamal Washington,Product,Associate Product Manager,jamal.washington@company.internal,2023-10-01
""",

    "meeting_notes.md": """# Engineering & Product Meeting Notes (October 2024)

## Sprint 42 Planning - 2024-10-02
- Attendees: Alice Chen, Clara Oswald, Bob Martinez, David Kim
- Key Decision: Migrate search index from legacy Elasticsearch to OpenSearch 2.11.
- Target Release: Sprint 44 (November 15, 2024).
- Clara Oswald assigned to benchmark vector retrieval latency under 100 QPS load.
- Bob Martinez agreed to finalize PRD for user workspaces by 2024-10-18.

## Security Review - 2024-10-09
- Attendees: Elena Rostova, Alice Chen, Grace Hopper
- Finding SEC-801: Database credentials rotation script failed in staging.
- Remediation: Rotate staging RDS secrets immediately and enforce 30-day automated rotation.
- Status: Resolved on 2024-10-10 by Alice Chen.

## Q4 Roadmap Sync - 2024-10-16
- Attendees: Bob Martinez, Jamal Washington, David Kim, Farhan Siddiqui
- Decision: Postpone multi-tenant billing tier to Q1 2025.
- Priority for Q4 is customer retention dashboard and self-serve onboarding wizard.
- Expected budget ceiling for Q4 tooling is $45,000.
""",

    "changelog.md": """# Product Changelog

## [v2.4.0] - 2024-09-30
### Added
- Granular RBAC permissions for workspace admin roles.
- Dark mode theme toggle in user preferences.
- Webhook endpoints for Slack and Discord alerts.
### Fixed
- Fixed memory leak in websocket message streaming listener (#1420).
- Corrected currency formatting on invoice PDF generation.

## [v2.3.0] - 2024-08-15
### Added
- CSV batch export for audit logs up to 100,000 records.
- Support for OAuth2 SSO via Okta and Google Workspace.
### Deprecated
- Legacy API v1 endpoints will sunset on 2024-12-31.

## [v2.2.0] - 2024-06-20
### Added
- Native tool calling support for agent assistants.
- Vector search filtering by metadata tags.
### Security
- Upgraded OpenSSL to 3.0.13 to remediate CVE-2024-2511.
""",

    "expenses_q3.csv": """department,budget_allocated,actual_spend,variance,approved_by
Engineering,450000,432500,17500,Grace Hopper
Product,180000,192000,-12000,Bob Martinez
Design,85000,78200,6800,David Kim
Security,120000,114000,6000,Elena Rostova
Marketing,210000,225000,-15000,Isabella Rossi
Data,160000,153500,6500,Farhan Siddiqui
Sales,250000,241000,9000,Henry Zhang
""",

    "server_metrics.json": json.dumps({
        "cluster_name": "prod-us-east-1",
        "timestamp": "2024-10-25T12:00:00Z",
        "nodes": [
            {"node_id": "node-01", "role": "api-gateway", "cpu_pct": 42.5, "memory_pct": 68.1, "uptime_days": 114, "active_connections": 1420},
            {"node_id": "node-02", "role": "worker-heavy", "cpu_pct": 88.4, "memory_pct": 91.2, "uptime_days": 45, "active_connections": 380},
            {"node_id": "node-03", "role": "worker-general", "cpu_pct": 55.0, "memory_pct": 74.3, "uptime_days": 82, "active_connections": 890},
            {"node_id": "node-04", "role": "db-replica", "cpu_pct": 61.2, "memory_pct": 84.7, "uptime_days": 210, "active_connections": 2100},
            {"node_id": "node-05", "role": "cache-redis", "cpu_pct": 24.1, "memory_pct": 45.0, "uptime_days": 180, "active_connections": 5400}
        ],
        "cluster_health": "WARNING_HIGH_WORKER_LOAD"
    }, indent=2),

    "shipping_rates.json": json.dumps({
        "zones": {
            "ZONE_1": {"region": "North America Domestic", "base_rate": 8.50, "per_kg": 1.25, "sla_days": 3},
            "ZONE_2": {"region": "Western Europe", "base_rate": 18.00, "per_kg": 3.50, "sla_days": 5},
            "ZONE_3": {"region": "Asia Pacific", "base_rate": 24.50, "per_kg": 4.75, "sla_days": 7},
            "ZONE_4": {"region": "Rest of World", "base_rate": 35.00, "per_kg": 6.00, "sla_days": 10}
        },
        "express_multiplier": 1.75,
        "free_shipping_threshold_domestic": 75.00
    }, indent=2),

    "project_roadmap.md": """# Engineering Project Roadmap 2024-2025

## Q4 2024
- Milestone: OpenSearch 2.11 Migration (Lead: Clara Oswald, Status: In Progress)
- Milestone: Onboarding Wizard Revamp (Lead: David Kim, Status: Ready for QA)
- Milestone: Automated RDS Secret Rotation (Lead: Alice Chen, Status: Completed)

## Q1 2025
- Milestone: Multi-Tenant Billing Engine (Lead: Bob Martinez, Status: Scheduled)
- Milestone: SOC2 Type II Audit Certification (Lead: Elena Rostova, Status: Scheduled)
- Milestone: Distributed Tracing with OpenTelemetry (Lead: Grace Hopper, Status: Proposed)
""",

    "inventory_audit.csv": """item_id,warehouse_bin,physical_count,system_count,discrepancy,reorder_point
SKU-1001,A-12,148,150,-2,50
SKU-1002,A-15,85,85,0,30
SKU-1003,B-04,41,42,-1,20
SKU-1004,C-01,210,210,0,80
SKU-1005,B-09,72,75,-3,25
SKU-1006,D-02,18,18,0,10
SKU-1007,A-08,95,95,0,40
SKU-1008,C-07,130,130,0,50
SKU-1009,D-05,20,22,-2,15
SKU-1010,C-11,298,300,-2,100
""",

    "security_policy.md": """# Corporate Security Baseline Guidelines

## Access Control
- Master passwords must be at least 16 characters with high entropy.
- Mandatory password rotation every 90 days for administrative accounts.
- Multi-factor authentication (MFA) via FIDO2 hardware token or TOTP is required for all employees.
- Maximum idle session timeout across web interfaces is 15 minutes.

## Incident Escalation
- Severity 1 (Critical Outage / Breach): Notify On-Call Commander within 5 minutes.
- Severity 2 (Degraded Security Control): Notify Security Team within 30 minutes.
- Elena Rostova is the primary Security Operations lead for Q4 2024.
""",

    "customer_tickets.csv": """ticket_id,priority,customer_tier,category,resolution_hours,status
TICK-501,P1,Enterprise,API Authentication,2.4,Resolved
TICK-502,P3,Standard,UI Dashboard,14.8,Resolved
TICK-503,P2,Enterprise,Export Data Timeout,5.1,Resolved
TICK-504,P1,Enterprise,Database Connection Drop,1.8,Resolved
TICK-505,P4,Free,Feature Request,48.0,Closed
TICK-506,P2,Standard,Billing Calculation,8.5,Resolved
TICK-507,P3,Enterprise,Webhook Delivery,11.2,Resolved
TICK-508,P1,Enterprise,SSO SAML Failure,3.0,Resolved
""",

    "app_config.json": json.dumps({
        "environment": "production",
        "database": {
            "max_connections": 100,
            "min_idle_connections": 10,
            "connection_timeout_ms": 3000,
            "statement_timeout_ms": 10000
        },
        "rate_limiting": {
            "anonymous_rpm": 60,
            "authenticated_rpm": 600,
            "enterprise_rpm": 3000
        },
        "cache": {
            "default_ttl_seconds": 3600,
            "max_memory_mb": 4096,
            "eviction_policy": "allkeys-lru"
        },
        "features": {
            "enable_vector_search": True,
            "enable_beta_workflows": False,
            "maintenance_mode": False
        }
    }, indent=2),

    "vendor_contracts.csv": """vendor_name,service_type,monthly_cost_usd,renewal_date,sla_guarantee,primary_contact
CloudScale AWS,Cloud Infrastructure,14500,2025-06-30,99.99%,sarah@cloudscale.net
DataDog,Observability & APM,3200,2025-02-28,99.9%,matt.apm@datadoghq.com
PagerDuty,Incident Management,1100,2024-12-15,99.95%,support@pagerduty.com
AuthZero,Identity Provider,2400,2025-04-30,99.99%,enterprise@auth0.com
GitHub Enterprise,Source Control & CI,1850,2025-08-31,99.9%,sales@github.com
Snowflake,Data Warehousing,6500,2025-01-31,99.9%,analytics@snowflake.com
""",

    "interview_schedule.txt": """Candidate Interview Schedule - Week of Nov 4, 2024

Candidate: Marcus Vance
Role: Staff Security Engineer
Date: 2024-11-04
Time: 10:00 - 11:30
Interviewers: Elena Rostova, Grace Hopper
Room: Turing Conference Room B

Candidate: Priya Patel
Role: Senior Frontend Engineer
Date: 2024-11-05
Time: 14:00 - 15:30
Interviewers: David Kim, Clara Oswald
Room: Lovelace Conference Room A

Candidate: Liam O'Connor
Role: Product Manager (Growth)
Date: 2024-11-06
Time: 11:00 - 12:00
Interviewers: Bob Martinez, Jamal Washington
Room: Hopper Boardroom
""",

    "faq.md": """# Internal Team FAQ

## Office & Work Guidelines
- Core collaboration hours are 10:00 to 16:00 local time.
- Annual home office stipend is $1,200 per employee, reimbursable through Expensify.
- Monthly wellness benefit is $75 per month.

## Expense Reports
- All receipts must be submitted within 30 days of the transaction date.
- Expenses over $500 require explicit pre-approval from department VP.
- Flight bookings for business travel must be economy class for flights under 6 hours.
"""
}

for filename, content in FILES.items():
    file_path = SANDBOX_DIR / filename
    file_path.write_text(content.strip() + "\n", encoding="utf-8")

print(f"Generated {len(FILES)} synthetic files in {SANDBOX_DIR}")
