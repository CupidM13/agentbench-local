# Product Changelog

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
