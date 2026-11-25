# Security Policy

## Supported use

This repository is currently an educational and engineering prototype. It is **not approved for production data, regulated workloads, or multi-tenant deployment**. Until the production-readiness gates below are implemented, use synthetic or non-sensitive CSV files only.

## Reporting a vulnerability

Please do not disclose credentials, personal data, or exploitable details in a public issue. Use GitHub's private vulnerability reporting for this repository when enabled, or contact the repository maintainer privately through an existing trusted channel. Include the affected commit, component, reproduction conditions, and impact. Do not attach real customer data.

## Current security controls

- Only CSV files are accepted by the pipeline copy function; file size, parsed row count and column count are limited by `MAX_UPLOAD_MB`, `MAX_DATA_ROWS` and `MAX_DATA_COLUMNS`.
- UTF-8 headers must be non-empty and unique case-insensitively; business-context and analysis-question lengths are bounded.
- Uploaded files are copied into the application workspace before ETL.
- Each successful ETL run receives an immutable run directory and independent SQLite database.
- The active pipeline manifest is atomically replaced only after all run artifacts have been completed.
- The SQL interface accepts a single `SELECT` or `WITH` statement, rejects comments and multiple statements, opens SQLite read-only, restricts reads to `silver_data`, applies a query time budget, and caps returned rows.
- Agents are exposed only to fixed Python tools; no shell or arbitrary Python tool is supplied.
- The model executes locally through Transformers. No external hosted-agent endpoint is configured by this application.
- Business descriptions and sample values are treated as data/context, not as instructions that can extend tool permissions.

These controls reduce specific risks; they do not constitute a formal security certification or complete threat model.

## Known limitations and production blockers

- No user authentication, identity integration, role-based authorization, or tenant isolation.
- One application workspace and one active manifest are shared by all sessions in the process.
- Gradio share URLs are public to anyone who obtains the URL; the application does not provide an access-control layer for a shared URL.
- Workspace files and SQLite databases are stored locally without application-managed encryption at rest.
- Run artifacts are retained; automated retention, secure deletion, backup, and recovery policies are not implemented.
- There is no centralized audit trail, SIEM integration, rate limiting, abuse detection, or operational alerting.
- There is no formal privacy classification, PII detection/redaction, or customer data governance workflow.
- Dependency compatibility and security need a maintained lockfile and automated vulnerability review before a production release.
- The current Gradio dependency range constrains the Pillow version below 13.0.0. Pillow 13.0.0 contains security fixes that are not backported to earlier versions; this dependency conflict must be resolved before accepting untrusted image content or treating the dependency set as production-hardened. The application currently accepts CSV, not image uploads, but that does not replace dependency review.

## Production deployment rule

Do not expose this prototype to untrusted users or sensitive datasets. Do not change the Gradio share setting to public for customer data. Complete and review every applicable item in [Production Readiness](docs/PRODUCTION-READINESS.md) before production deployment.
