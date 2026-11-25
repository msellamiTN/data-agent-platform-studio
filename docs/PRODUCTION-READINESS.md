# Production Readiness — Release Gates

This document distinguishes implemented prototype controls from capabilities required before a production deployment. A checked code-level control is not a deployment approval.

## Current baseline

| Domain | Current state | Production release gate |
|---|---|---|
| Data ingestion | CSV allowlist, upload/row/column and prompt size caps, UTF-8 unique headers, workspace copy | Content validation, malware scanning where appropriate, quotas, retention and ownership |
| Pipeline consistency | Immutable run artifacts; atomic active manifest publication | Concurrent-user isolation, run lifecycle, cleanup and recovery |
| Data quality | Trim, duplicate and empty-row rules; per-column profile | Configurable contracts, schema evolution, rejection/quarantine policy, business validation |
| Analytical store | Run-scoped SQLite warehouse | Governed storage service, encryption, backup, restore, concurrency and lifecycle controls |
| Query safety | SELECT/WITH allowlist, active-table requirement, read-only URI, SQLite limits, query time, streaming row, cell and response-byte bounds | Resource isolation, per-user authorization, audit trails, query policy tests and load testing |
| Agent boundary | Fixed ETL, schema-inspection and SQL tools | Tool-level identity, scoped authorization, policy evaluation, prompt-injection red-team tests |
| Model runtime | Local Qwen through Transformers, CPU/GPU mode | Model artifact provenance, pinned hashes, licensing review, resource quotas and measured performance |
| UI | Gradio Blocks, readiness state, execution-mode disclosure | Authenticated deployment, accessibility review, browser matrix and design-system QA |
| Operations | GitHub Actions compile + deterministic tests | Dependency lock, SBOM, vulnerability scanning, observability, alerting, incident response and SLOs |
| Multi-tenancy | Not implemented | Tenant isolation for data, model requests, artifacts, logs and lifecycle operations |
| Privacy | Synthetic-data recommendation | Data classification, lawful basis, minimization, retention, deletion and privacy review |

## Implemented engineering invariants

1. A successful pipeline run has its own immutable Bronze, Silver, Gold and metadata artifacts.
2. The active manifest changes only after all run artifacts have been written successfully.
3. An unsuccessful pipeline attempt does not replace the previously active manifest.
4. Analysis uses the database path from the active manifest and validates that it is inside the pipeline workspace.
5. SQL execution can read only the canonical `silver_data` table and is time-, row-, cell- and response-byte-bounded; results are streamed and BLOBs are omitted.
6. The model input has a configurable token budget; oversized prompts are rejected, not silently truncated.
6. Supported count intents are matched against actual column names; an unknown count column returns a correction message rather than an unrelated preview.
7. Numeric aggregate summaries are generated from executed SQL values rather than unverified model prose.
8. The UI distinguishes an aggregate answer, a raw read-only query result, and a preview-only fallback.
9. Analysis is unavailable until a valid active pipeline exists; publishing a new pipeline clears stale analysis output.
10. The CrewAI/Torch integration is loaded lazily so deterministic data functions remain importable when optional model/orchestration dependencies fail.

## Required release gates

### G0 — Reproducible build and dependency governance
- [ ] Pin supported Python versions and produce a reproducible dependency lock.
- [ ] Resolve the Pillow/Gradio version constraint safely; verify the complete runtime install with `pip check`.
- [ ] Generate an SBOM and run automated vulnerability scanning.
- [ ] Record model repository revision, artifact hash, license and provenance.

### G1 — Correctness and data contracts
- [ ] Add versioned schema contracts and schema-drift behaviour.
- [ ] Validate CSV encoding, delimiters, malformed rows, duplicate column names and type coercion.
- [ ] Add golden-dataset acceptance tests for all supported query intents.
- [ ] Define ambiguity, unsupported-question and confidence behaviour for natural-language queries.
- [ ] Measure precision/recall of generated SQL on a reviewed benchmark before business use.

### G2 — Security and privacy
- [ ] Add OIDC/SAML-backed authentication and role-based access control.
- [ ] Isolate workspace, run manifest, database, logs and model context by tenant.
- [ ] Encrypt storage and transport; manage keys and secrets through an approved service.
- [ ] Add audit events for upload, pipeline publication, query execution, export and deletion.
- [ ] Define data retention, secure deletion, backup and restore.
- [ ] Conduct prompt-injection, tool-abuse, SQL-policy and upload abuse tests.
- [ ] Complete privacy and regulatory assessments applicable to the deployment.

### G3 — Reliability and operations
- [ ] Add bounded job execution, cancellation, retry and idempotency policies.
- [ ] Add metrics, structured logs, traces, dashboards, alerts and incident runbooks.
- [ ] Run concurrency, memory, CPU/GPU saturation and failure-recovery tests.
- [ ] Define service objectives and operational ownership.
- [ ] Test rolling upgrade and rollback procedures.

### G4 — Product and accessibility
- [ ] Verify the UX in supported browsers and responsive breakpoints.
- [ ] Complete keyboard, contrast and screen-reader accessibility review.
- [ ] Add clear loading, cancellation, empty, success, warning and failure states.
- [ ] Ensure all displayed answers link to the exact dataset/run and executed query.
- [ ] Complete user acceptance with representative CSVs and domain questions.

## Release decision

The repository currently supplies a **production-minded prototype baseline**, not a production-ready service. Do not promote the PR or deploy with sensitive data until the relevant gates have evidence and an accountable reviewer has approved the result.
