# Data Agent Platform Studio — UX / Visual Design Specification

## Product experience

The Studio is a focused data workspace for a single active data pipeline and two specialized local agents. The interface should make three facts immediately visible:

1. Which dataset is active?
2. What stage of the shared pipeline has completed?
3. What can each agent do with that same dataset?

The UX avoids a chatbot-first landing screen. It presents a data workspace first, then exposes agent interaction within the relevant workflow.

## Information architecture

### Overview
Purpose: answer "where am I and what is ready?"

- Studio identity and runtime chips (local model, GPU/CPU, CrewAI).
- Hero statement explaining one pipeline / two agents.
- Active pipeline monitor.
- KPIs: active dataset, curated rows, profiled columns, duplicates removed.
- Bronze → Silver → Gold → manifest lifecycle.
- Two agent capability cards and hand-off explanation.

### Data Engineering
Purpose: prepare a dataset once and publish its contract.

- Source upload (optional upload; bundled retail example when empty).
- Optional business description to define metrics and domain language.
- Explicit controls for trimming, exact duplicates and fully empty rows.
- Primary action: Run data pipeline.
- Result summary with counts and pipeline ID.
- Progressive disclosure: quality report, Silver preview/download, active manifest.
- On success, refresh Overview and make the active pipeline visible to the Analysis Agent.

### Data Analysis
Purpose: ask a question about the active dataset, not upload another one.

- Question-first input and example prompts.
- Short explanation of the four-step agent process.
- Answer panel with execution mode and dataset/pipeline reference.
- SQL shown in a collapsed section for transparency without overwhelming business users.
- Result rows and chart shown beneath the answer.

### Platform
Purpose: explain execution and trust boundaries.

- Gradio → CrewAI → local Qwen → bounded Python tools → SQLite.
- Model and SQL safety explanations.
- Colab runtime caveat and share-link privacy note.

## Visual language

- **Canvas:** soft cool grey, to distinguish the working surface from white content panels.
- **Primary navigation:** deep navy identity and clear text labels.
- **Primary action:** blue, reserved for execution/commit actions.
- **Success:** teal/green, used only for completed pipeline states.
- **Warning:** amber, used for validation/execution attention.
- **Typography:** system/Inter-style sans-serif; strong page titles, compact uppercase section labels, readable body copy.
- **Surfaces:** restrained borders, small shadows, consistent 12–20px radii and ample spacing.
- **Data visualization:** white plotting surface with muted axes and legible labels.

## Interaction rules

- Never make users upload the same data twice for two separate agents.
- Never imply Data Analysis is ready until an active Gold pipeline manifest exists.
- Keep technical metadata and raw JSON in accordions.
- Always show whether an action used the CrewAI tool or a deterministic fallback.
- Always show pipeline ID and row counts after a successful pipeline run.
- Preserve the SQL query and bounded read-only semantics for user inspection.
- Explain errors in the context of the current task, not as raw tracebacks alone.
- Treat the Gradio share link as public; this prototype is for synthetic/training data only.

## Responsive behavior

- Four KPI cards collapse to two columns on narrower screens.
- Two-agent cards stack to one column.
- Five pipeline stages wrap to smaller grids.
- Topbar chips wrap and brand/navigation remain legible.
- Main task forms stack vertically where there is insufficient width.

## UX acceptance checklist

- [ ] First-time user can identify the active dataset and pipeline state from Overview.
- [ ] First-time user can run the sample pipeline without preparing an input file.
- [ ] A user can upload a CSV and provide optional business context.
- [ ] Successful Engineering run shows quality metrics, Silver preview and pipeline ID.
- [ ] Overview refreshes immediately after a successful Engineering run.
- [ ] Analysis screen contains no second dataset upload.
- [ ] Analysis screen explains how the same Gold dataset is used.
- [ ] Empty question and missing pipeline states are understandable.
- [ ] SQL is inspectable but not the dominant part of the Analysis UX.
- [ ] The two agent roles and their responsibilities are visually distinct.
- [ ] No external provider/hosted-agent dependency is present.
- [ ] Layout remains usable on a narrow mobile viewport.

## Non-goals for this iteration

- Multi-user identity and role management.
- Persistent workspace across Colab runtime resets.
- Long-running job orchestration, cancellation and resumable executions.
- Production design system package or a separately hosted frontend.
- Production-grade accessibility audit and screen-reader certification.
