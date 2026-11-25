# Data Agent Platform Studio

**Independent data-agent platform · CUDA / CPU · Gradio · CrewAI · Hugging Face**

A self-contained Studio for two specialized AI agents that operate on **one shared data pipeline**. There are no external hosted-agent connections, vendor agent APIs or remote orchestration dependencies.

## Product concept

~~~mermaid
flowchart LR
    SRC[CSV / File] --> DE[Data Engineering Agent]
    DE --> B[Bronze: source]
    B --> ETL[Profile / Clean / Validate]
    ETL --> S[Silver: curated data]
    S --> G[Gold: SQLite dataset]
    G --> M[Manifest + schema + quality + semantic context]
    M --> DA[Data Analysis Agent]
    DA --> SQL[Read-only SQL tool]
    SQL --> RES[Table / Chart / Explanation]
~~~

**One dataset, one lineage, two agents.** The Data Analysis Agent cannot load a second unrelated file. It consumes the active Gold dataset and its metadata from the Data Engineering run.

## Run in Google Colab

[Open the Colab T4 notebook](https://colab.research.google.com/github/msellamiTN/data-agent-platform-studio/blob/feature/colab-crewai-dual-agent-studio/notebooks/Colab_T4_Gradio_Studio.ipynb)

1. Choose **Runtime → Change runtime type → T4 GPU** when available; a CPU runtime also works.
2. Run the notebook cells in order.
3. The notebook detects CUDA; otherwise it selects CPU inference automatically, then clones this branch, installs dependencies and launches the Gradio Studio.
4. Open the temporary Gradio URL.
5. Run **01 · Data Engineering Agent** first.
6. Then run **02 · Data Analysis Agent** against the active dataset.

The notebook checks the actual runtime. CUDA is preferred when available; CPU-only inference uses FP32 and is slower, with higher RAM requirements. Set `HF_DEVICE=cpu` to force CPU or `HF_DEVICE=cuda` to require a GPU. The model is loaded lazily on its first agent execution. The install requirements pin Pillow 11.3.0 to avoid a known `_Ink` import mismatch in some Colab images; if Pillow was already imported before installation, restart the runtime before importing Gradio or CrewAI.

## Runtime and core stack

| Layer | Technology |
|---|---|
| Notebook/runtime | Google Colab CPU or GPU (T4 target) |
| Web UX | Gradio Blocks |
| Agent orchestration | CrewAI |
| Local language model | Qwen/Qwen2.5-1.5B-Instruct |
| Model runtime | Hugging Face Transformers + Accelerate (FP16 CUDA / FP32 CPU) |
| Data processing | Python + Pandas |
| Local analytical store | SQLite |
| Visualization | Plotly |

Both CrewAI agents use the same local Hugging Face model adapter. CUDA uses FP16; CPU uses FP32 and does not require a GPU. CPU inference is supported but may be substantially slower and the 1.5B model needs several GB of available RAM. No external model API key is required for the public model checkpoint. The agent tools are deterministic Python functions with bounded permissions.

## Scenario 1 — Data Engineering Agent

**Purpose:** transform an uploaded source file into a documented, query-ready dataset.

**Flow:** Source → Bronze → Profile → Clean → Validate → Silver → Gold → Manifest.

The Gradio UX allows the user to:
- upload a CSV (up to 25 MB, 250,000 rows and 250 columns) or select the bundled Retail Sales example;
- reject empty or duplicate column names before loading;
- constrain business descriptions and analysis questions to configured character limits;
- trim text fields;
- remove exact duplicate records;
- remove fully empty rows;
- provide an optional business description.

The Python pipeline performs the actual data operations. CrewAI orchestrates the fixed pipeline tool and summarizes the resulting quality report.

### Shared pipeline artifacts

Each execution is stored as an immutable run under `workspace/pipeline/runs/<pipeline_id>/`. The active manifest at `workspace/pipeline/current/manifest.json` is atomically replaced only after every artifact is complete:

~~~text
workspace/pipeline/
├── current/
│   └── manifest.json          # atomic pointer to the active run
└── runs/
    └── <pipeline_id>/
        ├── bronze/source.csv
        ├── silver/cleaned.csv
        ├── gold/warehouse.sqlite
        ├── metadata/quality_report.json
        └── manifest.json      # immutable run manifest
~~~

A later run does not overwrite a database being read by an earlier analysis request. Failed runs do not replace the previous active manifest. Runs are retained until explicitly removed; persistent retention and cleanup policies are not yet implemented.

Gold SQLite exposes:
- `silver_data`: canonical curated table consumed by the Data Analysis Agent;
- `staging_upload`: compatibility alias for introductory ETL exercises.

The manifest contains a pipeline ID, dataset name, run status, paths, row counts, column profile, semantic catalog and user-supplied business context.

## Scenario 2 — Data Analysis Agent

**Purpose:** answer business questions using the same dataset prepared by the Data Engineering Agent.

**Flow:** Business Question → Pipeline Context → Schema Inspection → SQL Generation → Read-only Execution → Table / Chart / Explanation.

The analysis agent receives:
- the active pipeline ID and dataset name;
- the exact Silver/Gold table and column names;
- column types, null counts, distinct counts and sample values;
- candidate dimensions, measures and date fields;
- the optional business description from the Data Engineering screen.

The agent must inspect the schema before generating SQL. It cannot query an assumed table or a different dataset.

### SQL controls

- Only one `SELECT` or `WITH ... SELECT` statement is accepted.
- SQL comments and multiple statements are rejected.
- SQLite is opened with read-only mode.
- A SQLite authorizer permits reads only from the canonical `silver_data` table and blocks writes, schema access and unsafe functions.
- Query execution is interrupted after the configured `MAX_SQL_EXECUTION_SECONDS` budget (default: 5 seconds).
- Query output is capped at 100 rows.
- Common row-count, column-count and distinct-count intents are planned against real schema names; an ambiguous count request returns a clarification instead of silently showing a preview.

If the small local model does not complete its tool call, the application uses a clearly labeled Python fallback. The fallback supports only explicitly implemented schema-aware query patterns; a row preview is labeled as a preview and is never presented as an aggregate answer. Count questions generate `COUNT` / `COUNT(DISTINCT ...)` queries against a matching real column. The UI distinguishes CrewAI tool execution, deterministic query-plan verification and preview-only fallback. A verified aggregate answer is composed from the actual SQL result, not from unverified model narration.

## Functional patterns implemented locally

The Studio implements useful data-agent capabilities without connecting to an external agent service:

| Capability | Local implementation |
|---|---|
| Data preparation | Profile, clean, validate, deduplicate and load |
| Dataset context | Shared pipeline manifest |
| Metadata / semantic context | Column profile, candidate dimensions/measures/dates and business description |
| Natural-language analysis | CrewAI Data Analysis Agent using local Qwen |
| Text-to-SQL | Schema-grounded SQL generation |
| Tool orchestration | CrewAI tools with bounded Python functions |
| Query governance | Read-only SQL policy, row limit and workspace boundary |
| Results | DataFrame, Plotly chart and explanation |
| Traceability | Pipeline ID, quality report and Bronze/Silver/Gold paths |

## Local setup

~~~bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
# Edit .env for HF_DEVICE, HF_CPU_THREADS and other local settings.
python app.py
~~~

Open `http://localhost:7860`. The application loads `.env` from the repository root without overriding variables already set by the shell or notebook. CPU is selected automatically when CUDA is unavailable. To force CPU, set `HF_DEVICE=cpu`; to require CUDA, set `HF_DEVICE=cuda`; `HF_DEVICE=auto` prefers CUDA and falls back to CPU. On CPU, `HF_CPU_THREADS=4` controls the PyTorch thread count. To create a temporary Gradio share URL from a trusted environment, set `GRADIO_SHARE=true`.

## Repository structure

~~~text
.
├── app.py
├── studio/
│   ├── config.py
│   ├── etl.py
│   ├── llm.py
│   ├── runtime.py
│   ├── scenarios.py
│   └── sql.py
├── data/sales_demo.csv
├── docs/ARCHITECTURE.md
├── notebooks/Colab_T4_Gradio_Studio.ipynb
├── tests/
│   ├── test_etl.py
│   ├── test_llm_runtime.py
│   └── test_sql_guard.py
├── requirements.txt
└── requirements-dev.txt
~~~

## Professional Studio UX

The Gradio interface uses a focused workspace layout rather than a notebook-style form:

- **Overview:** active pipeline monitor, Bronze/Silver/Gold lifecycle, dataset KPIs and the two-agent hand-off.
- **Data Engineering:** source upload, optional business description, explicit quality controls, run summary, quality report, Silver preview and manifest.
- **Data Analysis:** question-first interaction, question ideas derived from the active dataset's real columns, answer, SQL disclosure and result table/visualization.
- **Platform:** local execution stack and clear model/data safety boundaries.

The visual system uses a restrained navy/blue/teal palette, clear typographic hierarchy, compact status chips, metric cards, progressive disclosure for technical details and responsive layouts for narrower screens. It remains a Gradio application and does not introduce an external UI service.

See [the UX design specification](docs/UX-DESIGN.md) for the layout and interaction principles.

## Security and production readiness

This is a production-minded prototype baseline, **not a production-ready multi-user service**. It has no user authentication, tenant isolation, application-managed encryption at rest, automated data retention, or operational audit/alerting. Use synthetic or non-sensitive data only. See [SECURITY.md](SECURITY.md) and the [production-readiness release gates](docs/PRODUCTION-READINESS.md) before any deployment beyond a controlled demo.

The current Gradio dependency range constrains Pillow below 13.0.0, while Pillow's 13.0.0 release contains upstream security fixes that are not backported to earlier releases. The Colab notebook currently installs the compatible Pillow 11.3.0 line to avoid the reported import mismatch; this compatibility/security trade-off is explicitly recorded as a production release blocker. Do not treat the dependency set as security-hardened until Gradio's constraint and the Pillow floor are reconciled and the complete environment passes compatibility and vulnerability checks.

## Tests

~~~bash
pip install -r requirements-dev.txt
pytest -q
~~~

Tests cover immutable run publication, active-manifest switching, failed-run isolation, read-only table scope, SQL timeout enforcement, bounded semantic query plans and runtime-device selection. They do not download the Hugging Face model or claim that a Colab T4 runtime was exercised.

## Safety and scope

- This is an educational prototype, not a production multi-tenant service.
- Anyone who obtains the Gradio share URL may be able to access the demo; use synthetic data only.
- Colab storage is temporary; download outputs you need to keep.
- Uploads are limited to 25 MB, 250,000 parsed rows and 250 columns. Column names must be non-empty and unique case-insensitively. Business context and analysis questions have configurable character limits.
- The Data Engineering Agent has a fixed ETL tool; it has no shell or arbitrary Python execution tool.
- The Data Analysis Agent has schema-inspection and read-only SQL tools; it cannot write to the database.
- The model proposes and explains; Python performs the data operations and enforces policy.
