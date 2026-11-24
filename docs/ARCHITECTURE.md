# Architecture — Pipeline-first Data Agent Platform Studio

## Product rule

**The two scenarios are views over one data pipeline, not two unrelated demos.** A Data Engineering run creates the active dataset contract; Data Analysis consumes that contract or routes the same business question to a configured platform-native agent.

## One shared data pipeline

~~~mermaid
flowchart LR
    S[CSV / File / Export] --> DE[CrewAI Data Engineering Agent]
    DE --> B[Bronze: immutable source copy]
    B --> T[Python ETL: profile, clean, deduplicate]
    T --> SL[Silver: cleaned.csv]
    SL --> G[Gold: warehouse.sqlite]
    G --> M[Manifest + schema + quality report]
    M --> L[Agent routing layer]
    L --> LC[Independent CrewAI + HF]
    L --> DG[Databricks Genie Agent]
    L --> SC[Snowflake Cortex Agent]
    LC --> R[Normalized answer / SQL / result]
    DG --> R
    SC --> R
    R --> V[Gradio table / chart / explanation]
~~~

### Active pipeline contract

The Colab implementation writes the current run under `workspace/pipeline/current/`:

~~~text
workspace/pipeline/current/
├── bronze/
│   └── source.csv
├── silver/
│   └── cleaned.csv
├── gold/
│   └── warehouse.sqlite
├── metadata/
│   └── quality_report.json
└── manifest.json
~~~

Gold SQLite exposes:
- `silver_data` — canonical table consumed by the independent Data Analysis Agent.
- `staging_upload` — compatibility alias for ETL teaching exercises.

The manifest records pipeline ID, dataset name, artifact paths, row counts, schema, status and creation time. Both local agents read the same active manifest and Gold table.

## Agent roles and provider boundaries

| Role | Independent implementation | Databricks | Snowflake |
|---|---|---|---|
| Data Engineering | CrewAI + Python/Pandas pipeline tools | Lakeflow Connect, Lakeflow Pipelines and Lakeflow Jobs; Genie Code assists engineering work | Snowpark, SQL, Tasks and Dynamic Tables |
| Data Analysis | CrewAI + Qwen2.5-1.5B + read-only SQL | Genie Agent over curated Unity Catalog assets | Cortex Agent over configured Cortex Analyst / Cortex Search and other tools |

**Do not treat every vendor data agent as the same API.**
- Databricks Genie Agents are natural-language analytics over Unity Catalog datasets, knowledge-store metadata, example SQL and a SQL warehouse. They generate SQL and return query results. They are not the generic ETL runner.
- Databricks data engineering is a separate Lakeflow family (Connect, Pipelines, Jobs); Genie Code provides engineering-oriented assistance.
- Snowflake Cortex Agents are managed orchestrators that can use Cortex Analyst semantic views, Cortex Search, code execution, Data-to-Chart, custom tools and MCP connectors.
- The independent CrewAI agent is the vendor-neutral teaching/demo path and directly queries the Gold SQLite table created in Colab.

Official product references:
- [Databricks Genie Agents](https://developers.databricks.com/docs/lakehouse/genie)
- [Databricks Genie Conversation API](https://docs.databricks.com/api/genie/v1/conversation)
- [Databricks Lakeflow data engineering](https://docs.databricks.com/aws/en/data-engineering/)
- [Snowflake Cortex Agents](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents)
- [Snowflake Cortex Agent Run API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-run)

## Provider adapters

The Gradio analysis selector supports three provider choices:

1. **Independent · CrewAI + Hugging Face** — active pipeline Gold SQLite, schema inspection, SELECT-only SQL tool, maximum 100 rows.
2. **Databricks Genie** — calls the Genie Conversation API using `DATABRICKS_HOST`, `DATABRICKS_TOKEN` and `DATABRICKS_GENIE_SPACE_ID`; starts a conversation, polls the asynchronous message and returns Genie text/SQL.
3. **Snowflake Cortex** — calls the Cortex Agent Run API using `SNOWFLAKE_HOST`, `SNOWFLAKE_TOKEN`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA` and `SNOWFLAKE_AGENT_NAME`.

The native providers execute against **their own configured, governed data assets**. The Colab CSV is not automatically uploaded to Unity Catalog or Snowflake. To compare the providers on equivalent data, publish the Silver/Gold dataset to both platforms and curate matching tables, metrics, semantic views, permissions and agent instructions.

This initial adapter layer returns native provider text and SQL where available. A provider-specific structured-result normalizer and native data-publishing connectors are extension points, not claims of completed functionality.

## Independent scenario example — Retail Sales

1. Upload `data/sales_demo.csv` or another CSV through the Data Engineering tab.
2. The pipeline writes Bronze source, Silver cleaned data and Gold SQLite.
3. Ask: **“What is revenue by region in the first quarter?”**
4. Run with the independent CrewAI provider. The agent inspects the actual `silver_data` schema, writes a single SELECT query and calls the read-only tool.
5. Switch to Genie or Cortex only after configuring those agents against corresponding published retail-sales tables and semantics.

A new dataset can be used without changing the orchestration code: the local agent receives the schema from the active pipeline rather than relying on hard-coded `sales` columns.

## Data Engineering tool contract

Python performs the deterministic pipeline:
1. Copy the source into Bronze.
2. Profile the CSV.
3. Optionally trim text, remove fully empty rows and exact duplicates.
4. Write Silver cleaned CSV and a quality report.
5. Load Gold SQLite tables and update the active manifest.

CrewAI calls one bounded tool and summarizes its verified output. If the small model fails to invoke the tool, the deterministic pipeline runs as a clearly labeled fallback.

## Data Analysis controls

For the independent local provider:
- only one SELECT or WITH ... SELECT statement is accepted;
- SQL comments and multiple statements are rejected;
- SQLite is opened in read-only URI mode;
- a SQLite authorizer permits SELECT, READ and safe function operations only;
- results are capped at 100 rows.

Databricks and Snowflake native providers apply their own identity, privileges, semantic metadata, warehouse and tool-governance rules. The application does not replace those provider-side controls.

## Runtime, UX and model

- Google Colab GPU runtime; T4 when available.
- Gradio Blocks with a shared pipeline screen, an analysis screen and a provider setup screen.
- Qwen/Qwen2.5-1.5B-Instruct is loaded locally through Transformers, with automatic device placement and FP16 on CUDA.
- CrewAI receives the local Transformers model through a custom BaseLLM adapter.
- Native Genie/Cortex API calls are remote; the local Qwen model is used only by the independent CrewAI provider.

## Security and limitations

- This is a lab prototype, not a production multi-tenant service.
- A Gradio share URL is public to anyone who obtains it; do not upload confidential data.
- Colab workspace artifacts are ephemeral.
- Native provider credentials must be supplied through Colab Secrets or environment variables; they are never stored in the repository or displayed in the UI.
- Remote agents must be pre-configured against published platform data. Automatic cross-cloud replication/publishing is not implemented in this iteration.
