# Data Agent Platform Studio

**Pipeline-first · Google Colab T4 · Gradio · CrewAI · Hugging Face · Genie · Cortex**

A web Studio for a **single data pipeline consumed by multiple agent providers**. The Data Engineering Agent prepares the active dataset once; the Data Analysis UX can query it using an independent CrewAI/Hugging Face agent or invoke a configured native Databricks Genie or Snowflake Cortex Agent.

## Product model

~~~mermaid
flowchart LR
    SRC[CSV / Data Source] --> DE[Data Engineering Agent]
    DE --> BR[Bronze]
    BR --> ETL[Clean / Validate / Transform]
    ETL --> SI[Silver]
    SI --> GO[Gold dataset + manifest]
    GO --> ROUTER[Agent provider selector]
    ROUTER --> LOCAL[CrewAI + HF]
    ROUTER --> GENIE[Databricks Genie]
    ROUTER --> CORTEX[Snowflake Cortex]
    LOCAL --> OUT[Answer / SQL / Results]
    GENIE --> OUT
    CORTEX --> OUT
~~~

The scenarios are related through the shared pipeline manifest and canonical Gold asset. **Data Analysis no longer uploads or builds a second unrelated dataset.**

## Colab quick start

[Open the Colab T4 notebook](https://colab.research.google.com/github/msellamiTN/data-agent-platform-studio/blob/feature/colab-crewai-dual-agent-studio/notebooks/Colab_T4_Gradio_Studio.ipynb)

1. Select **Runtime → Change runtime type → T4 GPU** when available.
2. Run the notebook cells in order.
3. The notebook clones this development branch, installs dependencies, checks CUDA, and starts Gradio.
4. Open the temporary Gradio share URL.
5. Run **01 · Shared Data Pipeline** first; then ask questions in **02 · Data Analysis Agent**.

T4 availability depends on Colab quota and runtime availability. The notebook checks the actual GPU assigned. Qwen is loaded lazily on the first local-agent execution.

## The shared pipeline

The Colab runtime creates one active dataset contract at `workspace/pipeline/current/`:

~~~text
workspace/pipeline/current/
├── bronze/source.csv
├── silver/cleaned.csv
├── gold/warehouse.sqlite
├── metadata/quality_report.json
└── manifest.json
~~~

Gold SQLite contains:
- `silver_data`: canonical cleaned dataset queried by the independent Data Analysis Agent.
- `staging_upload`: alias for introductory ETL exercises.

The manifest records pipeline ID, dataset, status, schema, row counts and artifact locations. Every local analysis run uses the active pipeline ID and Gold database.

## Data Engineering Agent

Upload any CSV (maximum 25 MB) or use the bundled Retail Sales example.

**Flow:** Source → Bronze → profile/clean/validate → Silver → Gold → Manifest.

Python performs deterministic trimming, exact-duplicate removal, empty-row removal, report creation and SQLite loading. CrewAI orchestrates a fixed tool and summarizes the returned report; a deterministic fallback is clearly labeled if the small model does not invoke the tool.

## Data Analysis Agent — provider selector

Choose one provider in the analysis UX:

| Provider | What it does | Data it queries |
|---|---|---|
| Independent · CrewAI + Hugging Face | Inspects schema, generates SQL, calls a read-only tool, summarizes | Active Colab Gold SQLite table `silver_data` |
| Databricks Genie | Natural-language analytics over curated Unity Catalog assets | Configured Genie Agent / Unity Catalog tables |
| Snowflake Cortex | Managed agent orchestration across configured tools such as Cortex Analyst and Cortex Search | Configured Snowflake agent, semantic views and tool resources |

### Provider boundaries

Genie and Cortex are **not interchangeable ETL engines**:
- Databricks Genie Agents are native analytics agents over Unity Catalog assets. Databricks data engineering capabilities belong to Lakeflow (Connect, Pipelines, Jobs); Genie Code provides engineering-oriented assistance.
- Snowflake Cortex Agents orchestrate configured tools in Snowflake, including Cortex Analyst, Cortex Search, code execution, Data-to-Chart, custom tools and MCP connectors.

To compare Genie and Cortex on the same dataset, publish the Silver/Gold data to the respective platform and configure both agents with equivalent tables, business definitions, semantic metadata, permissions and instructions. This prototype does **not** silently replicate Colab files to those platforms.

## Native provider configuration

Store these values in **Colab Secrets** or your local environment. Do not commit credentials.

Databricks Genie:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_GENIE_SPACE_ID`

Snowflake Cortex:
- `SNOWFLAKE_HOST`
- `SNOWFLAKE_TOKEN`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `SNOWFLAKE_AGENT_NAME`

The application displays only whether configuration is complete and which variable names are missing, never credential values.

## Independent data scenario — Retail Sales

The demo file `data/sales_demo.csv` is processed through the exact same shared pipeline as a user-uploaded CSV. Example question:

> What is the revenue by region in the first quarter?

The local CrewAI agent receives the actual schema from Gold, then queries `silver_data` with a SELECT-only SQL tool. For any other dataset, it receives that dataset's own columns rather than hard-coded sales fields.

## Model and implementation

- **Local model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), through Transformers.
- **Orchestration:** [CrewAI](https://github.com/crewAIInc/crewAI).
- **UI:** [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks).
- **Databricks:** [Genie Agents](https://developers.databricks.com/docs/lakehouse/genie) and [Conversation API](https://docs.databricks.com/api/genie/v1/conversation).
- **Snowflake:** [Cortex Agents](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents) and [Agent Run API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-run).

## Local setup

~~~bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
~~~

Open `http://localhost:7860`. For an optional Gradio share URL, set `GRADIO_SHARE=true`.

## Repository structure

~~~text
.
├── app.py
├── studio/
│   ├── config.py
│   ├── etl.py
│   ├── integrations.py
│   ├── llm.py
│   ├── scenarios.py
│   └── sql.py
├── data/sales_demo.csv
├── docs/ARCHITECTURE.md
├── notebooks/Colab_T4_Gradio_Studio.ipynb
├── tests/
├── .env.example
├── requirements.txt
└── requirements-dev.txt
~~~

## Tests

~~~bash
pip install -r requirements-dev.txt
pytest -q
~~~

Tests cover the deterministic shared pipeline and read-only SQL policy. They do not require cloud credentials or make live calls to Genie/Cortex. A live Colab T4 run and native-provider credentials are required for end-to-end acceptance.

## Safety and scope

- Training prototype; not a production multi-tenant deployment.
- Treat the Gradio share URL as public to anyone who obtains it.
- Do not upload confidential customer data.
- Local analysis allows one SELECT/WITH statement, opens SQLite read-only, applies an authorizer and caps results at 100 rows.
- Databricks/Snowflake remote agents run under their native platform permissions and configured tool resources.
- Credentials remain outside Git. Colab workspaces are temporary.
- Provider-native data publishing and provider-specific structured-result normalization are explicit next integration milestones.
