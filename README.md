# Data Agent Platform Studio

**Colab + T4 GPU · Gradio · CrewAI · Hugging Face local model**

A compact web studio with two separate user experiences and two CrewAI agents:

1. **Data Engineering Agent** — upload a CSV, profile it, clean text, remove duplicates/empty rows, export a cleaned file and load it into a SQLite staging table.
2. **Data Analysis Agent** — upload a sales CSV (or use the demo dataset), ask a business question, inspect schema, generate SQL, execute it read-only, and receive a result table, chart and summary.

## Run in Google Colab

[Open the Colab T4 notebook](https://colab.research.google.com/github/msellamiTN/data-agent-platform-studio/blob/feature/colab-crewai-dual-agent-studio/notebooks/Colab_T4_Gradio_Studio.ipynb)

1. In Colab, select **Runtime → Change runtime type → T4 GPU** when available.
2. Run the notebook cells in order.
3. The notebook clones this branch, installs the dependencies, checks the GPU, and starts the Gradio web application.
4. Open the temporary Gradio share URL printed by Colab.
5. Use either the **Data Engineering Agent** tab or the **Data Analysis Agent** tab.

A T4 is availability/quota dependent; the notebook checks the GPU actually assigned by Colab. The model is downloaded lazily the first time an agent runs. No OpenAI API key or Hugging Face token is required for the public model checkpoint.

## Model and orchestration

- **Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), loaded locally with Transformers and automatic device mapping; FP16 is used on CUDA.
- **Agent framework:** [CrewAI](https://github.com/crewAIInc/crewAI), with one CrewAI agent/crew per scenario.
- **Web UX:** [Gradio Blocks](https://gradio.app/docs/gradio/blocks), with distinct inputs, actions and outputs for each scenario.
- **Colab:** [Google Colab](https://colab.research.google.com/).

The Hugging Face model is exposed to CrewAI through a small custom BaseLLM adapter. The adapter uses CrewAI's text-based tool loop rather than claiming native function calling. The small model may occasionally fail to invoke a tool; in that case the application runs a clearly labeled deterministic Python/SQL fallback so the lab remains testable.

## Architecture

~~~mermaid
flowchart LR
    U[User] --> G[Gradio Studio]
    G --> DE[CrewAI Data Engineering Agent]
    G --> DA[CrewAI Data Analysis Agent]
    DE <--> HF[Qwen 1.5B local model]
    DA <--> HF
    DE --> T1[CSV profile / clean / stage tool]
    T1 --> C[Clean CSV + SQLite staging_upload]
    DA --> T2[Schema inspection tool]
    T2 --> T3[Read-only SQL tool]
    T3 --> DB[(Per-run SQLite sales database)]
    DB --> R[Table + Chart + Summary]
~~~

## Scenario 1 — Data Engineering

**Input:** CSV file (maximum 25 MB).

**Flow:** Upload → profile → trim text (optional) → remove fully empty rows (optional) → remove exact duplicates (optional) → cleaned CSV + JSON quality report → SQLite staging table.

The Python ETL tool does the actual file and database work. CrewAI orchestrates the tool call and summarizes the verified report. The staging table is named `staging_upload` in `workspace/lakehouse.sqlite`.

## Scenario 2 — Data Analysis

**Input:** a sales CSV with a `region` column and either a `revenue` column or both `quantity` and `unit_price`. The optional date and product columns are used when present. When no upload is provided, the bundled `data/sales_demo.csv` is used.

**Flow:** Business question → schema inspection → SQL generation → read-only SQL tool → result table + chart + LLM summary.

The database is isolated per run. SQL controls accept only one SELECT or WITH...SELECT statement, reject comments and multiple statements, open SQLite in read-only mode, apply a SQLite authorizer, and cap results at 100 rows.

## Local setup

Python 3.10–3.13 is supported by the selected CrewAI release range. A CUDA GPU is recommended for practical local model inference.

~~~bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
~~~

Open `http://localhost:7860`. To expose a Gradio share URL from a trusted environment, set `GRADIO_SHARE=true` before starting the app.

## Repository map

~~~text
.
├── app.py
├── data/
│   └── sales_demo.csv
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/
│   └── Colab_T4_Gradio_Studio.ipynb
├── studio/
│   ├── config.py
│   ├── etl.py
│   ├── llm.py
│   ├── scenarios.py
│   └── sql.py
├── tests/
│   ├── test_etl.py
│   └── test_sql_guard.py
├── requirements.txt
└── requirements-dev.txt
~~~

## Tests

~~~bash
pip install -r requirements-dev.txt
pytest -q
~~~

The tests cover deterministic ETL transformations and the read-only SQL guard. They do not download the Hugging Face model or claim that a Colab T4 runtime was exercised.

## Safety and scope

- This is an educational prototype, not a production multi-tenant platform.
- A Gradio share URL is accessible to anyone who obtains it; do not upload sensitive or confidential datasets.
- Colab runtimes are temporary; download outputs you need to keep.
- Uploaded CSVs are limited to 25 MB.
- The Data Engineering agent is restricted to a fixed CSV pipeline tool; it does not have shell or arbitrary Python access.
- The Data Analysis agent can only use the schema tool and bounded read-only SQL tool; writes, DDL and multiple statements are blocked.
- The local model proposes and explains; deterministic Python performs the data operations and SQL policy enforcement.
