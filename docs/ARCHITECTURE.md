# Architecture — Data Agent Platform Studio

## High-level design

The application exposes two distinct user experiences over a shared local Hugging Face model.

- **Data Engineering Agent:** CSV upload → CrewAI task/tool orchestration → deterministic Python profile/clean/load tool → cleaned CSV + SQLite staging table.
- **Data Analysis Agent:** business question → CrewAI schema-inspection task → SQL generation → read-only SQLite tool → result table + chart + natural-language summary.

## Component map

~~~mermaid
flowchart LR
    U[User] --> UI[Gradio Studio]
    UI --> DE[CrewAI Data Engineering Crew]
    UI --> DA[CrewAI Data Analysis Crew]
    DE <--> HF[Qwen2.5 1.5B local model]
    DA <--> HF
    DE --> ETL[Profile / clean / validate / load tools]
    ETL --> CSV[Clean CSV]
    ETL --> STG[(SQLite staging_upload)]
    DA --> SCHEMA[Schema tool]
    DA --> SQL[Read-only SQL tool]
    SQL --> DB[(Per-run SQLite sales DB)]
    SQL --> OUT[DataFrame / Plot / Summary]
~~~

## Data Engineering tool contract

Input: one CSV file copied to the application workspace.

Operations performed by Python:
1. Read and profile the CSV.
2. Optionally trim text, drop fully empty rows and remove exact duplicates.
3. Write a cleaned CSV and JSON quality report.
4. Load the cleaned frame to the SQLite table named staging_upload.

The LLM is not allowed to write files or execute arbitrary Python. CrewAI can call one bounded tool; if the small model does not call it, the application runs the same deterministic Python pipeline and reports the fallback mode.

## Data Analysis tool contract

The upload (or bundled demo dataset) is loaded to a per-run SQLite database with a sales table.

CrewAI is instructed to:
1. Inspect the table schema through the schema tool.
2. Generate one SELECT query.
3. Call the read-only SQL tool.
4. Summarize only returned rows.

SQL controls:
- only SELECT or WITH ... SELECT is accepted;
- multiple statements and SQL comments are rejected;
- database is opened in read-only URI mode;
- SQLite authorizer permits read/select/function operations only;
- results are capped at 100 rows.

When CrewAI does not complete a SQL tool call, a small deterministic SELECT fallback is used and explicitly labeled in the UI.

## Runtime and model

- Google Colab GPU runtime; use a T4 where available.
- Gradio serves a web UI using a temporary share URL.
- Qwen/Qwen2.5-1.5B-Instruct is loaded locally through Transformers with automatic device mapping and FP16 on CUDA.
- CrewAI receives the local Transformers model through a custom BaseLLM adapter.
- Model weights are loaded lazily on first agent execution.

## Security boundaries and limitations

This is a training prototype, not a production multi-tenant service. The Gradio share URL should be treated as public to anyone who obtains it. Do not upload confidential customer datasets. The local workspace is ephemeral in Colab. No SQL write access, shell tool, arbitrary Python execution tool, or cloud credential is exposed to the agents.
