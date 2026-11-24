from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from .config import SAMPLE_SALES_CSV, WORKSPACE_DIR
from .etl import copy_upload_to_workspace, run_csv_pipeline
from .integrations import run_provider_analysis
from .llm import get_local_llm
from .sql import (
    describe_pipeline_schema,
    execute_readonly_query,
    fallback_pipeline_sql,
    get_current_pipeline,
)


LOCAL_PROVIDER = "Independent · CrewAI + Hugging Face"


def _make_agent_llm():
    return get_local_llm()


def run_data_engineering(
    upload_path: str,
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
) -> dict[str, Any]:
    """Run the Data Engineering Agent and publish the shared pipeline manifest."""
    run_dir = WORKSPACE_DIR / f"uploads_{uuid.uuid4().hex[:10]}"
    workspace_csv = copy_upload_to_workspace(upload_path, run_dir)
    captured: dict[str, Any] = {}

    @tool("run_shared_data_pipeline")
    def run_shared_data_pipeline() -> str:
        """Run the shared Bronze-to-Silver-to-Gold pipeline and register its manifest."""
        report = run_csv_pipeline(
            workspace_csv,
            trim_text=trim_text,
            drop_duplicates=drop_duplicates,
            drop_empty_rows=drop_empty_rows,
        )
        captured["report"] = report
        return json.dumps(report, ensure_ascii=False, default=str)

    crew_summary = ""
    crew_error = None
    try:
        agent = Agent(
            role="Data Engineering Agent",
            goal=(
                "Execute the shared data pipeline through its fixed tool, then "
                "explain the verified quality and lineage report."
            ),
            backstory=(
                "You are a careful data engineer. The shared pipeline writes Bronze, "
                "Silver and Gold artifacts. Never invent row counts or claim that "
                "data was published to Databricks or Snowflake unless a provider says so."
            ),
            llm=_make_agent_llm(),
            tools=[run_shared_data_pipeline],
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )
        task = Task(
            description=(
                "Call run_shared_data_pipeline exactly once. Summarize the source, "
                "row counts, duplicate removals, quality report and active Gold table. "
                "State that the pipeline manifest is shared with the Data Analysis "
                "scenario. Do not claim that data was published to external platforms."
            ),
            expected_output=(
                "A concise report based only on the pipeline tool output, including "
                "pipeline_id, Bronze/Silver/Gold artifact paths and active dataset."
            ),
            agent=agent,
        )
        crew_result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
        crew_summary = str(crew_result)
    except Exception as exc:
        crew_error = f"{type(exc).__name__}: {exc}"

    execution_mode = "crewai_tool"
    if "report" not in captured:
        captured["report"] = run_csv_pipeline(
            workspace_csv,
            trim_text=trim_text,
            drop_duplicates=drop_duplicates,
            drop_empty_rows=drop_empty_rows,
        )
        execution_mode = "python_fallback"

    report = captured["report"]
    cleaned_file = Path(report["cleaned_csv"])
    preview = pd.read_csv(cleaned_file).head(25)
    if not crew_summary:
        crew_summary = (
            f"Pipeline {report['pipeline_id']} READY: {report['rows_before']} lignes "
            f"en entrée, {report['rows_after']} après nettoyage, "
            f"{report['duplicates_removed']} doublon(s) retiré(s). "
            "Le résultat Gold est maintenant partagé avec le scénario Data Analysis."
        )

    return {
        "summary": crew_summary,
        "report": report,
        "cleaned_file": str(cleaned_file),
        "preview": preview,
        "execution_mode": execution_mode,
        "crew_error": crew_error,
    }


def run_data_analysis(question: str, provider: str = LOCAL_PROVIDER) -> dict[str, Any]:
    """Run analysis against the current pipeline or a configured native data agent."""
    if not question or not question.strip():
        raise ValueError("Enter a business question before running the agent.")

    # Native providers query their own already-published governed data assets.
    # They do not read the ephemeral Colab SQLite file.
    if provider != LOCAL_PROVIDER:
        remote = run_provider_analysis(provider, question.strip())
        rows = remote.get("rows") or []
        dataframe = pd.DataFrame(rows)
        warnings = remote.get("warnings") or []
        summary = remote.get("answer", "")
        if remote.get("note"):
            summary += "\n\n" + remote["note"]
        if warnings:
            summary += "\n\nProvider warnings: " + json.dumps(warnings, ensure_ascii=False)
        return {
            "provider": remote["provider"],
            "summary": summary,
            "sql": remote.get("sql", ""),
            "rows": rows,
            "dataframe": dataframe,
            "database": "",
            "execution_mode": "native_provider",
            "conversation_id": remote.get("conversation_id", ""),
            "crew_error": None,
            "provider_note": remote.get("note", ""),
        }

    manifest = get_current_pipeline()
    database_path = Path(manifest["database"]).resolve()
    captured: dict[str, Any] = {}

    @tool("inspect_active_pipeline")
    def inspect_active_pipeline() -> str:
        """Inspect schema and sample rows from the active Gold pipeline table."""
        schema = describe_pipeline_schema()
        return json.dumps(schema, ensure_ascii=False, default=str)

    @tool("query_active_pipeline_read_only")
    def query_active_pipeline_read_only(sql: str) -> str:
        """Execute one SELECT query against the active Gold table; writes are blocked."""
        result = execute_readonly_query(database_path, sql)
        captured["sql_result"] = result
        return json.dumps(result, ensure_ascii=False, default=str)

    crew_summary = ""
    crew_error = None
    try:
        agent = Agent(
            role="Independent Data Analysis Agent",
            goal=(
                "Answer a business question from the active shared pipeline with "
                "a schema-grounded, read-only SQL query."
            ),
            backstory=(
                "You are vendor-independent. You operate on the Gold table silver_data "
                "published by the Data Engineering pipeline. Inspect the schema first. "
                "Never assume column names and never invent numeric results."
            ),
            llm=_make_agent_llm(),
            tools=[inspect_active_pipeline, query_active_pipeline_read_only],
            allow_delegation=False,
            max_iter=5,
            verbose=False,
        )
        task = Task(
            description=(
                f"Active pipeline ID: {manifest['pipeline_id']}. "
                f"Dataset: {manifest['dataset']}. Table: silver_data.\n"
                f"Business question: {question.strip()}\n\n"
                "Mandatory sequence: (1) call inspect_active_pipeline; "
                "(2) use only the returned columns; (3) write one SELECT query "
                "against silver_data; (4) call query_active_pipeline_read_only; "
                "(5) summarize only returned rows. No INSERT, UPDATE, DELETE, DDL, "
                "PRAGMA, ATTACH or multi-statement SQL."
            ),
            expected_output=(
                "A concise business answer, SQL used, and interpretation based "
                "only on results returned by the read-only query tool."
            ),
            agent=agent,
        )
        crew_result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
        crew_summary = str(crew_result)
    except Exception as exc:
        crew_error = f"{type(exc).__name__}: {exc}"

    execution_mode = "crewai_tool"
    if "sql_result" not in captured:
        fallback_sql = fallback_pipeline_sql(question)
        captured["sql_result"] = execute_readonly_query(database_path, fallback_sql)
        execution_mode = "safe_preview_fallback"

    payload = captured["sql_result"]
    dataframe = pd.DataFrame(payload["rows"], columns=payload["columns"])
    if not crew_summary:
        crew_summary = (
            f"Lecture seule sur le pipeline {manifest['pipeline_id']} : "
            f"{payload['row_count']} ligne(s) retournée(s)."
        )

    return {
        "provider": LOCAL_PROVIDER,
        "summary": crew_summary,
        "sql": payload["sql"],
        "rows": payload["rows"],
        "dataframe": dataframe,
        "database": str(database_path),
        "execution_mode": execution_mode,
        "conversation_id": "",
        "crew_error": crew_error,
        "provider_note": (
            f"Dataset partagé : {manifest['dataset']} · pipeline {manifest['pipeline_id']} · "
            "Gold.silver_data"
        ),
    }


def load_demo_sales_pipeline(
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
) -> dict[str, Any]:
    """Use the bundled retail example through the exact same shared ETL pipeline."""
    if not SAMPLE_SALES_CSV.is_file():
        raise FileNotFoundError(f"Demo dataset is missing: {SAMPLE_SALES_CSV}")
    run_dir = WORKSPACE_DIR / f"demo_{uuid.uuid4().hex[:10]}"
    source = copy_upload_to_workspace(SAMPLE_SALES_CSV, run_dir)
    return run_csv_pipeline(
        source,
        trim_text=trim_text,
        drop_duplicates=drop_duplicates,
        drop_empty_rows=drop_empty_rows,
    )
