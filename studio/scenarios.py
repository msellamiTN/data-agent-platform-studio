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
from .llm import get_local_llm
from .sql import (
    describe_pipeline_schema,
    execute_readonly_query,
    fallback_pipeline_sql,
    get_current_pipeline,
)


def _make_agent_llm():
    """Return the shared local Hugging Face model adapter for both agents."""
    return get_local_llm()


def run_data_engineering(
    upload_path: str,
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
    business_context: str = "",
) -> dict[str, Any]:
    """CrewAI Data Engineering Agent creates and publishes the shared pipeline."""

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
            business_context=business_context,
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
                "explain the verified quality, metadata and lineage report."
            ),
            backstory=(
                "You are a vendor-independent data engineer. You profile and "
                "prepare data through deterministic Python tools, preserve source "
                "lineage and never invent row counts or quality metrics."
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
                "row counts, duplicate removals, quality report, semantic catalog and "
                "active Gold table. State that the pipeline manifest is shared with "
                "the Data Analysis Agent. Do not claim that data was published externally. "
                f"Business context supplied by the user: {business_context.strip() or 'not supplied'}."
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
            business_context=business_context,
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
            "Le Gold partagé est disponible pour le Data Analysis Agent."
        )

    return {
        "summary": crew_summary,
        "report": report,
        "cleaned_file": str(cleaned_file),
        "preview": preview,
        "execution_mode": execution_mode,
        "crew_error": crew_error,
    }


def run_data_analysis(question: str) -> dict[str, Any]:
    """CrewAI Data Analysis Agent queries the exact active Gold dataset."""
    if not question or not question.strip():
        raise ValueError("Enter a business question before running the agent.")

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
        """Execute one SELECT query against Gold.silver_data; all writes are blocked."""
        result = execute_readonly_query(database_path, sql)
        captured["sql_result"] = result
        return json.dumps(result, ensure_ascii=False, default=str)

    crew_summary = ""
    crew_error = None
    try:
        agent = Agent(
            role="Data Analysis Agent",
            goal=(
                "Answer business questions from the active shared pipeline with "
                "schema-grounded, read-only SQL and evidence-backed explanations."
            ),
            backstory=(
                "You are an independent BI analyst. You use the active pipeline "
                "manifest, inspect real columns, generate one SELECT query and "
                "summarize only the rows returned by the protected SQL tool."
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
                f"Dataset: {manifest['dataset']}. Gold table: silver_data.\n"
                f"Business context: {manifest.get('business_context') or 'not supplied'}.\n"
                f"Semantic catalog: {json.dumps(manifest.get('semantic_catalog', {}), ensure_ascii=False)}.\n"
                f"Business question: {question.strip()}\n\n"
                "Mandatory sequence: (1) call inspect_active_pipeline; "
                "(2) use only the returned columns; (3) write one SELECT query "
                "against silver_data; (4) call query_active_pipeline_read_only; "
                "(5) summarize only returned rows. Never use INSERT, UPDATE, "
                "DELETE, CREATE, DROP, ATTACH, PRAGMA or multiple SQL statements. "
                "Do not invent numeric results."
            ),
            expected_output=(
                "A concise business answer, the SQL used, and an interpretation "
                "based only on results returned by the read-only query tool."
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
    if execution_mode == "safe_preview_fallback":
        if payload["rows"] and payload["columns"] and payload["columns"][0] in {"value_count", "distinct_count", "row_count"}:
            value = payload["rows"][0][payload["columns"][0]]
            target = "distinct values" if payload["columns"][0] == "distinct_count" else "non-null values"
            if payload["columns"][0] == "row_count":
                crew_summary = f"The active dataset contains {value} rows."
            else:
                crew_summary = f"The requested column contains {value} {target}."
        else:
            crew_summary = (
                f"CrewAI did not complete. A safe preview returned {payload['row_count']} row(s); "
                "this preview is not an aggregate answer to your question."
            )
    elif not crew_summary:
        crew_summary = (
            f"Read-only query completed on pipeline {manifest['pipeline_id']}: "
            f"{payload['row_count']} row(s) returned."
        )

    return {
        "provider": "Independent CrewAI + Hugging Face",
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
    """Use the bundled retail example through the same shared ETL pipeline."""
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
