from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from .config import WORKSPACE_DIR
from .etl import copy_upload_to_workspace, run_csv_pipeline
from .llm import get_local_llm
from .sql import (
    build_sales_database,
    describe_sales_schema,
    execute_readonly_query,
    fallback_sales_sql,
)


def _make_agent_llm():
    return get_local_llm()


def run_data_engineering(
    upload_path: str,
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
) -> dict[str, Any]:
    """Run the Data Engineering Crew and return deterministic pipeline artifacts."""
    run_dir = WORKSPACE_DIR / f"etl_{uuid.uuid4().hex[:10]}"
    workspace_csv = copy_upload_to_workspace(upload_path, run_dir)
    captured: dict[str, Any] = {}

    @tool("run_csv_quality_etl")
    def run_csv_quality_etl() -> str:
        """Profile, clean, validate and load the uploaded CSV into SQLite staging."""
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
                "Execute the supplied CSV quality and staging pipeline tool, "
                "then explain its verified report."
            ),
            backstory=(
                "You are a careful data engineer. You never invent row counts. "
                "The Python ETL tool performs the actual data operations."
            ),
            llm=_make_agent_llm(),
            tools=[run_csv_quality_etl],
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )
        task = Task(
            description=(
                "Call the run_csv_quality_etl tool exactly once. Wait for its "
                "result. Summarize rows before/after, duplicates removed, missing "
                "values and the staging table. Do not claim any unreported action."
            ),
            expected_output=(
                "A concise report based only on the ETL tool output, including "
                "the cleaned CSV artifact and SQLite staging table."
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
    except Exception as exc:  # The deterministic ETL still runs if model/tool setup fails.
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
            f"CSV traité : {report['rows_before']} lignes en entrée, "
            f"{report['rows_after']} lignes après nettoyage, "
            f"{report['duplicates_removed']} doublon(s) retiré(s). "
            f"Chargement SQLite : {report['loaded_table']}."
        )

    return {
        "summary": crew_summary,
        "report": report,
        "cleaned_file": str(cleaned_file),
        "preview": preview,
        "execution_mode": execution_mode,
        "crew_error": crew_error,
    }


def run_data_analysis(question: str, upload_path: str | None = None) -> dict[str, Any]:
    """Run the Data Analysis Crew over an isolated, read-only SQLite dataset."""
    if not question or not question.strip():
        raise ValueError("Enter a business question before running the agent.")

    database_path = build_sales_database(upload_path)
    captured: dict[str, Any] = {}

    @tool("get_sales_schema")
    def get_sales_schema() -> str:
        """Return the sales table columns and sample rows before SQL generation."""
        schema = describe_sales_schema(database_path)
        return json.dumps(schema, ensure_ascii=False, default=str)

    @tool("execute_read_only_sql")
    def execute_read_only_sql(sql: str) -> str:
        """Execute one SELECT query only; writes, DDL and multiple statements are blocked."""
        result = execute_readonly_query(database_path, sql)
        captured["sql_result"] = result
        return json.dumps(result, ensure_ascii=False, default=str)

    crew_summary = ""
    crew_error = None
    try:
        agent = Agent(
            role="Data Analysis Agent",
            goal=(
                "Answer a business question with a validated read-only SQL query "
                "and evidence returned by the SQL tool."
            ),
            backstory=(
                "You are a cautious BI analyst. Inspect the schema first, use only "
                "existing columns, execute a single SELECT, and summarize only "
                "the rows returned by the tool."
            ),
            llm=_make_agent_llm(),
            tools=[get_sales_schema, execute_read_only_sql],
            allow_delegation=False,
            max_iter=5,
            verbose=False,
        )
        task = Task(
            description=(
                "Business question: "
                + question.strip()
                + "\n\nMandatory sequence: (1) call get_sales_schema; "
                "(2) inspect the returned columns; (3) write one SELECT query; "
                "(4) call execute_read_only_sql with that query; "
                "(5) summarize only the returned data. Never use INSERT, UPDATE, "
                "DELETE, CREATE, DROP, ATTACH, PRAGMA or multiple statements. "
                "Do not invent numeric results."
            ),
            expected_output=(
                "A concise French business answer, the SQL used, and a short "
                "interpretation based on the query result."
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
        fallback_sql = fallback_sales_sql(question, database_path)
        captured["sql_result"] = execute_readonly_query(database_path, fallback_sql)
        execution_mode = "safe_sql_fallback"

    payload = captured["sql_result"]
    rows = payload["rows"]
    dataframe = pd.DataFrame(rows, columns=payload["columns"])

    if not crew_summary:
        crew_summary = (
            f"Requête exécutée en lecture seule : {payload['row_count']} ligne(s) "
            "retournée(s). Vérifiez le SQL et le tableau avant toute décision métier."
        )

    return {
        "summary": crew_summary,
        "sql": payload["sql"],
        "rows": rows,
        "dataframe": dataframe,
        "database": str(database_path),
        "execution_mode": execution_mode,
        "crew_error": crew_error,
    }
