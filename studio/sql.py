from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import (
    MAX_SQL_EXECUTION_SECONDS,
    MAX_SQL_ROWS,
    PIPELINE_MANIFEST,
    PIPELINE_ROOT,
)


def get_current_pipeline() -> dict[str, Any]:
    """Return the active immutable run manifest after validating its Gold path."""
    if not PIPELINE_MANIFEST.is_file():
        raise FileNotFoundError(
            "No active data pipeline. Run the Data Engineering pipeline first."
        )
    try:
        manifest = json.loads(PIPELINE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The active pipeline manifest is unreadable or invalid.") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("The active pipeline manifest must be a JSON object.")
    required = ("pipeline_id", "dataset", "database", "analysis_table", "status")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise RuntimeError(f"The active pipeline manifest is missing: {', '.join(missing)}.")
    if manifest.get("status") != "READY" or manifest.get("analysis_table") != "silver_data":
        raise RuntimeError("The active pipeline is not ready for analysis.")

    database = Path(manifest["database"]).expanduser().resolve()
    try:
        database.relative_to(PIPELINE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Pipeline manifest references a database outside the pipeline workspace.") from exc
    if not database.is_file():
        raise RuntimeError("The active Gold database is missing.")
    return manifest


def _check_sql(sql: str) -> str:
    statement = sql.strip()
    if not statement:
        raise ValueError("SQL query is empty.")
    if len(statement) > 10000:
        raise ValueError("SQL query is too long.")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise ValueError("SQL comments are not allowed.")
    if ";" in statement.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")
    statement = statement.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", statement, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")
    return statement


def execute_readonly_query(database_path: str | Path, sql: str) -> dict[str, Any]:
    """Run one bounded SELECT against Gold.silver_data.

    The database is opened read-only, the authorizer limits reads to the
    canonical table and safe SQL functions, results are capped, and SQLite's
    progress handler interrupts queries that exceed the execution budget.
    """
    db = Path(database_path).expanduser().resolve()
    try:
        db.relative_to(PIPELINE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Database path must be inside the pipeline workspace.") from exc
    if not db.is_file():
        raise FileNotFoundError("The Gold database does not exist.")

    statement = _check_sql(sql)
    deadline = time.monotonic() + max(0.05, MAX_SQL_EXECUTION_SECONDS)
    connection = sqlite3.connect(
        f"file:{db.as_posix()}?mode=ro",
        uri=True,
        timeout=2.0,
    )
    allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
    if recursive_action is not None:
        allowed_actions.add(recursive_action)
    blocked_functions = {"load_extension", "readfile", "writefile", "fts3_tokenizer"}

    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        del database, trigger
        if action not in allowed_actions:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and arg1 != "silver_data":
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (arg2 or arg1 or "").casefold()
            if function_name in blocked_functions:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def interrupt_if_over_budget() -> int:
        return 1 if time.monotonic() > deadline else 0

    try:
        connection.set_authorizer(authorizer)
        connection.set_progress_handler(interrupt_if_over_budget, 1000)
        cursor = connection.execute(statement)
        columns = [item[0] for item in (cursor.description or [])]
        rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
        truncated = len(rows) > MAX_SQL_ROWS
        rows = rows[:MAX_SQL_ROWS]
        serialised = [
            {column: value for column, value in zip(columns, row)}
            for row in rows
        ]
        return {
            "sql": statement,
            "columns": columns,
            "rows": serialised,
            "row_count": len(serialised),
            "truncated": truncated,
        }
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).casefold():
            raise TimeoutError(
                f"Read-only SQL exceeded the {MAX_SQL_EXECUTION_SECONDS:g}s execution budget."
            ) from exc
        raise
    finally:
        connection.close()


def describe_pipeline_schema() -> dict[str, Any]:
    """Describe the exact Gold table from the active immutable pipeline run."""
    manifest = get_current_pipeline()
    database = Path(manifest["database"]).resolve()
    sample_result = execute_readonly_query(
        database,
        "SELECT * FROM silver_data LIMIT 5",
    )
    columns = sample_result["columns"]
    return {
        "pipeline_id": manifest["pipeline_id"],
        "dataset": manifest["dataset"],
        "table": "silver_data",
        "columns": columns,
        "column_profile": manifest.get("column_profile", {}),
        "semantic_catalog": manifest.get("semantic_catalog", {}),
        "business_context": manifest.get("business_context", ""),
        "row_count_after_cleaning": manifest.get("rows_after"),
        "sample_rows": sample_result["rows"],
    }


def _normalise_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def plan_supported_question(question: str, columns: list[str]) -> dict[str, Any] | None:
    """Return a deterministic query plan for safe, unambiguous question patterns.

    Supported semantic intents are row count, non-null column count, distinct
    column count and explicit preview. Unsupported count requests return an
    error plan rather than degrading into an unrelated row preview.
    """
    normalized = _normalise_text(question)
    quoted = lambda name: '"' + name.replace('"', '""') + '"'
    count_trigger = re.search(
        r"\b(how many|number of|count|combien|nombre de)\b",
        normalized,
    )
    row_trigger = re.search(
        r"\b(rows?|records?|entries|lignes?|enregistrements?)\b",
        normalized,
    )
    if count_trigger and row_trigger:
        return {
            "intent": "row_count",
            "sql": "SELECT COUNT(*) AS row_count FROM silver_data",
            "answer_semantics": "total_rows",
        }

    if count_trigger:
        ordered_columns = sorted((str(column) for column in columns), key=len, reverse=True)
        matched_column = None
        for column in ordered_columns:
            candidate = _normalise_text(column)
            if not candidate:
                continue
            if re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", normalized):
                matched_column = column
                break
        if matched_column is None:
            return {
                "intent": "unsupported_count",
                "error": (
                    "I couldn't match the requested count to a column in this dataset. "
                    f"Use one of these column names: {', '.join(str(c) for c in columns)}."
                ),
            }
        distinct = bool(re.search(r"\b(distinct|unique|different|uniques|distincts)\b", normalized))
        expression = (
            f"COUNT(DISTINCT {quoted(matched_column)})"
            if distinct
            else f"COUNT({quoted(matched_column)})"
        )
        alias = "distinct_count" if distinct else "value_count"
        return {
            "intent": "distinct_count" if distinct else "column_count",
            "column": matched_column,
            "sql": f"SELECT {expression} AS {quoted(alias)} FROM silver_data",
            "answer_semantics": "distinct_values" if distinct else "non_null_values",
        }

    preview_trigger = re.search(
        r"\b(preview|sample|show rows|show me rows|first rows|display rows|afficher|montre)\b",
        normalized,
    )
    if preview_trigger:
        limit_match = re.search(r"\b(?:first|top|premieres?|premiers?)\s+(\d+)\b", normalized)
        limit = min(100, max(1, int(limit_match.group(1)))) if limit_match else 25
        return {
            "intent": "preview",
            "sql": f"SELECT * FROM silver_data LIMIT {limit}",
            "answer_semantics": "preview",
        }
    return None

