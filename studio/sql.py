from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import (
    MAX_SQL_CELL_CHARS,
    MAX_SQL_EXECUTION_SECONDS,
    MAX_SQL_RESULT_BYTES,
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
    # SQLite runtime limits are applied in addition to the application-level
    # row, cell and response-byte budgets. These reduce oversized-value and
    # pathological-query exposure before Python materializes result objects.
    limits = (
        (getattr(sqlite3, "SQLITE_LIMIT_LENGTH", None), 1_048_576),
        (getattr(sqlite3, "SQLITE_LIMIT_SQL_LENGTH", None), 10_000),
        (getattr(sqlite3, "SQLITE_LIMIT_COLUMN", None), 250),
        (getattr(sqlite3, "SQLITE_LIMIT_ATTACHED", None), 0),
        (getattr(sqlite3, "SQLITE_LIMIT_COMPOUND_SELECT", None), 10),
        (getattr(sqlite3, "SQLITE_LIMIT_EXPR_DEPTH", None), 100),
    )
    for category, limit in limits:
        if category is not None:
            connection.setlimit(category, limit)

    allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
    if recursive_action is not None:
        allowed_actions.add(recursive_action)
    blocked_functions = {"load_extension", "readfile", "writefile", "fts3_tokenizer"}

    active_table_read = False

    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        nonlocal active_table_read
        del database, trigger
        if action not in allowed_actions:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            if arg1 != "silver_data":
                return sqlite3.SQLITE_DENY
            active_table_read = True
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (arg2 or arg1 or "").casefold()
            if function_name in blocked_functions:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def interrupt_if_over_budget() -> int:
        return 1 if time.monotonic() > deadline else 0

    def normalise_cell(value: Any) -> tuple[Any, bool]:
        """Return JSON-safe SQL scalar data under the per-cell size budget."""
        if isinstance(value, str):
            if len(value) > MAX_SQL_CELL_CHARS:
                return value[:MAX_SQL_CELL_CHARS] + "…", True
            return value, False
        if isinstance(value, bytes):
            return f"[binary value omitted: {len(value):,} bytes]", True
        if isinstance(value, float) and not math.isfinite(value):
            return str(value), True
        if value is None or isinstance(value, (int, float)):
            return value, False
        # sqlite3 normally returns only primitive scalar values. Fail closed if
        # a future adapter or extension introduces an unsupported Python type.
        return str(value)[:MAX_SQL_CELL_CHARS], True

    try:
        connection.set_authorizer(authorizer)
        connection.set_progress_handler(interrupt_if_over_budget, 1000)
        cursor = connection.execute(statement)
        if not active_table_read:
            raise ValueError("Query must read from the active silver_data table.")
        columns = [
            str(item[0])[:128]
            for item in (cursor.description or [])
        ]
        serialised: list[dict[str, Any]] = []
        response_bytes = len(
            json.dumps(
                {"sql": statement, "columns": columns},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if response_bytes > MAX_SQL_RESULT_BYTES:
            raise ValueError(
                "SQL result column metadata exceeds the configured response-byte budget."
            )

        truncated_cells = 0
        row_limit_truncated = False
        byte_limit_truncated = False
        fetched_count = 0
        while fetched_count < MAX_SQL_ROWS:
            row = cursor.fetchone()
            if row is None:
                break
            fetched_count += 1
            safe_row: dict[str, Any] = {}
            row_truncated_cells = 0
            for column, value in zip(columns, row):
                safe_value, was_truncated = normalise_cell(value)
                safe_row[column] = safe_value
                row_truncated_cells += int(was_truncated)
            row_bytes = len(
                json.dumps(
                    safe_row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            if response_bytes + row_bytes > MAX_SQL_RESULT_BYTES:
                byte_limit_truncated = True
                break
            serialised.append(safe_row)
            response_bytes += row_bytes
            truncated_cells += row_truncated_cells

        if not byte_limit_truncated and fetched_count == MAX_SQL_ROWS:
            row_limit_truncated = cursor.fetchone() is not None

        truncated = row_limit_truncated or byte_limit_truncated or truncated_cells > 0
        reasons = []
        if row_limit_truncated:
            reasons.append("row_limit")
        if byte_limit_truncated:
            reasons.append("result_byte_limit")
        if truncated_cells:
            reasons.append("cell_value_limit")
        return {
            "sql": statement,
            "columns": columns,
            "rows": serialised,
            "row_count": len(serialised),
            "truncated": truncated,
            "truncation_reasons": reasons,
            "truncated_cells": truncated_cells,
            "result_bytes": response_bytes,
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
    """Return compact, schema-grounded context for the local analysis agent."""
    manifest = get_current_pipeline()
    columns = [str(column) for column in manifest.get("columns", [])]
    raw_profile = manifest.get("column_profile", {})
    compact_profile: dict[str, Any] = {}
    for column in columns:
        profile = raw_profile.get(column, {})
        samples = profile.get("sample_values", [])
        compact_profile[column] = {
            "dtype": profile.get("dtype", "unknown"),
            "null_count": profile.get("null_count"),
            "distinct_count": profile.get("distinct_count"),
            "sample_values": [
                str(value)[:64] + ("…" if len(str(value)) > 64 else "")
                for value in samples[:1]
            ],
        }
    semantic = manifest.get("semantic_catalog", {})
    return {
        "pipeline_id": manifest["pipeline_id"],
        "dataset": manifest["dataset"],
        "table": "silver_data",
        "columns": columns,
        "column_profile": compact_profile,
        "semantic_catalog": {
            "dimensions": semantic.get("dimensions", []),
            "measures": semantic.get("measures", []),
            "date_columns": semantic.get("date_columns", []),
        },
        "business_context": manifest.get("business_context", ""),
        "row_count_after_cleaning": manifest.get("rows_after"),
        "schema_context_note": (
            "Column type and profile classifications are heuristic metadata; "
            "verify business meaning with the user before making business-critical claims."
        ),
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

