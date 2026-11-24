from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .config import CURRENT_PIPELINE_DIR, MAX_SQL_ROWS, PIPELINE_MANIFEST, WORKSPACE_DIR


def get_current_pipeline() -> dict[str, Any]:
    """Return the shared current pipeline manifest or explain that ETL has not run."""
    if not PIPELINE_MANIFEST.is_file():
        raise FileNotFoundError(
            "No active data pipeline. Run the Data Engineering pipeline first."
        )
    manifest = json.loads(PIPELINE_MANIFEST.read_text(encoding="utf-8"))
    database = Path(manifest["database"]).resolve()
    try:
        database.relative_to(CURRENT_PIPELINE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Pipeline manifest references a database outside its workspace.") from exc
    if not database.is_file() or manifest.get("status") != "READY":
        raise RuntimeError("The active pipeline is not ready for analysis.")
    return manifest


def _check_sql(sql: str) -> str:
    statement = sql.strip()
    if not statement:
        raise ValueError("SQL query is empty.")
    if len(statement) > 10000:
        raise ValueError("SQL query is too long.")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise ValueError("SQL comments are not allowed in this lab.")
    if ";" in statement.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")
    statement = statement.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\\b", statement, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")
    return statement


def execute_readonly_query(database_path: str | Path, sql: str) -> dict[str, Any]:
    """Execute one query against the pipeline's Gold SQLite file in read-only mode."""
    db = Path(database_path).expanduser().resolve()
    root = WORKSPACE_DIR.resolve()
    try:
        db.relative_to(root)
    except ValueError as exc:
        raise ValueError("Database path must be inside the Studio workspace.") from exc
    if not db.is_file():
        raise FileNotFoundError("The Gold database does not exist.")

    statement = _check_sql(sql)
    connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
    if recursive_action is not None:
        allowed.add(recursive_action)

    def authorizer(action: int, arg1: str | None, arg2: str | None,
                   database: str | None, trigger: str | None) -> int:
        del arg1, arg2, database, trigger
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    try:
        connection.set_authorizer(authorizer)
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
    finally:
        connection.close()


def describe_pipeline_schema() -> dict[str, Any]:
    """Describe the exact Gold table produced by the shared Data Engineering pipeline."""
    manifest = get_current_pipeline()
    database = Path(manifest["database"]).resolve()
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        cursor = connection.execute("SELECT * FROM silver_data LIMIT 5")
        columns = [item[0] for item in cursor.description or []]
        sample = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return {
        "pipeline_id": manifest["pipeline_id"],
        "dataset": manifest["dataset"],
        "table": "silver_data",
        "columns": columns,
        "sample_rows": sample,
    }


def fallback_pipeline_sql(question: str) -> str:
    """Build a conservative preview/aggregation query from the active pipeline schema."""
    schema = describe_pipeline_schema()
    columns = {str(column).casefold(): str(column) for column in schema["columns"]}
    lowered = question.casefold()

    group_candidates = ("region", "area", "country", "product", "category", "department")
    metric_candidates = ("revenue", "sales", "amount", "montant", "total", "quantity", "count")
    group = next((columns[key] for key in group_candidates if key in columns), None)
    metric = next((columns[key] for key in metric_candidates if key in columns), None)

    def quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    if group and metric and any(word in lowered for word in ("revenue", "sales", "chiffre", "ca", "total", "par region", "by region")):
        return (
            f"SELECT {quote(group)} AS dimension, SUM(CAST({quote(metric)} AS REAL)) AS metric_total "
            f"FROM silver_data GROUP BY {quote(group)} "
            "ORDER BY metric_total DESC LIMIT 100"
        )
    return "SELECT * FROM silver_data LIMIT 25"
