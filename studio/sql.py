from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MAX_SQL_ROWS, MAX_UPLOAD_BYTES, SAMPLE_SALES_CSV, WORKSPACE_DIR


def build_sales_database(csv_path: str | Path | None = None) -> Path:
    """Create a per-run SQLite database with a normalized sales table."""
    source = Path(csv_path) if csv_path else SAMPLE_SALES_CSV
    if not source.is_file():
        raise FileNotFoundError(f"Sales CSV not found: {source}")
    if source.suffix.lower() != ".csv":
        raise ValueError("The analysis upload must be a CSV file.")
    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Sales CSV exceeds the configured upload size limit.")

    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("The sales dataset is empty.")
    frame.columns = [
        re.sub(r"[^a-z0-9_]+", "_", str(column).strip().lower()).strip("_")
        for column in frame.columns
    ]

    if "region" not in frame.columns:
        raise ValueError("The analysis dataset must contain a 'region' column.")

    if "revenue" not in frame.columns:
        if {"quantity", "unit_price"}.issubset(frame.columns):
            frame["revenue"] = frame["quantity"] * frame["unit_price"]
        else:
            raise ValueError(
                "Add a 'revenue' column or both 'quantity' and 'unit_price'."
            )

    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
    frame["region"] = frame["region"].astype("string").str.strip()
    frame = frame.dropna(subset=["region", "revenue"])
    if frame.empty:
        raise ValueError("No rows have both a valid region and numeric revenue.")

    if "order_date" not in frame.columns:
        frame["order_date"] = "2026-01-01"
    else:
        frame["order_date"] = frame["order_date"].astype("string").str.strip()
    if "product" not in frame.columns:
        frame["product"] = "Unknown"
    else:
        frame["product"] = frame["product"].astype("string").str.strip()

    run_dir = WORKSPACE_DIR / f"analysis_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "analysis.sqlite"

    with sqlite3.connect(database_path) as connection:
        frame.to_sql("sales", connection, if_exists="replace", index=False)

    return database_path


def _check_sql(sql: str) -> str:
    statement = sql.strip()
    if not statement:
        raise ValueError("SQL query is empty.")
    if len(statement) > 10000:
        raise ValueError("SQL query is too long.")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise ValueError("SQL comments are not allowed in the lab.")
    if ";" in statement.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")
    statement = statement.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", statement, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")
    return statement


def execute_readonly_query(database_path: str | Path, sql: str) -> dict[str, Any]:
    """Execute a single read-only SQLite query with both URI and authorizer guards."""
    db = Path(database_path).expanduser().resolve()
    root = WORKSPACE_DIR.resolve()
    try:
        db.relative_to(root)
    except ValueError as exc:
        raise ValueError("Database path must be inside the Studio workspace.") from exc
    if not db.is_file():
        raise FileNotFoundError("Analysis database does not exist.")

    statement = _check_sql(sql)
    connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)

    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
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


def describe_sales_schema(database_path: str | Path) -> dict[str, Any]:
    """Return sales columns and sample rows; no model-generated SQL is involved."""
    db = Path(database_path).expanduser().resolve()
    db.relative_to(WORKSPACE_DIR.resolve())
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as connection:
        cursor = connection.execute("SELECT * FROM sales LIMIT 3")
        columns = [item[0] for item in cursor.description or []]
        sample = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return {"table": "sales", "columns": columns, "sample_rows": sample}


def fallback_sales_sql(question: str, database_path: str | Path) -> str:
    """Conservative query fallback if the small local model fails to call a tool."""
    schema = describe_sales_schema(database_path)
    columns = set(schema["columns"])
    lowered = question.casefold()

    if "product" in lowered or "produit" in lowered:
        group_column = "product"
    else:
        group_column = "region"

    if group_column not in columns:
        group_column = "region"

    date_filter = ""
    if (
        ("trimestre" in lowered or "quarter" in lowered or "t1" in lowered)
        and "order_date" in columns
    ):
        month_condition = (
            "CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 1 AND 3"
        )
        date_filter = f" WHERE {month_condition}"

    return (
        f"SELECT {group_column}, ROUND(SUM(revenue), 2) AS total_revenue "
        f"FROM sales{date_filter} "
        f"GROUP BY {group_column} ORDER BY total_revenue DESC LIMIT 100"
    )
