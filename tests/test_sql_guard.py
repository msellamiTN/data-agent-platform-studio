import sqlite3

import pytest

from studio.sql import execute_readonly_query


def test_select_query_runs_read_only(tmp_path, monkeypatch):
    import studio.sql as sql_module

    monkeypatch.setattr(sql_module, "WORKSPACE_DIR", tmp_path)
    db_path = tmp_path / "sales.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE sales(region TEXT, revenue REAL)")
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [("North", 10.0), ("South", 20.0)],
        )

    result = execute_readonly_query(
        db_path,
        "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY total DESC",
    )
    assert result["row_count"] == 2
    assert result["rows"][0]["region"] == "South"


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM sales",
        "UPDATE sales SET revenue = 0",
        "DROP TABLE sales",
        "SELECT 1; DELETE FROM sales",
        "SELECT 1 -- bypass",
        "PRAGMA table_info(sales)",
    ],
)
def test_non_readonly_or_multiple_statements_are_rejected(tmp_path, monkeypatch, statement):
    import studio.sql as sql_module

    monkeypatch.setattr(sql_module, "WORKSPACE_DIR", tmp_path)
    db_path = tmp_path / "sales.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE sales(region TEXT, revenue REAL)")

    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        execute_readonly_query(db_path, statement)
