import sqlite3

import pytest

from studio.sql import execute_readonly_query


def test_select_query_runs_read_only(tmp_path, monkeypatch):
    import studio.sql as sql_module

    monkeypatch.setattr(sql_module, "WORKSPACE_DIR", tmp_path)
    db_path = tmp_path / "warehouse.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE silver_data(region TEXT, revenue REAL)")
        connection.executemany(
            "INSERT INTO silver_data VALUES (?, ?)",
            [("North", 10.0), ("South", 20.0)],
        )

    result = execute_readonly_query(
        db_path,
        "SELECT region, SUM(revenue) AS total FROM silver_data GROUP BY region ORDER BY total DESC",
    )
    assert result["row_count"] == 2
    assert result["rows"][0]["region"] == "South"


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM silver_data",
        "UPDATE silver_data SET revenue = 0",
        "DROP TABLE silver_data",
        "SELECT 1; DELETE FROM silver_data",
        "SELECT 1 -- bypass",
        "PRAGMA table_info(silver_data)",
    ],
)
def test_non_readonly_or_multiple_statements_are_rejected(tmp_path, monkeypatch, statement):
    import studio.sql as sql_module

    monkeypatch.setattr(sql_module, "WORKSPACE_DIR", tmp_path)
    db_path = tmp_path / "warehouse.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE silver_data(region TEXT, revenue REAL)")

    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        execute_readonly_query(db_path, statement)


def test_fallback_count_question_uses_count_for_requested_column(monkeypatch):
    import studio.sql as sql_module

    monkeypatch.setattr(
        sql_module,
        "describe_pipeline_schema",
        lambda: {"columns": ["titre", "Prix", "Nom", "prenom"]},
    )
    query = sql_module.fallback_pipeline_sql("how many titre?")
    assert query == 'SELECT COUNT("titre") AS "value_count" FROM silver_data'


def test_fallback_distinct_count_question_uses_distinct(monkeypatch):
    import studio.sql as sql_module

    monkeypatch.setattr(
        sql_module,
        "describe_pipeline_schema",
        lambda: {"columns": ["titre", "Prix", "Nom", "prenom"]},
    )
    query = sql_module.fallback_pipeline_sql("how many distinct titre?")
    assert query == 'SELECT COUNT(DISTINCT "titre") AS "distinct_count" FROM silver_data'


def test_fallback_row_count_question_counts_all_rows(monkeypatch):
    import studio.sql as sql_module

    monkeypatch.setattr(
        sql_module,
        "describe_pipeline_schema",
        lambda: {"columns": ["titre", "Prix", "Nom", "prenom"]},
    )
    query = sql_module.fallback_pipeline_sql("how many rows?")
    assert query == "SELECT COUNT(*) AS row_count FROM silver_data"
