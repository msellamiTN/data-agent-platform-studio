import sqlite3

import pytest

import studio.sql as sql_module
from studio.sql import execute_readonly_query, plan_supported_question


def _database(tmp_path, monkeypatch):
    pipeline_root = tmp_path / "pipeline"
    database_dir = pipeline_root / "runs" / "test-run" / "gold"
    database_dir.mkdir(parents=True)
    monkeypatch.setattr(sql_module, "PIPELINE_ROOT", pipeline_root)
    db_path = database_dir / "warehouse.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE silver_data(region TEXT, revenue REAL, title TEXT)")
        connection.executemany(
            "INSERT INTO silver_data VALUES (?, ?, ?)",
            [
                ("North", 10.0, "A"),
                ("South", 20.0, "B"),
                ("South", 30.0, "B"),
            ],
        )
        connection.execute("CREATE TABLE staging_upload(secret TEXT)")
        connection.execute("INSERT INTO staging_upload VALUES ('must-not-be-readable')")
    return db_path


def test_select_query_runs_read_only(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
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
        "SELECT * FROM staging_upload",
        "SELECT name FROM sqlite_master",
    ],
)
def test_non_readonly_or_out_of_scope_statements_are_rejected(tmp_path, monkeypatch, statement):
    db_path = _database(tmp_path, monkeypatch)
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        execute_readonly_query(db_path, statement)


def test_query_execution_budget_interrupts_expensive_recursive_cte(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(sql_module, "MAX_SQL_EXECUTION_SECONDS", 0.05)
    expensive = (
        "WITH RECURSIVE counter(value) AS ("
        "SELECT 1 UNION ALL SELECT value + 1 FROM counter WHERE value < 100000000"
        ") SELECT MAX(value) AS max_value FROM counter"
    )
    with pytest.raises(TimeoutError, match="execution budget"):
        execute_readonly_query(db_path, expensive)


def test_supported_column_count_plan_uses_exact_real_column():
    plan = plan_supported_question(
        "how many titre?",
        ["titre", "Prix", "Nom", "prenom"],
    )
    assert plan["intent"] == "column_count"
    assert plan["column"] == "titre"
    assert plan["sql"] == 'SELECT COUNT("titre") AS "value_count" FROM silver_data'


def test_supported_distinct_count_plan_uses_distinct():
    plan = plan_supported_question(
        "how many distinct titre values?",
        ["titre", "Prix", "Nom", "prenom"],
    )
    assert plan["intent"] == "distinct_count"
    assert plan["sql"] == (
        'SELECT COUNT(DISTINCT "titre") AS "distinct_count" FROM silver_data'
    )


def test_row_count_plan_counts_all_rows():
    plan = plan_supported_question("how many rows?", ["titre", "Prix"])
    assert plan["intent"] == "row_count"
    assert plan["sql"] == "SELECT COUNT(*) AS row_count FROM silver_data"


def test_unknown_count_column_does_not_degrade_to_preview():
    plan = plan_supported_question("how many customers?", ["titre", "Prix"])
    assert plan["intent"] == "unsupported_count"
    assert "titre, Prix" in plan["error"]


def test_explicit_preview_plan_is_bounded():
    plan = plan_supported_question("Preview the first 500 rows", ["titre", "Prix"])
    assert plan["intent"] == "preview"
    assert plan["sql"] == "SELECT * FROM silver_data LIMIT 100"


def test_unrecognised_question_does_not_claim_supported_intent():
    assert plan_supported_question("Compare categories by average revenue", ["category", "revenue"]) is None
