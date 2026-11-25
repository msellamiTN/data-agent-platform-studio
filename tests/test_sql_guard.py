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
                ("South", 30.0, "A title that is longer than five characters"),
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
        "SELECT COUNT(*) FROM silver_data UNION ALL SELECT value + 1 FROM counter WHERE value < 100000000"
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



def test_constant_select_without_active_dataset_is_rejected(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must read from the active silver_data"):
        execute_readonly_query(db_path, "SELECT 1")


def test_result_rows_are_capped_at_configured_row_limit(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(sql_module, "MAX_SQL_ROWS", 2)
    result = execute_readonly_query(db_path, "SELECT region FROM silver_data")
    assert result["row_count"] == 2
    assert result["truncated"] is True
    assert "row_limit" in result["truncation_reasons"]


def test_oversized_text_cells_are_truncated_and_reported(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(sql_module, "MAX_SQL_CELL_CHARS", 5)
    result = execute_readonly_query(
        db_path,
        "SELECT title FROM silver_data ORDER BY revenue DESC",
    )
    assert result["rows"][0]["title"] == "A titl…"
    assert result["truncated"] is True
    assert result["truncated_cells"] >= 1
    assert "cell_value_limit" in result["truncation_reasons"]


def test_result_payload_obeys_byte_budget(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(sql_module, "MAX_SQL_CELL_CHARS", 16)
    monkeypatch.setattr(sql_module, "MAX_SQL_RESULT_BYTES", 80)
    result = execute_readonly_query(db_path, "SELECT region, title FROM silver_data")
    assert result["result_bytes"] <= 80
    assert result["truncated"] is True
    assert "result_byte_limit" in result["truncation_reasons"]


def test_blob_values_are_json_safe_and_omitted_from_payload(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    with sqlite3.connect(db_path) as connection:
        connection.execute("ALTER TABLE silver_data ADD COLUMN payload BLOB")
        connection.execute(
            "UPDATE silver_data SET payload = ? WHERE region = 'North'",
            (sqlite3.Binary(b"sample-binary-data"),),
        )
    result = execute_readonly_query(
        db_path,
        "SELECT payload FROM silver_data WHERE region = 'North'",
    )
    assert result["rows"][0]["payload"] == "[binary value omitted: 18 bytes]"
    assert result["truncated"] is True
    import json
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_unsafe_sql_functions_are_denied(tmp_path, monkeypatch):
    db_path = _database(tmp_path, monkeypatch)
    with pytest.raises(sqlite3.DatabaseError):
        execute_readonly_query(
            db_path,
            "SELECT load_extension('extension.so') FROM silver_data LIMIT 1",
        )
