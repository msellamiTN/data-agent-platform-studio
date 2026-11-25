import json
import sqlite3

import pytest
from pathlib import Path

import pandas as pd

import studio.etl as etl_module
import studio.sql as sql_module


def _configure_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    pipeline_root = workspace / "pipeline"
    current_dir = pipeline_root / "current"
    manifest_path = current_dir / "manifest.json"
    monkeypatch.setattr(etl_module, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(etl_module, "PIPELINE_ROOT", pipeline_root)
    monkeypatch.setattr(etl_module, "CURRENT_PIPELINE_DIR", current_dir)
    monkeypatch.setattr(etl_module, "PIPELINE_MANIFEST", manifest_path)
    monkeypatch.setattr(sql_module, "PIPELINE_ROOT", pipeline_root)
    monkeypatch.setattr(sql_module, "PIPELINE_MANIFEST", manifest_path)
    return workspace, pipeline_root, manifest_path


def test_csv_pipeline_trims_deduplicates_and_publishes_shared_layers(tmp_path, monkeypatch):
    workspace, pipeline_root, manifest_path = _configure_workspace(tmp_path, monkeypatch)
    source = workspace / "input.csv"
    workspace.mkdir(parents=True)
    pd.DataFrame(
        [
            {"customer": " Alice ", "city": "Tunis"},
            {"customer": "Alice", "city": "Tunis"},
            {"customer": "Bob", "city": "Sousse"},
            {"customer": None, "city": None},
        ]
    ).to_csv(source, index=False)

    report = etl_module.run_csv_pipeline(
        source,
        trim_text=True,
        drop_duplicates=True,
        drop_empty_rows=True,
        business_context="Customer delivery dataset used for the retail demo.",
    )

    run_dir = pipeline_root / "runs" / report["pipeline_id"]
    assert report["status"] == "READY"
    assert report["rows_before"] == 4
    assert report["rows_after"] == 2
    assert report["duplicates_removed"] == 1
    assert report["analysis_table"] == "silver_data"
    assert report["business_context"] == "Customer delivery dataset used for the retail demo."
    assert "customer" in report["column_profile"]
    assert "city" in report["semantic_catalog"]["dimensions"]
    assert manifest_path.is_file()
    assert Path(report["layers"]["bronze"]).is_file()
    assert Path(report["layers"]["silver"]).is_file()
    assert Path(report["database"]).is_file()
    assert Path(report["quality_report"]).is_file()
    assert Path(report["manifest_path"]).is_file()
    assert Path(report["database"]).is_relative_to(run_dir)
    assert report["source_sha256"]
    assert report["silver_sha256"]
    assert report["gold_sha256"]

    cleaned = pd.read_csv(report["cleaned_csv"])
    assert cleaned.iloc[0]["customer"] == "Alice"

    with sqlite3.connect(report["database"]) as connection:
        silver_count = connection.execute("SELECT COUNT(*) FROM silver_data").fetchone()[0]
        staging_count = connection.execute("SELECT COUNT(*) FROM staging_upload").fetchone()[0]
    assert silver_count == 2
    assert staging_count == 2

    schema = sql_module.describe_pipeline_schema()
    assert schema["pipeline_id"] == report["pipeline_id"]
    assert schema["table"] == "silver_data"
    assert "customer" in schema["columns"]
    assert schema["business_context"] == "Customer delivery dataset used for the retail demo."
    assert "customer" in schema["column_profile"]
    query = sql_module.execute_readonly_query(
        report["database"],
        'SELECT customer FROM silver_data ORDER BY customer',
    )
    assert query["row_count"] == 2

    quality = json.loads(Path(report["quality_report"]).read_text(encoding="utf-8"))
    assert quality["rows_after"] == 2


def test_pipeline_runs_are_immutable_and_active_manifest_switches_atomically(tmp_path, monkeypatch):
    workspace, pipeline_root, manifest_path = _configure_workspace(tmp_path, monkeypatch)
    workspace.mkdir(parents=True)
    first = workspace / "first.csv"
    second = workspace / "second.csv"
    pd.DataFrame({"name": ["Ada", "Lin"]}).to_csv(first, index=False)
    pd.DataFrame({"name": ["Grace"]}).to_csv(second, index=False)

    first_manifest = etl_module.run_csv_pipeline(first)
    second_manifest = etl_module.run_csv_pipeline(second)

    assert first_manifest["pipeline_id"] != second_manifest["pipeline_id"]
    assert first_manifest["database"] != second_manifest["database"]
    assert Path(first_manifest["database"]).is_file()
    assert Path(second_manifest["database"]).is_file()
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert current["pipeline_id"] == second_manifest["pipeline_id"]

    # The first run remains queryable even after a second run becomes active.
    first_result = sql_module.execute_readonly_query(
        first_manifest["database"], "SELECT COUNT(*) AS row_count FROM silver_data"
    )
    assert first_result["rows"][0]["row_count"] == 2


def test_failed_pipeline_does_not_replace_previous_active_manifest(tmp_path, monkeypatch):
    workspace, _, manifest_path = _configure_workspace(tmp_path, monkeypatch)
    workspace.mkdir(parents=True)
    valid = workspace / "valid.csv"
    invalid = workspace / "all_empty.csv"
    pd.DataFrame({"name": ["A"]}).to_csv(valid, index=False)
    pd.DataFrame({"name": [None]}).to_csv(invalid, index=False)

    active = etl_module.run_csv_pipeline(valid)
    with pytest.raises(ValueError, match="All rows were removed"):
        etl_module.run_csv_pipeline(invalid)

    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert current["pipeline_id"] == active["pipeline_id"]



def test_pipeline_rejects_row_limit_exceeded(tmp_path, monkeypatch):
    workspace, _, _ = _configure_workspace(tmp_path, monkeypatch)
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "too_many_rows.csv"
    pd.DataFrame({"name": ["A", "B", "C"]}).to_csv(source, index=False)
    monkeypatch.setattr(etl_module, "MAX_DATA_ROWS", 2)
    with pytest.raises(ValueError, match="2-row processing limit"):
        etl_module.run_csv_pipeline(source)


def test_pipeline_rejects_column_limit_exceeded(tmp_path, monkeypatch):
    workspace, _, _ = _configure_workspace(tmp_path, monkeypatch)
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "too_many_columns.csv"
    pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(source, index=False)
    monkeypatch.setattr(etl_module, "MAX_DATA_COLUMNS", 2)
    with pytest.raises(ValueError, match="2-column processing limit"):
        etl_module.run_csv_pipeline(source)


def test_pipeline_rejects_oversized_business_context(tmp_path, monkeypatch):
    workspace, _, _ = _configure_workspace(tmp_path, monkeypatch)
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "context.csv"
    pd.DataFrame({"name": ["A"]}).to_csv(source, index=False)
    monkeypatch.setattr(etl_module, "MAX_BUSINESS_CONTEXT_CHARS", 5)
    with pytest.raises(ValueError, match="5-character limit"):
        etl_module.run_csv_pipeline(source, business_context="too long")
