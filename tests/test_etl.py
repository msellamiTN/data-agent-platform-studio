import json
import sqlite3

import pandas as pd

import studio.etl as etl_module


def test_csv_pipeline_trims_deduplicates_and_loads_shared_layers(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipeline" / "current"
    manifest_path = pipeline_dir / "manifest.json"
    monkeypatch.setattr(etl_module, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(etl_module, "CURRENT_PIPELINE_DIR", pipeline_dir)
    monkeypatch.setattr(etl_module, "PIPELINE_MANIFEST", manifest_path)

    source = tmp_path / "input.csv"
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
    )

    assert report["status"] == "READY"
    assert report["rows_before"] == 4
    assert report["rows_after"] == 2
    assert report["duplicates_removed"] == 1
    assert report["analysis_table"] == "silver_data"
    assert manifest_path.is_file()
    assert (pipeline_dir / "bronze" / "source.csv").is_file()
    assert (pipeline_dir / "silver" / "cleaned.csv").is_file()
    assert (pipeline_dir / "gold" / "warehouse.sqlite").is_file()

    cleaned = pd.read_csv(report["cleaned_csv"])
    assert cleaned.iloc[0]["customer"] == "Alice"

    with sqlite3.connect(report["database"]) as connection:
        silver_count = connection.execute("SELECT COUNT(*) FROM silver_data").fetchone()[0]
        staging_count = connection.execute("SELECT COUNT(*) FROM staging_upload").fetchone()[0]
    assert silver_count == 2
    assert staging_count == 2

    quality = json.loads(open(report["quality_report"], encoding="utf-8").read())
    assert quality["rows_after"] == 2
