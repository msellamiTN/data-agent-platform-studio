import json
import sqlite3

import pandas as pd

import studio.etl as etl_module


def test_csv_pipeline_trims_deduplicates_and_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(etl_module, "WORKSPACE_DIR", tmp_path)
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

    assert report["rows_before"] == 4
    assert report["rows_after"] == 2
    assert report["duplicates_removed"] == 1
    assert report["loaded_table"] == "staging_upload"

    cleaned = pd.read_csv(report["cleaned_csv"])
    assert cleaned.iloc[0]["customer"] == "Alice"

    with sqlite3.connect(report["database"]) as connection:
        count = connection.execute("SELECT COUNT(*) FROM staging_upload").fetchone()[0]
    assert count == 2

    quality = json.loads(open(report["quality_report"], encoding="utf-8").read())
    assert quality["rows_after"] == 2
