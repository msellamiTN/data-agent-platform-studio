from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MAX_UPLOAD_BYTES, WORKSPACE_DIR


def _workspace_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    root = WORKSPACE_DIR.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("The CSV path must be inside the Studio workspace.") from exc
    if candidate.suffix.lower() != ".csv" or not candidate.is_file():
        raise ValueError("A readable CSV file inside the workspace is required.")
    return candidate


def run_csv_pipeline(
    csv_path: str | Path,
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
) -> dict[str, Any]:
    """Profile, clean, export, and load a CSV to a local SQLite staging table."""
    source = _workspace_file(csv_path)
    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the configured upload size limit.")

    frame = pd.read_csv(source)
    rows_before = len(frame)
    duplicates_before = int(frame.duplicated().sum())
    nulls_before = {str(k): int(v) for k, v in frame.isna().sum().items()}

    if trim_text:
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            frame[column] = frame[column].replace("", pd.NA)

    if drop_empty_rows:
        frame = frame.dropna(how="all")

    duplicates_removed = 0
    if drop_duplicates:
        before = len(frame)
        frame = frame.drop_duplicates()
        duplicates_removed = before - len(frame)

    output_csv = source.with_name(f"{source.stem}_cleaned.csv")
    frame.to_csv(output_csv, index=False)

    database_path = WORKSPACE_DIR / "lakehouse.sqlite"
    with sqlite3.connect(database_path) as connection:
        frame.to_sql("staging_upload", connection, if_exists="replace", index=False)

    report = {
        "source_file": source.name,
        "rows_before": rows_before,
        "rows_after": len(frame),
        "duplicates_detected_before_cleaning": duplicates_before,
        "duplicates_removed": duplicates_removed,
        "nulls_before": nulls_before,
        "nulls_after": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "columns": [str(column) for column in frame.columns],
        "loaded_table": "staging_upload",
        "database": str(database_path),
        "cleaned_csv": str(output_csv),
    }
    report_path = source.with_name("quality_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["quality_report"] = str(report_path)
    return report


def copy_upload_to_workspace(upload_path: str | Path, run_dir: Path) -> Path:
    """Copy a Gradio upload into a per-run workspace before exposing it to tools."""
    source = Path(upload_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Upload file was not found.")
    if source.suffix.lower() != ".csv":
        raise ValueError("Only CSV uploads are supported in this prototype.")
    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the configured size limit.")

    run_dir.mkdir(parents=True, exist_ok=True)
    destination = (run_dir / "input.csv").resolve()
    destination.write_bytes(source.read_bytes())
    return destination
