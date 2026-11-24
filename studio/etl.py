from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CURRENT_PIPELINE_DIR,
    MAX_UPLOAD_BYTES,
    PIPELINE_MANIFEST,
    WORKSPACE_DIR,
)


def _workspace_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    root = WORKSPACE_DIR.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("The CSV path must be inside the Studio workspace.") from exc
    if candidate.suffix.lower() != ".csv" or not candidate.is_file():
        raise ValueError("A readable CSV file inside the workspace is required.")
    if candidate.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the configured upload size limit.")
    return candidate


def run_csv_pipeline(
    csv_path: str | Path,
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
) -> dict[str, Any]:
    """Run one shared Bronze → Silver → Gold pipeline for both agent scenarios."""
    source = _workspace_file(csv_path)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

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

    if frame.empty:
        raise ValueError("All rows were removed by the selected cleaning rules.")

    # The single active pipeline is the shared hand-off contract to every agent.
    pipeline_id = uuid.uuid4().hex[:12]
    bronze_dir = CURRENT_PIPELINE_DIR / "bronze"
    silver_dir = CURRENT_PIPELINE_DIR / "silver"
    gold_dir = CURRENT_PIPELINE_DIR / "gold"
    metadata_dir = CURRENT_PIPELINE_DIR / "metadata"
    for directory in (bronze_dir, silver_dir, gold_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bronze_file = bronze_dir / "source.csv"
    silver_file = silver_dir / "cleaned.csv"
    database_path = gold_dir / "warehouse.sqlite"
    report_path = metadata_dir / "quality_report.json"

    shutil.copy2(source, bronze_file)
    frame.to_csv(silver_file, index=False)

    with sqlite3.connect(database_path) as connection:
        frame.to_sql("silver_data", connection, if_exists="replace", index=False)
        # Stable alias for agents and examples that expect a staging table.
        frame.to_sql("staging_upload", connection, if_exists="replace", index=False)

    report = {
        "pipeline_id": pipeline_id,
        "status": "READY",
        "dataset": source.stem,
        "source_file": source.name,
        "rows_before": rows_before,
        "rows_after": len(frame),
        "duplicates_detected_before_cleaning": duplicates_before,
        "duplicates_removed": duplicates_removed,
        "nulls_before": nulls_before,
        "nulls_after": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "columns": [str(column) for column in frame.columns],
        "layers": {
            "bronze": str(bronze_file),
            "silver": str(silver_file),
            "gold_database": str(database_path),
            "gold_table": "silver_data",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        **report,
        "database": str(database_path),
        "analysis_table": "silver_data",
        "cleaned_csv": str(silver_file),
        "quality_report": str(report_path),
    }
    PIPELINE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def copy_upload_to_workspace(upload_path: str | Path, run_dir: Path) -> Path:
    """Copy one Gradio upload into a per-run workspace before exposing it to tools."""
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
