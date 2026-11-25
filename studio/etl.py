from __future__ import annotations

import csv
import hashlib
import json
import os
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
    MAX_DATA_COLUMNS,
    MAX_DATA_ROWS,
    MAX_BUSINESS_CONTEXT_CHARS,
    PIPELINE_MANIFEST,
    PIPELINE_ROOT,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON by atomic replacement so readers never observe partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2, default=str)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_csv_pipeline(
    csv_path: str | Path,
    trim_text: bool = True,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
    business_context: str = "",
) -> dict[str, Any]:
    """Create an immutable run and atomically publish it as the active pipeline.

    All artifacts are written under pipeline/runs/<pipeline_id>. Only after the
    Bronze, Silver, Gold and metadata artifacts are complete is the active
    manifest replaced. In-flight readers therefore keep using their run's
    immutable SQLite database while a later run is being prepared.
    """
    source = _workspace_file(csv_path)
    context = str(business_context or "").strip()
    if len(context) > MAX_BUSINESS_CONTEXT_CHARS:
        raise ValueError(
            f"Business description exceeds the {MAX_BUSINESS_CONTEXT_CHARS:,}-character limit."
        )
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as csv_file:
            headers = next(csv.reader(csv_file), [])
    except UnicodeDecodeError as exc:
        raise ValueError("CSV headers must be valid UTF-8 text.") from exc
    if not headers or any(not str(header).strip() for header in headers):
        raise ValueError("CSV must contain non-empty column names in its first row.")
    normalized_headers = [str(header).strip().casefold() for header in headers]
    if len(set(normalized_headers)) != len(normalized_headers):
        raise ValueError("CSV column names must be unique (case-insensitive).")
    if len(headers) > MAX_DATA_COLUMNS:
        raise ValueError(
            f"CSV exceeds the {MAX_DATA_COLUMNS:,}-column processing limit."
        )

    frame = pd.read_csv(source, encoding="utf-8-sig", nrows=MAX_DATA_ROWS + 1)
    if len(frame) > MAX_DATA_ROWS:
        raise ValueError(
            f"CSV exceeds the {MAX_DATA_ROWS:,}-row processing limit."
        )
    if len(frame.columns) > MAX_DATA_COLUMNS:
        raise ValueError(
            f"CSV exceeds the {MAX_DATA_COLUMNS:,}-column processing limit."
        )
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if len(frame.columns) == 0:
        raise ValueError("The uploaded CSV must contain at least one column.")

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

    # The profile is a compact shared contract, not a substitute for a governed
    # business glossary. Classification is intentionally heuristic.
    column_profile: dict[str, Any] = {}
    dimensions: list[str] = []
    measures: list[str] = []
    date_columns: list[str] = []
    for column in frame.columns:
        series = frame[column]
        dtype_name = str(series.dtype)
        non_null = series.dropna()
        samples = []
        for value in non_null.head(5).tolist():
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif hasattr(value, "item"):
                value = value.item()
            samples.append(value if isinstance(value, (str, int, float, bool)) else str(value))
        column_profile[str(column)] = {
            "dtype": dtype_name,
            "null_count": int(series.isna().sum()),
            "distinct_count": int(series.nunique(dropna=True)),
            "sample_values": samples,
        }
        name = str(column).casefold()
        if pd.api.types.is_numeric_dtype(series):
            measures.append(str(column))
        elif any(token in name for token in ("date", "time", "timestamp", "jour")):
            date_columns.append(str(column))
        else:
            dimensions.append(str(column))

    semantic_catalog = {
        "dimensions": dimensions,
        "measures": measures,
        "date_columns": date_columns,
        "business_context": context,
        "classification_note": "Heuristic candidates inferred from column names and pandas dtypes; validate before business-critical use.",
    }

    pipeline_id = uuid.uuid4().hex[:12]
    run_dir = PIPELINE_ROOT / "runs" / pipeline_id
    bronze_dir = run_dir / "bronze"
    silver_dir = run_dir / "silver"
    gold_dir = run_dir / "gold"
    metadata_dir = run_dir / "metadata"
    for directory in (bronze_dir, silver_dir, gold_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=False)

    bronze_file = bronze_dir / "source.csv"
    silver_file = silver_dir / "cleaned.csv"
    database_path = gold_dir / "warehouse.sqlite"
    report_path = metadata_dir / "quality_report.json"
    run_manifest_path = run_dir / "manifest.json"

    try:
        shutil.copy2(source, bronze_file)
        frame.to_csv(silver_file, index=False)

        # A unique, run-scoped database avoids replacement races with readers.
        with sqlite3.connect(database_path) as connection:
            frame.to_sql("silver_data", connection, if_exists="fail", index=False)
            frame.to_sql("staging_upload", connection, if_exists="fail", index=False)

        report = {
            "pipeline_id": pipeline_id,
            "status": "READY",
            "dataset": source.stem,
            "source_file": source.name,
            "source_sha256": _sha256(bronze_file),
            "silver_sha256": _sha256(silver_file),
            "gold_sha256": _sha256(database_path),
            "rows_before": rows_before,
            "rows_after": len(frame),
            "duplicates_detected_before_cleaning": duplicates_before,
            "duplicates_removed": duplicates_removed,
            "nulls_before": nulls_before,
            "nulls_after": {str(k): int(v) for k, v in frame.isna().sum().items()},
            "columns": [str(column) for column in frame.columns],
            "column_profile": column_profile,
            "semantic_catalog": semantic_catalog,
            "business_context": context,
            "layers": {
                "bronze": str(bronze_file),
                "silver": str(silver_file),
                "gold_database": str(database_path),
                "gold_table": "silver_data",
            },
            "run_dir": str(run_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json_write(report_path, report)
        manifest = {
            **report,
            "database": str(database_path),
            "analysis_table": "silver_data",
            "cleaned_csv": str(silver_file),
            "quality_report": str(report_path),
            "manifest_path": str(run_manifest_path),
        }
        _atomic_json_write(run_manifest_path, manifest)

        # This small pointer manifest is the only mutable part of the contract.
        # Atomic replacement preserves the previous active run until this point.
        _atomic_json_write(PIPELINE_MANIFEST, manifest)
        return manifest
    except Exception:
        # Keep failed artifacts out of the active manifest. The incomplete run
        # is retained for diagnostics and can be removed by an explicit cleanup.
        raise


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
