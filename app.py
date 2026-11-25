from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px

from studio.config import MODEL_ID, SAMPLE_SALES_CSV
from studio.scenarios import run_data_analysis, run_data_engineering
from studio.sql import get_current_pipeline


# ---------------------------------------------------------------------------
# Visual system — neutral canvas, deep-navy navigation, calm teal action color
# ---------------------------------------------------------------------------

CSS = """
:root {
  --studio-ink: #17253a;
  --studio-muted: #708096;
  --studio-line: #e6ebf1;
  --studio-canvas: #f5f7fa;
  --studio-card: #ffffff;
  --studio-navy: #12243a;
  --studio-blue: #2867d8;
  --studio-teal: #0e9f8a;
  --studio-amber: #d99527;
}
.gradio-container {
  max-width: 1520px !important;
  margin: 0 auto !important;
  padding: 20px 26px 36px !important;
  background: var(--studio-canvas) !important;
}
body, .gradio-container, .prose, .wrap {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
               "Segoe UI", sans-serif !important;
}
#studio-shell { gap: 18px !important; }
#studio-topbar {
  display:flex; align-items:center; justify-content:space-between; gap:18px;
  padding:15px 20px; border:1px solid var(--studio-line); border-radius:16px;
  background:#fff; box-shadow:0 2px 12px rgba(17,36,58,.035);
}
.brand-lockup { display:flex; align-items:center; gap:12px; min-width:0; }
.brand-mark {
  width:40px; height:40px; border-radius:12px; display:grid; place-items:center;
  color:#fff; font-weight:800; letter-spacing:-1px; font-size:16px;
  background:linear-gradient(145deg,#183b62,#2867d8);
  box-shadow:0 5px 12px rgba(40,103,216,.2);
}
.brand-name { color:var(--studio-ink); font-size:16px; font-weight:750; letter-spacing:-.35px; }
.brand-sub { color:var(--studio-muted); font-size:11px; margin-top:2px; }
.topbar-right { display:flex; align-items:center; gap:9px; flex-wrap:wrap; justify-content:flex-end; }
.top-chip {
  display:inline-flex; align-items:center; gap:7px; border:1px solid var(--studio-line);
  border-radius:999px; padding:7px 11px; color:#526276; font-size:11px; font-weight:650;
  background:#fbfcfd; white-space:nowrap;
}
.dot { width:7px; height:7px; border-radius:50%; display:inline-block; background:#0e9f8a; }
.dot.gray { background:#9aa7b7; }
#studio-hero {
  position:relative; overflow:hidden; border-radius:20px; padding:28px 32px;
  color:white; background:linear-gradient(112deg,#10243a 0%,#183b62 62%,#23558a 100%);
  box-shadow:0 12px 30px rgba(18,36,58,.10);
}
#studio-hero:after {
  content:""; position:absolute; width:290px; height:290px; border-radius:50%;
  right:-70px; top:-145px; border:1px solid rgba(255,255,255,.12);
  box-shadow:0 0 0 28px rgba(255,255,255,.025),0 0 0 58px rgba(255,255,255,.02);
}
.hero-eyebrow { color:#a9c8ef; text-transform:uppercase; font-size:10px; letter-spacing:1.7px; font-weight:800; }
.hero-title { font-size:30px; line-height:1.16; letter-spacing:-1.1px; font-weight:760; margin:10px 0 8px; color:#fff; }
.hero-copy { color:#d0deed; max-width:760px; font-size:13px; line-height:1.65; }
.hero-meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }
.hero-tag { font-size:10px; border:1px solid rgba(255,255,255,.18); color:#e2edf8; border-radius:999px; padding:6px 10px; background:rgba(255,255,255,.06); }
.section-kicker { color:#78889c; font-size:10px; font-weight:800; letter-spacing:1.25px; text-transform:uppercase; margin:4px 0 8px; }
.section-heading { color:var(--studio-ink); font-size:19px; font-weight:760; letter-spacing:-.45px; margin:0 0 4px; }
.section-sub { color:var(--studio-muted); font-size:12px; line-height:1.55; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:14px 0 4px; }
.kpi-card {
  background:#fff; border:1px solid var(--studio-line); border-radius:14px; padding:16px 17px;
  box-shadow:0 2px 10px rgba(17,36,58,.025); min-width:0;
}
.kpi-label { color:#7b899a; font-size:10px; font-weight:750; letter-spacing:.75px; text-transform:uppercase; }
.kpi-value { color:var(--studio-ink); font-size:23px; line-height:1.25; font-weight:780; letter-spacing:-.7px; margin-top:8px; overflow-wrap:anywhere; }
.kpi-foot { color:#8a97a7; font-size:10px; margin-top:4px; }
.content-card {
  background:#fff; border:1px solid var(--studio-line); border-radius:15px; padding:18px;
  box-shadow:0 2px 10px rgba(17,36,58,.025);
}
.pipeline-flow { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:9px; margin:15px 0 4px; }
.flow-node {
  border:1px solid #e5ebf2; border-radius:12px; background:#fbfcfe; padding:12px 10px;
  display:flex; gap:9px; align-items:center; min-width:0;
}
.flow-index { width:25px; height:25px; flex:0 0 25px; display:grid; place-items:center; border-radius:8px; background:#edf2f8; color:#51677f; font-size:10px; font-weight:800; }
.flow-node.done { border-color:#cce9e3; background:#f4fbf9; }
.flow-node.done .flow-index { background:#d9f2eb; color:#087c69; }
.flow-label { font-size:11px; font-weight:750; color:#27394e; }
.flow-detail { font-size:9px; color:#8996a6; margin-top:2px; }
.agent-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:13px; }
.agent-card { border:1px solid var(--studio-line); border-radius:15px; background:#fff; padding:18px; }
.agent-icon { width:36px; height:36px; display:grid; place-items:center; border-radius:11px; margin-bottom:13px; }
.agent-icon.blue { background:#eaf1fe; color:#2c68ce; }
.agent-icon.teal { background:#e3f6f1; color:#0b8c77; }
.agent-name { color:var(--studio-ink); font-size:14px; font-weight:760; }
.agent-description { color:var(--studio-muted); font-size:11px; line-height:1.55; margin-top:5px; }
.agent-steps { display:flex; flex-wrap:wrap; gap:5px; margin-top:12px; }
.agent-step { padding:5px 8px; border-radius:6px; background:#f3f6fa; color:#596a7f; font-size:9px; font-weight:650; }
.status-pill {
  display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 9px;
  font-size:10px; font-weight:750; background:#e9f7f2; color:#087c69;
}
.status-pill.pending { background:#f1f4f7; color:#748397; }
.status-pill.warning { background:#fff5e5; color:#9c6716; }
#studio-tabs { background:transparent !important; }
#studio-tabs button[role="tab"] {
  border:0 !important; border-bottom:2px solid transparent !important; border-radius:0 !important;
  background:transparent !important; color:#6b7b8e !important; padding:13px 16px !important;
  font-weight:700 !important; font-size:12px !important;
}
#studio-tabs button[role="tab"][aria-selected="true"] {
  border-bottom-color:#2867d8 !important; color:#1b3b60 !important;
}
#studio-tabs .tabitem { padding-top:18px !important; }
button.primary {
  background:#2867d8 !important; border:1px solid #2867d8 !important; border-radius:10px !important;
  box-shadow:0 5px 12px rgba(40,103,216,.13) !important; font-weight:750 !important;
}
button.primary:hover { background:#1e56ba !important; }
button.secondary { border-radius:10px !important; }
label, .label-wrap span { color:#34465c !important; font-weight:650 !important; }
textarea, input, .file-preview, .wrap[data-testid="block-info"] {
  border-radius:9px !important;
}
/* Normalize Gradio's default Markdown/group chrome to the Studio design system. */
.gradio-container .prose h3, .gradio-container .prose h2 {
  color:var(--studio-ink) !important; font-size:14px !important; font-weight:760 !important;
  letter-spacing:-.2px !important; margin:0 0 12px !important; padding:0 !important;
  background:transparent !important; border:0 !important;
}
.gradio-container .block, .gradio-container .form {
  border-color:var(--studio-line) !important;
}
.gradio-container .wrap[data-testid="block-info"],
.gradio-container [data-testid="form"] {
  border-radius:11px !important;
}
.gradio-container .label-wrap span { color:#53657a !important; }
.gradio-container textarea, .gradio-container input, .gradio-container select {
  border-color:#dfe6ee !important; background:#fff !important;
}
.gradio-container .file-preview, .gradio-container .upload-container {
  border-radius:10px !important; border-color:#dfe6ee !important;
}
.output-card { border:1px solid var(--studio-line); border-radius:12px; background:#fff; padding:15px; }
.output-title { font-size:11px; font-weight:800; letter-spacing:.7px; color:#738398; text-transform:uppercase; }
.output-value { color:#1d344d; font-size:15px; font-weight:760; margin-top:6px; }
.empty-state {
  border:1px dashed #cbd5e1; border-radius:14px; background:#fbfcfe; padding:22px;
  color:#697b90; text-align:center; font-size:12px; line-height:1.6;
}
.helper-note { color:#8290a0; font-size:10px; line-height:1.55; }
#studio-footer { color:#8a97a7; font-size:10px; padding:7px 2px; border-top:1px solid #e6ebf1; }
.context-banner {
  display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;
  border:1px solid #cfe8e1; border-radius:12px; background:#f3fbf8;
  padding:11px 14px; margin:0 0 14px; color:#176d5f; font-size:11px;
}
.context-banner span { color:#58796f; font-size:10px; }
.context-empty { background:#fbfcfe; border-color:#dbe3ec; color:#56677b; }
.context-empty span { color:#7a8999; }
.status-pill.warning .dot { background:#d99527; }
.panel-title {
  color:var(--studio-ink); font-size:13px; font-weight:780; letter-spacing:-.2px;
  margin:0 0 13px; padding:0 0 10px; border-bottom:1px solid var(--studio-line);
}
.gradio-container .gradio-group,
.gradio-container .form,
.gradio-container .block.gradio-group {
  background:#fff !important; border:1px solid var(--studio-line) !important;
  border-radius:14px !important; box-shadow:0 2px 10px rgba(17,36,58,.025) !important;
  padding:15px !important; gap:12px !important;
}
.gradio-container .prose { color:var(--studio-ink) !important; }
.gradio-container .prose h1, .gradio-container .prose h2,
.gradio-container .prose h3, .gradio-container .prose h4 {
  background:transparent !important; border:0 !important; color:var(--studio-ink) !important;
  box-shadow:none !important; padding:0 !important; margin:0 0 12px !important;
  font-size:14px !important; font-weight:780 !important;
}
.gradio-container [data-testid="dataframe"] {
  border:1px solid var(--studio-line) !important; border-radius:10px !important;
  overflow:hidden !important;
}
.gradio-container [data-testid="accordion"] {
  border:1px solid var(--studio-line) !important; border-radius:11px !important;
  background:#fff !important; margin-bottom:9px !important;
}
.gradio-container button, .gradio-container input, .gradio-container textarea,
.gradio-container select { font-size:13px !important; }
.gradio-container .wrap .label-wrap { margin-bottom:6px !important; }
.gradio-container .secondary { background:#f8fafc !important; }
@media (max-width: 850px) {
  .gradio-container { padding:12px !important; }
  #studio-topbar { align-items:flex-start; flex-direction:column; }
  .hero-title { font-size:24px; }
  #studio-hero { padding:22px; }
  .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .pipeline-flow { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .agent-grid { grid-template-columns:1fr; }
}
"""


def _runtime_label() -> str:
    requested = os.getenv("HF_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "CPU · Transformers FP32"
    try:
        import torch
        if torch.cuda.is_available() and requested != "cpu":
            name = torch.cuda.get_device_name(0)
            return f"GPU · {name} · FP16"
    except Exception:
        pass
    return "CPU · Transformers FP32"


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _read_pipeline() -> dict[str, Any] | None:
    try:
        return get_current_pipeline()
    except Exception:
        return None


def _analysis_context_status(manifest: dict[str, Any] | None = None) -> str:
    manifest = manifest if manifest is not None else _read_pipeline()
    if not manifest:
        return (
            '<div class="context-banner context-empty"><b>No active Gold dataset</b>'
            '<span>Run Data Engineering to prepare a dataset before analysis.</span></div>'
        )
    dataset = _escape(manifest.get("dataset", "Dataset"))
    pipeline_id = _escape(manifest.get("pipeline_id", "—"))
    rows = int(manifest.get("rows_after", 0))
    return (
        '<div class="context-banner">'
        f'<b>Active dataset · {dataset}</b>'
        f'<span>{rows:,} curated rows · pipeline {pipeline_id} · Gold.silver_data</span>'
        '</div>'
    )


def _pipeline_dashboard(manifest: dict[str, Any] | None = None) -> str:
    manifest = manifest if manifest is not None else _read_pipeline()
    ready = bool(manifest and manifest.get("status") == "READY")
    if ready:
        dataset = _escape(manifest.get("dataset", "Dataset"))
        pipeline_id = _escape(manifest.get("pipeline_id", "—"))
        rows_before = int(manifest.get("rows_before", 0))
        rows_after = int(manifest.get("rows_after", 0))
        duplicates = int(manifest.get("duplicates_removed", 0))
        columns = len(manifest.get("columns", []))
        created = _escape(manifest.get("created_at", ""))
        stamp = created.replace("T", " ").split(".")[0].replace("+00:00", " UTC")
        status_html = '<span class="status-pill"><span class="dot"></span>Pipeline ready</span>'
        kpis = f"""
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Active dataset</div><div class="kpi-value">{dataset}</div><div class="kpi-foot">Current pipeline source</div></div>
          <div class="kpi-card"><div class="kpi-label">Curated records</div><div class="kpi-value">{rows_after:,}</div><div class="kpi-foot">From {rows_before:,} source rows</div></div>
          <div class="kpi-card"><div class="kpi-label">Columns profiled</div><div class="kpi-value">{columns}</div><div class="kpi-foot">Semantic metadata available</div></div>
          <div class="kpi-card"><div class="kpi-label">Duplicates removed</div><div class="kpi-value">{duplicates:,}</div><div class="kpi-foot">Exact duplicate records</div></div>
        </div>
        """
        flow = """
        <div class="pipeline-flow">
          <div class="flow-node done"><div class="flow-index">01</div><div><div class="flow-label">Source</div><div class="flow-detail">Captured</div></div></div>
          <div class="flow-node done"><div class="flow-index">02</div><div><div class="flow-label">Bronze</div><div class="flow-detail">Raw copy</div></div></div>
          <div class="flow-node done"><div class="flow-index">03</div><div><div class="flow-label">Silver</div><div class="flow-detail">Cleaned</div></div></div>
          <div class="flow-node done"><div class="flow-index">04</div><div><div class="flow-label">Gold</div><div class="flow-detail">Query ready</div></div></div>
          <div class="flow-node done"><div class="flow-index">05</div><div><div class="flow-label">Manifest</div><div class="flow-detail">Published</div></div></div>
        </div>
        """
        footer = f'<div class="helper-note" style="margin-top:10px">Pipeline ID: <b>{pipeline_id}</b> &nbsp;·&nbsp; Created: {stamp}</div>'
    else:
        status_html = '<span class="status-pill pending"><span class="dot gray"></span>No active pipeline</span>'
        kpis = """
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Active dataset</div><div class="kpi-value">—</div><div class="kpi-foot">Run Data Engineering first</div></div>
          <div class="kpi-card"><div class="kpi-label">Curated records</div><div class="kpi-value">—</div><div class="kpi-foot">No Gold table yet</div></div>
          <div class="kpi-card"><div class="kpi-label">Columns profiled</div><div class="kpi-value">—</div><div class="kpi-foot">Awaiting source data</div></div>
          <div class="kpi-card"><div class="kpi-label">Duplicates removed</div><div class="kpi-value">—</div><div class="kpi-foot">Not evaluated</div></div>
        </div>
        """
        flow = """
        <div class="pipeline-flow">
          <div class="flow-node"><div class="flow-index">01</div><div><div class="flow-label">Source</div><div class="flow-detail">Waiting</div></div></div>
          <div class="flow-node"><div class="flow-index">02</div><div><div class="flow-label">Bronze</div><div class="flow-detail">Waiting</div></div></div>
          <div class="flow-node"><div class="flow-index">03</div><div><div class="flow-label">Silver</div><div class="flow-detail">Waiting</div></div></div>
          <div class="flow-node"><div class="flow-index">04</div><div><div class="flow-label">Gold</div><div class="flow-detail">Waiting</div></div></div>
          <div class="flow-node"><div class="flow-index">05</div><div><div class="flow-label">Manifest</div><div class="flow-detail">Waiting</div></div></div>
        </div>
        """
        footer = '<div class="helper-note" style="margin-top:10px">Your analysis workspace will become available once the first pipeline run is complete.</div>'

    return f"""
    <div class="content-card">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">
        <div><div class="section-kicker">Pipeline monitor</div><div class="section-heading">Your data workspace</div></div>
        {status_html}
      </div>
      {kpis}
      <div style="height:1px;background:#edf1f5;margin:16px 0 8px"></div>
      <div class="section-kicker">Data lifecycle</div>
      {flow}
      {footer}
    </div>
    """


def _overview_refresh() -> str:
    return _pipeline_dashboard()


def _suggested_questions(columns: list[str]) -> list[str]:
    """Build conservative question ideas from the active dataset's real columns."""
    if not columns:
        return []
    first = str(columns[0])
    ideas = [
        "How many rows are in this dataset?",
        f'How many non-empty values are in "{first}"?',
        f'How many distinct values are in "{first}"?',
        "Preview the first 10 rows.",
    ]
    return ideas


def _engineering_handler(
    upload: Any,
    trim_text: bool,
    drop_duplicates: bool,
    drop_empty_rows: bool,
    business_context: str,
):
    upload_path = (
        upload if isinstance(upload, str)
        else getattr(upload, "name", str(upload)) if upload
        else str(SAMPLE_SALES_CSV)
    )
    try:
        result = run_data_engineering(
            upload_path,
            trim_text=trim_text,
            drop_duplicates=drop_duplicates,
            drop_empty_rows=drop_empty_rows,
            business_context=business_context,
        )
        report = result["report"]
        rows_before = int(report.get("rows_before", 0))
        rows_after = int(report.get("rows_after", 0))
        removed = int(report.get("duplicates_removed", 0))
        is_fallback = result["execution_mode"] != "crewai_tool"
        orchestration = result.get("orchestration_status", "not_completed")
        status_class = "warning" if is_fallback or orchestration != "completed" else ""
        status_label = "Python fallback used" if is_fallback else "CrewAI tool invoked"
        heading = "Pipeline artifacts published locally"
        orchestration_label = (
            "Completed" if orchestration == "completed" else
            "Tool invoked; agent did not complete" if orchestration == "failed_after_tool" else
            "Not completed"
        )
        summary = f"""
<div class="content-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
    <div><div class="section-kicker">Pipeline execution</div><div class="section-heading">{heading}</div></div>
    <span class="status-pill {status_class}"><span class="dot"></span>{status_label}</span>
  </div>
  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Source rows</div><div class="kpi-value">{rows_before:,}</div></div>
    <div class="kpi-card"><div class="kpi-label">Silver rows</div><div class="kpi-value">{rows_after:,}</div></div>
    <div class="kpi-card"><div class="kpi-label">Duplicates removed</div><div class="kpi-value">{removed:,}</div></div>
    <div class="kpi-card"><div class="kpi-label">ETL executor</div><div class="kpi-value" style="font-size:15px">{_escape("Python")}</div></div>
  </div>
  <div class="helper-note">Pipeline ID: <b>{_escape(report.get("pipeline_id"))}</b> · Gold table: <b>silver_data</b> · CrewAI orchestration: <b>{_escape(orchestration_label)}</b></div>
</div>
"""
        if result.get("crew_error"):
            summary += f'<div class="helper-note" style="margin-top:8px">CrewAI note: {_escape(result["crew_error"])}</div>'
        manifest_view = {
            "pipeline_id": report.get("pipeline_id"),
            "status": report.get("status"),
            "dataset": report.get("dataset"),
            "layers": report.get("layers"),
            "database": report.get("database"),
            "analysis_table": report.get("analysis_table"),
            "created_at": report.get("created_at"),
        }
        return (
            summary,
            report,
            result["cleaned_file"],
            result["preview"],
            manifest_view,
            _pipeline_dashboard(report),
            gr.update(choices=_suggested_questions(report.get("columns", [])), value=None),
            _analysis_context_status(report),
            gr.update(interactive=True),
            "",
            pd.DataFrame(),
            None,
            '<div class="empty-state">The active dataset changed. Ask a new question to generate results for this pipeline.</div>',
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return (
            f'<div class="content-card"><span class="status-pill warning">Pipeline needs attention</span><p>{_escape(message)}</p></div>',
            {"error": message},
            None,
            pd.DataFrame(),
            {"error": message},
            _pipeline_dashboard(),
            gr.update(choices=[], value=None),
            _analysis_context_status(),
            gr.update(interactive=bool(_read_pipeline())),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _analysis_handler(question: str):
    try:
        result = run_data_analysis(question)
        frame = result["dataframe"]
        figure = None
        if not frame.empty and result.get("execution_mode") not in {"safe_preview_fallback", "validated_query_plan"} and len(frame.index) > 1:
            numeric = [
                column for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
            ]
            categorical = [
                column for column in frame.columns
                if not pd.api.types.is_numeric_dtype(frame[column])
            ]
            if numeric and categorical:
                figure = px.bar(
                    frame,
                    x=categorical[0],
                    y=numeric[0],
                    title="Analysis result",
                    template="plotly_white",
                )
                figure.update_layout(
                    margin=dict(l=16, r=16, t=54, b=16),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#53657a"),
                )

        answer = result["summary"]
        note = result.get("provider_note") or ""
        mode = result.get("execution_mode", "unknown")
        status_labels = {
            "crewai_tool": ("Read-only query executed", ""),
            "validated_query_plan": ("Deterministic query plan executed", ""),
            "safe_preview_fallback": ("Safe preview only", "warning"),
        }
        status_label, status_class = status_labels.get(mode, ("Execution status unknown", "warning"))
        result_kind = result.get("result_kind", "query_result")
        heading = {
            "verified_aggregate": "Verified aggregate from your Gold dataset",
            "query_result": "Read-only query result · inspect the interpretation",
            "preview": "Dataset preview · not a business answer",
        }.get(result_kind, "Read-only query result")
        summary = f"""
<div class="content-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
    <div><div class="section-kicker">Analysis result</div><div class="section-heading">{heading}</div></div>
    <span class="status-pill {status_class}"><span class="dot"></span>{status_label}</span>
  </div>
  <div style="color:#34465c;font-size:13px;line-height:1.7;margin-top:12px">{_escape(answer)}</div>
  <div class="helper-note" style="margin-top:12px">{_escape(note)} · Execution: {_escape(mode)}</div>
</div>
"""
        if result.get("crew_error"):
            summary += f'<div class="helper-note" style="margin-top:8px">Agent note: {_escape(result["crew_error"])}</div>'
        return result.get("sql", ""), frame, figure, summary
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return "", pd.DataFrame(), None, (
            f'<div class="content-card"><span class="status-pill warning">Analysis could not run</span><p>{_escape(message)}</p><div class="helper-note">Run the Data Engineering pipeline first, then retry your question.</div></div>'
        )


# ---------------------------------------------------------------------------
# Application composition
# ---------------------------------------------------------------------------

runtime_label = _escape(_runtime_label())
with gr.Blocks(
    title="Data Agent Platform Studio",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
    ),
    css=CSS,
) as demo:
    with gr.Column(elem_id="studio-shell"):
        gr.HTML(
            f"""
            <div id="studio-topbar">
              <div class="brand-lockup">
                <div class="brand-mark">DA</div>
                <div><div class="brand-name">Data Agent Studio</div><div class="brand-sub">Independent data intelligence workspace</div></div>
              </div>
              <div class="topbar-right">
                <span class="top-chip"><span class="dot"></span>Local model · {_escape(MODEL_ID.rsplit("/", 1)[-1])}</span>
                <span class="top-chip">{runtime_label}</span>
                <span class="top-chip">CrewAI · bounded Python tools</span>
              </div>
            </div>
            """
        )

        with gr.Tabs(elem_id="studio-tabs"):
            with gr.Tab("Overview", id="overview"):
                gr.HTML(
                    """
                    <div id="studio-hero">
                      <div class="hero-eyebrow">Data Agent Platform Studio</div>
                      <div class="hero-title">Your data. One pipeline.<br/>Two intelligent specialists.</div>
                      <div class="hero-copy">Prepare trusted data once, then explore it with a local AI analyst. A shared Bronze → Silver → Gold pipeline keeps both agents grounded in the same dataset and lineage.</div>
                      <div class="hero-meta">
                        <span class="hero-tag">01 · Data Engineering</span>
                        <span class="hero-tag">02 · Data Analysis</span>
                        <span class="hero-tag">Local inference · Colab GPU</span>
                      </div>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-kicker">Workspace at a glance</div><div class="section-heading">Pipeline overview</div><div class="section-sub">The active, versioned data product shared by both workspaces.</div>')
                    with gr.Column(scale=0, min_width=130):
                        refresh_overview = gr.Button("Refresh status", size="sm", variant="secondary")
                overview_dashboard = gr.HTML(value=_pipeline_dashboard())
                refresh_overview.click(_overview_refresh, inputs=[], outputs=[overview_dashboard])

                gr.HTML(
                    """
                    <div style="margin:18px 0 10px"><div class="section-kicker">Your AI team</div><div class="section-heading">Two agents. One shared context.</div><div class="section-sub">Each specialist has a dedicated responsibility and a clear hand-off.</div></div>
                    <div class="agent-grid">
                      <div class="agent-card">
                        <div class="agent-icon blue">
                          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 5h16M4 12h16M4 19h16"/><path d="m8 2-4 3 4 3M16 9l4 3-4 3M8 16l-4 3 4 3"/></svg>
                        </div>
                        <div class="agent-name">Data Engineering Agent</div>
                        <div class="agent-description">Ingests, profiles, cleans and publishes the shared data pipeline with quality and semantic metadata.</div>
                        <div class="agent-steps"><span class="agent-step">Ingest</span><span class="agent-step">Profile</span><span class="agent-step">Transform</span><span class="agent-step">Publish</span></div>
                      </div>
                      <div class="agent-card">
                        <div class="agent-icon teal">
                          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19V5m0 14h17"/><path d="m7 15 4-4 3 2 5-7"/></svg>
                        </div>
                        <div class="agent-name">Data Analysis Agent</div>
                        <div class="agent-description">Uses the active Gold dataset, understands its schema and business context, then answers with governed SQL.</div>
                        <div class="agent-steps"><span class="agent-step">Understand</span><span class="agent-step">Query</span><span class="agent-step">Analyze</span><span class="agent-step">Explain</span></div>
                      </div>
                    </div>
                    """
                )

            with gr.Tab("Data Engineering", id="engineering"):
                gr.HTML(
                    """
                    <div style="margin-bottom:18px">
                      <div class="section-kicker">Workspace 01 / Engineering</div>
                      <div class="section-heading">Prepare and publish trusted data</div>
                      <div class="section-sub">Ingest a CSV, apply transparent quality rules, review the results and publish an immutable Gold run for downstream analysis.</div>
                    </div>
                    <div class="pipeline-flow" style="margin-bottom:18px">
                      <div class="flow-node"><div class="flow-index">01</div><div><div class="flow-label">Source</div><div class="flow-detail">Upload CSV</div></div></div>
                      <div class="flow-node"><div class="flow-index">02</div><div><div class="flow-label">Bronze</div><div class="flow-detail">Preserve raw</div></div></div>
                      <div class="flow-node"><div class="flow-index">03</div><div><div class="flow-label">Silver</div><div class="flow-detail">Clean & profile</div></div></div>
                      <div class="flow-node"><div class="flow-index">04</div><div><div class="flow-label">Gold</div><div class="flow-detail">Query ready</div></div></div>
                      <div class="flow-node"><div class="flow-index">05</div><div><div class="flow-label">Manifest</div><div class="flow-detail">Context hand-off</div></div></div>
                    </div>
                    """
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=330):
                        with gr.Group(elem_id="engineering-source-panel"):
                            gr.HTML('<div class="panel-title">01 · Source and business context</div>')
                            de_file = gr.File(
                                label="Upload a CSV source",
                                file_types=[".csv"],
                                type="filepath",
                            )
                            gr.Markdown('<div class="helper-note">No file? Leave this empty to use the bundled Retail Sales demo dataset. Maximum upload size: 25 MB.</div>')
                            de_context = gr.Textbox(
                                label="Business description",
                                placeholder="Example: Retail sales by product, region and order date. Revenue is net sales in TND.",
                                lines=3,
                            )
                        with gr.Group(elem_id="engineering-rules-panel"):
                            gr.HTML('<div class="panel-title">02 · Data quality rules</div>')
                            de_trim = gr.Checkbox(value=True, label="Trim whitespace in text fields")
                            de_duplicates = gr.Checkbox(value=True, label="Remove exact duplicate rows")
                            de_empty = gr.Checkbox(value=True, label="Remove fully empty rows")
                            gr.Markdown('<div class="helper-note">These rules are deterministic Python operations. The agent orchestrates the pipeline and summarizes the verified output.</div>')
                        de_run = gr.Button("Run data pipeline", variant="primary", size="lg")
                    with gr.Column(scale=7, min_width=360):
                        gr.HTML('<div class="panel-title">Pipeline execution</div>')
                        de_summary = gr.HTML(value='<div class="empty-state">Your execution summary will appear here after the pipeline runs.</div>')
                        with gr.Accordion("Quality report and column profile", open=False):
                            de_report = gr.JSON(label="Pipeline report")
                        with gr.Accordion("Silver data preview", open=True):
                            de_preview = gr.Dataframe(label="Curated records", interactive=False)
                            de_download = gr.File(label="Download cleaned CSV")
                        with gr.Accordion("Active pipeline manifest", open=False):
                            de_manifest = gr.JSON(label="Bronze / Silver / Gold lineage")

            with gr.Tab("Data Analysis", id="analysis"):
                gr.HTML(
                    """
                    <div style="margin-bottom:18px">
                      <div class="section-kicker">Workspace 02 / Analysis</div>
                      <div class="section-heading">Explore your active dataset</div>
                      <div class="section-sub">Ask questions against the exact Gold dataset published by Data Engineering. No second upload and no disconnected sample data.</div>
                    </div>
                    """
                )
                da_context_status = gr.HTML(value=_analysis_context_status())
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=330):
                        with gr.Group(elem_id="analysis-question-panel"):
                            gr.HTML('<div class="panel-title">Ask a business question</div>')
                            da_question = gr.Textbox(
                                label="Your question",
                                placeholder="e.g. How many rows are in this dataset?",
                                lines=4,
                            )
                            da_question_ideas = gr.Dropdown(
                                label="Question ideas · based on active dataset",
                                choices=[],
                                value=None,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            da_question_ideas.change(
                                lambda value: value or "",
                                inputs=[da_question_ideas],
                                outputs=[da_question],
                            )
                            da_run = gr.Button(
                                "Analyze dataset",
                                variant="primary",
                                size="lg",
                                interactive=bool(_read_pipeline()),
                            )
                        gr.HTML(
                            """
                            <div class="content-card" style="margin-top:12px">
                              <div class="section-kicker">How it works</div>
                              <div style="font-size:12px;line-height:1.8;color:#5d6e81">
                                <b>01</b> Read pipeline manifest<br/>
                                <b>02</b> Inspect real schema and semantic metadata<br/>
                                <b>03</b> Generate one SELECT query<br/>
                                <b>04</b> Execute with read-only controls
                              </div>
                            </div>
                            """
                        )
                    with gr.Column(scale=7, min_width=360):
                        gr.HTML('<div class="panel-title">Agent response</div>')
                        da_summary = gr.HTML(value='<div class="empty-state">Your answer, query and result visualization will appear here.</div>')
                        with gr.Accordion("Executed read-only SQL", open=False):
                            da_sql = gr.Code(label="Read-only SQL", language="sql")
                        with gr.Accordion("Query result and records", open=True):
                            da_table = gr.Dataframe(label="Result set", interactive=False)
                        da_plot = gr.Plot(label="Visualization")

                da_run.click(
                    _analysis_handler,
                    inputs=[da_question],
                    outputs=[da_sql, da_table, da_plot, da_summary],
                )

            with gr.Tab("Platform", id="platform"):
                gr.HTML(
                    """
                    <div style="margin-bottom:18px">
                      <div class="section-kicker">Platform / Architecture</div>
                      <div class="section-heading">A controlled local agent runtime</div>
                      <div class="section-sub">The Studio keeps data operations explicit and the model replaceable.</div>
                    </div>
                    <div class="content-card">
                      <div class="section-kicker">Execution stack</div>
                      <div class="pipeline-flow">
                        <div class="flow-node done"><div class="flow-index">01</div><div><div class="flow-label">Gradio</div><div class="flow-detail">User experience</div></div></div>
                        <div class="flow-node done"><div class="flow-index">02</div><div><div class="flow-label">CrewAI</div><div class="flow-detail">Orchestration</div></div></div>
                        <div class="flow-node done"><div class="flow-index">03</div><div><div class="flow-label">Qwen 1.5B</div><div class="flow-detail">Local inference</div></div></div>
                        <div class="flow-node done"><div class="flow-index">04</div><div><div class="flow-label">Python tools</div><div class="flow-detail">Execution</div></div></div>
                        <div class="flow-node done"><div class="flow-index">05</div><div><div class="flow-label">SQLite</div><div class="flow-detail">Gold storage</div></div></div>
                      </div>
                    </div>
                    <div class="agent-grid" style="margin-top:14px">
                      <div class="agent-card"><div class="agent-name">Model boundary</div><div class="agent-description">Qwen2.5-1.5B-Instruct runs locally through Transformers on CUDA (FP16) or CPU (FP32). CPU inference is slower; Python tools perform bounded data operations.</div></div>
                      <div class="agent-card"><div class="agent-name">Data safety</div><div class="agent-description">The analysis tool accepts one SELECT/WITH statement, opens SQLite read-only, applies a SQLite authorizer and caps results at 100 rows.</div></div>
                    </div>
                    <div class="content-card" style="margin-top:14px">
                      <div class="section-kicker">Runtime note</div>
                      <div class="section-sub">CUDA/T4 is the acceleration target, but GPU allocation depends on runtime availability. CPU-only execution is supported with FP32 and is slower. Gradio share links are public to anyone who obtains them. Use synthetic data for this lab.</div>
                    </div>
                    """
                )

        de_run.click(
            _engineering_handler,
            inputs=[de_file, de_trim, de_duplicates, de_empty, de_context],
            outputs=[
                de_summary, de_report, de_download, de_preview, de_manifest,
                overview_dashboard, da_question_ideas, da_context_status, da_run,
                da_sql, da_table, da_plot, da_summary,
            ],
        )

        gr.HTML(
            """
            <div id="studio-footer">
              DATA AGENT PLATFORM STUDIO &nbsp;·&nbsp; INDEPENDENT LOCAL AGENTS &nbsp;·&nbsp; CREWAI + HUGGING FACE
            </div>
            """
        )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
    )
