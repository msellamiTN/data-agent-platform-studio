from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px

from studio.config import SAMPLE_SALES_CSV
from studio.scenarios import run_data_analysis, run_data_engineering


CSS = """
.gradio-container {max-width: 1380px !important; margin: auto !important;}
#studio-header {padding: 18px 24px; border-radius: 16px; background: linear-gradient(110deg,#10253f,#1e4267); color: white;}
#studio-header h1 {color: white; margin-bottom: 6px;}
.scenario-note {border-left: 4px solid #d79a2b; padding: 8px 14px; background: #f8f4eb;}
"""


def _engineering_handler(
    upload: Any,
    trim_text: bool,
    drop_duplicates: bool,
    drop_empty_rows: bool,
):
    if not upload:
        return {"error": "Upload a CSV file to start the Data Engineering scenario."}, None, pd.DataFrame(), "No file selected."

    try:
        upload_path = upload if isinstance(upload, str) else getattr(upload, "name", str(upload))
        result = run_data_engineering(
            upload_path,
            trim_text=trim_text,
            drop_duplicates=drop_duplicates,
            drop_empty_rows=drop_empty_rows,
        )
        mode_note = (
            "CrewAI tool executed the ETL pipeline."
            if result["execution_mode"] == "crewai_tool"
            else "CrewAI did not execute the tool; the deterministic Python ETL fallback ran."
        )
        if result.get("crew_error"):
            mode_note += "\n\nCrewAI/model detail: " + result["crew_error"]
        return result["report"], result["cleaned_file"], result["preview"], (
            result["summary"] + "\n\n**Execution:** " + mode_note
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, None, pd.DataFrame(), "ETL failed."


def _analysis_handler(upload: Any, question: str):
    try:
        upload_path = None
        if upload:
            upload_path = upload if isinstance(upload, str) else getattr(upload, "name", str(upload))
        result = run_data_analysis(question, upload_path)
        frame = result["dataframe"]
        figure = None
        if not frame.empty:
            numeric_candidates = [
                column for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
            ]
            category_candidates = [
                column for column in frame.columns
                if not pd.api.types.is_numeric_dtype(frame[column])
            ]
            if numeric_candidates and category_candidates:
                figure = px.bar(
                    frame,
                    x=category_candidates[0],
                    y=numeric_candidates[0],
                    title="Result preview",
                    template="plotly_white",
                )
        mode_note = (
            "CrewAI generated and executed the read-only SQL tool call."
            if result["execution_mode"] == "crewai_tool"
            else "CrewAI did not complete a SQL tool call; a bounded deterministic SELECT fallback ran."
        )
        if result.get("crew_error"):
            mode_note += "\n\nCrewAI/model detail: " + result["crew_error"]
        summary = result["summary"] + "\n\n**Execution:** " + mode_note
        return result["sql"], frame, figure, summary
    except Exception as exc:
        return "", pd.DataFrame(), None, f"**Analysis failed:** {type(exc).__name__}: {exc}"


with gr.Blocks(
    title="Data Agent Platform Studio",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        <div id="studio-header">
        <h1>Data Agent Platform Studio</h1>
        <p>Two focused AI agents · CrewAI orchestration · Hugging Face local model · Colab T4</p>
        </div>
        """
    )
    gr.Markdown(
        "Choose a workspace. Each scenario has its own workflow, inputs and outputs.",
        elem_classes=["scenario-note"],
    )

    with gr.Tabs():
        with gr.Tab("01 · Data Engineering Agent"):
            gr.Markdown("## Prepare and load data")
            gr.Markdown(
                "Upload a CSV. The agent is instructed to call the profiling and cleaning tool; "
                "Python performs deterministic transformations and stages the result in SQLite."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    de_file = gr.File(
                        label="Source CSV",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    de_trim = gr.Checkbox(value=True, label="Trim text fields")
                    de_duplicates = gr.Checkbox(value=True, label="Remove exact duplicates")
                    de_empty = gr.Checkbox(value=True, label="Remove fully empty rows")
                    de_run = gr.Button("Run Data Engineering Agent", variant="primary")
                with gr.Column(scale=2):
                    de_summary = gr.Markdown("ETL status will appear here.")
                    de_report = gr.JSON(label="Quality report")
            de_download = gr.File(label="Cleaned CSV")
            de_preview = gr.Dataframe(label="Cleaned data preview", interactive=False)
            de_run.click(
                _engineering_handler,
                inputs=[de_file, de_trim, de_duplicates, de_empty],
                outputs=[de_report, de_download, de_preview, de_summary],
            )

        with gr.Tab("02 · Data Analysis Agent"):
            gr.Markdown("## Ask questions about the sales data")
            gr.Markdown(
                "Upload a compatible sales CSV or use the included demo dataset. "
                "The agent inspects the schema, proposes SQL and calls a read-only query tool."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    da_file = gr.File(
                        label="Optional sales CSV (leave empty to use demo data)",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    da_question = gr.Textbox(
                        label="Business question",
                        value="What is the revenue by region in the first quarter?",
                        lines=3,
                    )
                    gr.Examples(
                        examples=[
                            ["What is the revenue by region in the first quarter?"],
                            ["Show total revenue by product."],
                            ["Which region has the highest revenue?"],
                        ],
                        inputs=[da_question],
                    )
                    da_run = gr.Button("Run Data Analysis Agent", variant="primary")
                with gr.Column(scale=2):
                    da_summary = gr.Markdown("Analysis output will appear here.")
                    da_sql = gr.Code(label="Generated SQL (read only)", language="sql")
            da_table = gr.Dataframe(label="Query result", interactive=False)
            da_plot = gr.Plot(label="Result visualization")
            da_run.click(
                _analysis_handler,
                inputs=[da_file, da_question],
                outputs=[da_sql, da_table, da_plot, da_summary],
            )

    gr.Markdown(
        """
        ---
        **Lab safety:** CSV uploads are limited to 25 MB. Data Engineering writes only to the
        local workspace. Data Analysis uses a per-run SQLite database, permits SELECT/CTE
        statements only, applies a SQLite authorizer, and returns at most 100 rows.
        """
    )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
    )
