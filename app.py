from __future__ import annotations

import os
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px

from studio.config import SAMPLE_SALES_CSV
from studio.scenarios import run_data_analysis, run_data_engineering


CSS = """
.gradio-container {max-width: 1420px !important; margin: auto !important;}
#studio-header {padding: 20px 26px; border-radius: 16px; background: linear-gradient(110deg,#10253f,#1e4267); color: white;}
#studio-header h1 {color: white; margin-bottom: 6px;}
.scenario-note {border-left: 4px solid #d79a2b; padding: 10px 14px; background: #f8f4eb;}
"""


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
        mode_note = (
            "CrewAI called the bounded pipeline tool."
            if result["execution_mode"] == "crewai_tool"
            else "The deterministic Python pipeline ran because CrewAI did not complete the tool call."
        )
        if result.get("crew_error"):
            mode_note += "\n\nCrewAI/model detail: " + result["crew_error"]
        return (
            result["report"],
            result["cleaned_file"],
            result["preview"],
            result["report"],
            result["summary"] + "\n\n**Execution:** " + mode_note,
        )
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}"}
        return error, None, pd.DataFrame(), error, "Pipeline failed."


def _analysis_handler(question: str):
    try:
        result = run_data_analysis(question)
        frame = result["dataframe"]
        figure = None
        if not frame.empty:
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
                    title="Data Analysis Agent · Gold dataset",
                    template="plotly_white",
                )

        summary = result["summary"]
        note = result.get("provider_note") or ""
        if note:
            summary += "\n\n**Data context:** " + note
        if result.get("crew_error"):
            summary += "\n\n**CrewAI/model detail:** " + result["crew_error"]
        summary += "\n\n**Execution mode:** " + result["execution_mode"]
        return result.get("sql", ""), frame, figure, summary
    except Exception as exc:
        return "", pd.DataFrame(), None, (
            f"**Analysis failed:** {type(exc).__name__}: {exc}"
        )


with gr.Blocks(
    title="Data Agent Platform Studio",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        <div id="studio-header">
        <h1>Data Agent Platform Studio</h1>
        <p>One shared data pipeline · Two specialized CrewAI agents · Local Hugging Face model · Colab T4</p>
        </div>
        """
    )
    gr.Markdown(
        "**One dataset, one lineage, two agents.** The Data Engineering Agent prepares "
        "the shared Bronze/Silver/Gold pipeline. The Data Analysis Agent consumes that "
        "exact Gold dataset. Both agents run locally through CrewAI and the same "
        "Hugging Face model on the Colab GPU.",
        elem_classes=["scenario-note"],
    )

    with gr.Tabs():
        with gr.Tab("01 · Data Engineering Agent"):
            gr.Markdown("## Prepare and publish the shared data pipeline")
            gr.Markdown(
                "Upload one CSV or use the built-in Retail Sales example. This run creates "
                "the active dataset manifest that the Data Analysis Agent will consume."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    de_file = gr.File(
                        label="Source CSV (optional — use Retail Sales demo when empty)",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    de_trim = gr.Checkbox(value=True, label="Trim text fields")
                    de_duplicates = gr.Checkbox(value=True, label="Remove exact duplicates")
                    de_empty = gr.Checkbox(value=True, label="Remove fully empty rows")
                    de_context = gr.Textbox(
                        label="Business context / dataset description (optional)",
                        placeholder="Example: Retail sales by product, region and order date. Revenue is net sales in TND.",
                        lines=3,
                    )
                    de_run = gr.Button("Run Data Engineering Agent", variant="primary")
                with gr.Column(scale=2):
                    de_summary = gr.Markdown("Run the pipeline to prepare the shared dataset.")
                    de_report = gr.JSON(label="Pipeline quality report and lineage")
            with gr.Row():
                de_download = gr.File(label="Silver · cleaned CSV")
                de_preview = gr.Dataframe(label="Silver data preview", interactive=False)
            de_manifest = gr.JSON(label="Active pipeline manifest")
            de_run.click(
                _engineering_handler,
                inputs=[de_file, de_trim, de_duplicates, de_empty, de_context],
                outputs=[de_report, de_download, de_preview, de_manifest, de_summary],
            )

        with gr.Tab("02 · Data Analysis Agent"):
            gr.Markdown("## Ask business questions about the active Gold dataset")
            gr.Markdown(
                "This agent does not load a second dataset. It reads the manifest and schema "
                "published by the Data Engineering Agent, generates a query for the actual "
                "columns, and executes it through the read-only SQL tool."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    da_question = gr.Textbox(
                        label="Business question",
                        value="What is the revenue by region in the first quarter?",
                        lines=3,
                    )
                    gr.Examples(
                        examples=[
                            ["What is the revenue by region in the first quarter?"],
                            ["Show total sales by product."],
                            ["Which region has the highest revenue?"],
                            ["Show the 5 rows with the highest revenue."],
                            ["Summarize the columns and data quality of this dataset."],
                        ],
                        inputs=[da_question],
                    )
                    da_run = gr.Button("Run Data Analysis Agent", variant="primary")
                with gr.Column(scale=2):
                    da_summary = gr.Markdown("Analysis result will appear here.")
                    da_sql = gr.Code(label="Generated SQL · read only", language="sql")
            da_table = gr.Dataframe(label="Query result", interactive=False)
            da_plot = gr.Plot(label="Result visualization")
            da_run.click(
                _analysis_handler,
                inputs=[da_question],
                outputs=[da_sql, da_table, da_plot, da_summary],
            )

    gr.Markdown(
        """
        ---
        **Platform capabilities:** shared pipeline manifest, schema and semantic metadata,
        user-supplied business context, data-quality profiling, source-to-Gold lineage,
        bounded CrewAI tools, SQL read-only policy, result limits and execution trace.
        Both agents run independently on the local Colab GPU; no external agent service is called.
        """
    )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
    )
