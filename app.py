from __future__ import annotations

import os
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px

from studio.config import SAMPLE_SALES_CSV
from studio.integrations import provider_configuration
from studio.scenarios import LOCAL_PROVIDER, run_data_analysis, run_data_engineering


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
):
    # One pipeline source feeds BOTH the Data Engineering and Data Analysis experiences.
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


def _analysis_handler(provider: str, question: str):
    try:
        result = run_data_analysis(question, provider)
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
                    title=f"Analysis result · {result['provider']}",
                    template="plotly_white",
                )

        note = result.get("provider_note") or ""
        summary = result["summary"]
        if note:
            summary += "\n\n**Data context:** " + note
        if result.get("crew_error"):
            summary += "\n\n**CrewAI/model detail:** " + result["crew_error"]
        if result.get("conversation_id"):
            summary += "\n\n**Provider conversation:** " + result["conversation_id"]
        summary += "\n\n**Execution mode:** " + result["execution_mode"]

        return result.get("sql", ""), frame, figure, summary
    except Exception as exc:
        return "", pd.DataFrame(), None, (
            f"**Analysis failed:** {type(exc).__name__}: {exc}"
        )


def _provider_status():
    status = provider_configuration()
    rows = []
    for key, label in (
        ("independent", "Independent CrewAI + Hugging Face"),
        ("databricks_genie", "Databricks Genie Agent"),
        ("snowflake_cortex", "Snowflake Cortex Agent"),
    ):
        entry = status[key]
        if entry["configured"]:
            rows.append({"provider": label, "status": "Configured", "missing": ""})
        else:
            rows.append({
                "provider": label,
                "status": "Needs configuration",
                "missing": ", ".join(entry["missing"]),
            })
    return pd.DataFrame(rows)


with gr.Blocks(
    title="Data Agent Platform Studio",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        <div id="studio-header">
        <h1>Data Agent Platform Studio</h1>
        <p>One shared data pipeline · Multiple AI agent providers · Colab T4 · Gradio</p>
        </div>
        """
    )
    gr.Markdown(
        "**One dataset, one lineage, multiple agents.** Data Engineering prepares the shared "
        "Bronze/Silver/Gold pipeline; Data Analysis agents consume the curated data through "
        "the selected local or platform-native provider.",
        elem_classes=["scenario-note"],
    )

    with gr.Tabs():
        with gr.Tab("01 · Shared Data Pipeline"):
            gr.Markdown("## Data Engineering Agent → shared pipeline")
            gr.Markdown(
                "Upload a CSV or use the Retail Sales example. Both agents work from the same "
                "active Gold dataset; Data Analysis does not ask for a second, unrelated file."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    de_file = gr.File(
                        label="Source CSV (optional — demo sales dataset is used when empty)",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    de_trim = gr.Checkbox(value=True, label="Trim text fields")
                    de_duplicates = gr.Checkbox(value=True, label="Remove exact duplicates")
                    de_empty = gr.Checkbox(value=True, label="Remove fully empty rows")
                    de_run = gr.Button("Run shared pipeline", variant="primary")
                with gr.Column(scale=2):
                    de_summary = gr.Markdown("Run the pipeline to publish a shared dataset.")
                    de_report = gr.JSON(label="Pipeline quality report and lineage")
            with gr.Row():
                de_download = gr.File(label="Silver · cleaned CSV")
                de_preview = gr.Dataframe(label="Silver data preview", interactive=False)
            de_manifest = gr.JSON(label="Active pipeline manifest")
            de_run.click(
                _engineering_handler,
                inputs=[de_file, de_trim, de_duplicates, de_empty],
                outputs=[de_report, de_download, de_preview, de_manifest, de_summary],
            )

        with gr.Tab("02 · Data Analysis Agent"):
            gr.Markdown("## Analyze the active pipeline")
            gr.Markdown(
                "Ask a business question about the dataset prepared in tab 01. "
                "Choose an independent CrewAI agent or a native data agent that is "
                "already configured against the corresponding curated platform assets."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    da_provider = gr.Dropdown(
                        choices=[
                            LOCAL_PROVIDER,
                            "Databricks Genie",
                            "Snowflake Cortex",
                        ],
                        value=LOCAL_PROVIDER,
                        label="Analysis agent provider",
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
                            ["Show the 5 rows with the highest revenue."],
                        ],
                        inputs=[da_question],
                    )
                    da_run = gr.Button("Run Data Analysis Agent", variant="primary")
                with gr.Column(scale=2):
                    da_summary = gr.Markdown("Analysis result will appear here.")
                    da_sql = gr.Code(label="SQL (when returned by provider)", language="sql")
            da_table = gr.Dataframe(label="Result table", interactive=False)
            da_plot = gr.Plot(label="Result visualization")
            da_run.click(
                _analysis_handler,
                inputs=[da_provider, da_question],
                outputs=[da_sql, da_table, da_plot, da_summary],
            )

        with gr.Tab("03 · Agent Providers & Data Platforms"):
            gr.Markdown("## Provider selection is separate from the data pipeline")
            gr.Markdown(
                """
                | Layer | Databricks | Snowflake | Independent |
                |---|---|---|---|
                | Data engineering | Lakeflow Connect / Pipelines / Jobs; Genie Code for engineering assistance | Snowpark / SQL / Tasks / Dynamic Tables | CrewAI + Python / Pandas / Spark |
                | Data analysis | Genie Agent over Unity Catalog assets | Cortex Agent over configured Analyst / Search / other tools | CrewAI + local Hugging Face model + read-only SQL tools |

                **Important:** Genie and Cortex are native platform agents. They query assets already
                configured in their respective environments; this Colab demo does not silently copy
                the uploaded CSV into Unity Catalog or Snowflake. Publish the matching Silver/Gold
                dataset and configure the provider agent/semantic layer before selecting a native agent.
                """
            )
            provider_table = gr.Dataframe(
                value=_provider_status(),
                label="Provider configuration status (credential values are never displayed)",
                interactive=False,
            )
            gr.Button("Refresh provider status").click(
                _provider_status,
                inputs=[],
                outputs=[provider_table],
            )
            gr.Markdown(
                """
                **Environment variables / Colab Secrets**

                Databricks: DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_GENIE_SPACE_ID.

                Snowflake: SNOWFLAKE_HOST, SNOWFLAKE_TOKEN, SNOWFLAKE_DATABASE,
                SNOWFLAKE_SCHEMA, SNOWFLAKE_AGENT_NAME.

                No token is stored in the repository or displayed in this interface.
                """
            )

    gr.Markdown(
        """
        ---
        **Pipeline contract:** Bronze keeps the uploaded source; Silver stores the cleaned CSV;
        Gold exposes the shared silver_data table and its manifest. The independent agent reads
        this exact Gold table with SELECT-only SQLite controls. Native Genie/Cortex agents run
        against their configured governed platform assets.
        """
    )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
    )
