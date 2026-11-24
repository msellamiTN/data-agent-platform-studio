from __future__ import annotations

import os
import time
from typing import Any

import requests

from .config import (
    DATABRICKS_GENIE_SPACE_ID,
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    SNOWFLAKE_AGENT_NAME,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_HOST,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_TOKEN,
)


def provider_configuration() -> dict[str, dict[str, Any]]:
    """Expose configuration state only; never return tokens or credential values."""
    return {
        "independent": {"configured": True, "missing": []},
        "databricks_genie": {
            "configured": all((DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_GENIE_SPACE_ID)),
            "missing": [
                name for name, value in (
                    ("DATABRICKS_HOST", DATABRICKS_HOST),
                    ("DATABRICKS_TOKEN", DATABRICKS_TOKEN),
                    ("DATABRICKS_GENIE_SPACE_ID", DATABRICKS_GENIE_SPACE_ID),
                ) if not value
            ],
        },
        "snowflake_cortex": {
            "configured": all((
                SNOWFLAKE_HOST, SNOWFLAKE_TOKEN, SNOWFLAKE_DATABASE,
                SNOWFLAKE_SCHEMA, SNOWFLAKE_AGENT_NAME,
            )),
            "missing": [
                name for name, value in (
                    ("SNOWFLAKE_HOST", SNOWFLAKE_HOST),
                    ("SNOWFLAKE_TOKEN", SNOWFLAKE_TOKEN),
                    ("SNOWFLAKE_DATABASE", SNOWFLAKE_DATABASE),
                    ("SNOWFLAKE_SCHEMA", SNOWFLAKE_SCHEMA),
                    ("SNOWFLAKE_AGENT_NAME", SNOWFLAKE_AGENT_NAME),
                ) if not value
            ],
        },
    }


def _extract_genie_answer(message: dict[str, Any]) -> tuple[str, str]:
    answers: list[str] = []
    sql = ""
    for attachment in message.get("attachments", []) or []:
        text_attachment = attachment.get("text") or {}
        if text_attachment.get("content"):
            answers.append(str(text_attachment["content"]))
        query_attachment = attachment.get("query") or {}
        if query_attachment.get("query") and not sql:
            sql = str(query_attachment["query"])
    return "\n\n".join(answers).strip(), sql


def run_databricks_genie(question: str, max_wait_seconds: int = 120) -> dict[str, Any]:
    """Ask a configured Databricks Genie Agent and poll its asynchronous message."""
    cfg = provider_configuration()["databricks_genie"]
    if not cfg["configured"]:
        raise RuntimeError(
            "Databricks Genie is not configured. Missing: " + ", ".join(cfg["missing"])
        )

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    start_url = (
        f"{DATABRICKS_HOST}/api/2.0/genie/spaces/"
        f"{DATABRICKS_GENIE_SPACE_ID}/start-conversation"
    )
    response = requests.post(
        start_url,
        headers=headers,
        json={"content": question, "enable_visualization": True},
        timeout=45,
    )
    response.raise_for_status()
    started = response.json()
    message_id = started.get("message_id") or (started.get("message") or {}).get("message_id")
    message = started.get("message") or {}
    conversation_id = message.get("conversation_id") or started.get("conversation_id")
    if not message_id or not conversation_id:
        raise RuntimeError("Genie start-conversation response did not include message_id and conversation_id.")

    message_url = (
        f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{DATABRICKS_GENIE_SPACE_ID}"
        f"/conversations/{conversation_id}/messages/{message_id}"
    )
    deadline = time.monotonic() + max_wait_seconds
    while True:
        poll = requests.get(message_url, headers=headers, timeout=30)
        poll.raise_for_status()
        message = poll.json()
        status = str(message.get("status", "")).upper()
        if status in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Genie message did not finish within {max_wait_seconds} seconds; "
                f"conversation_id={conversation_id}"
            )
        time.sleep(2)

    if status != "COMPLETED":
        error = message.get("error") or {}
        raise RuntimeError(f"Databricks Genie ended with {status}: {error}")

    answer, sql = _extract_genie_answer(message)
    return {
        "provider": "Databricks Genie",
        "status": status,
        "answer": answer or "Genie completed without a text attachment; inspect the provider conversation.",
        "sql": sql,
        "rows": [],
        "conversation_id": conversation_id,
        "message_id": message_id,
        "note": (
            "Genie queried its configured Unity Catalog assets. "
            "The uploaded Colab pipeline is not automatically copied into Unity Catalog."
        ),
    }


def _collect_snowflake_text(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"text", "content"} and isinstance(item, str):
                texts.append(item)
            else:
                texts.extend(_collect_snowflake_text(item))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_collect_snowflake_text(item))
    return texts


def _find_sql(value: Any) -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"sql", "query"} and isinstance(item, str):
                if item.strip().upper().startswith(("SELECT", "WITH")):
                    return item
            found = _find_sql(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_sql(item)
            if found:
                return found
    return ""


def run_snowflake_cortex(question: str) -> dict[str, Any]:
    """Invoke a configured Snowflake Cortex Agent object through its REST run API."""
    cfg = provider_configuration()["snowflake_cortex"]
    if not cfg["configured"]:
        raise RuntimeError(
            "Snowflake Cortex is not configured. Missing: " + ", ".join(cfg["missing"])
        )

    base = SNOWFLAKE_HOST
    if not base.startswith(("https://", "http://")):
        base = "https://" + base
    endpoint = (
        f"{base}/api/v2/databases/{SNOWFLAKE_DATABASE}/schemas/{SNOWFLAKE_SCHEMA}"
        f"/agents/{SNOWFLAKE_AGENT_NAME}:run"
    )
    headers = {
        "Authorization": f"Bearer {SNOWFLAKE_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ],
        "stream": False,
        "background": False,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=900)
    response.raise_for_status()
    body = response.json()
    content = body.get("content", [])
    text_parts = _collect_snowflake_text(content)
    answer = "\n".join(part for part in text_parts if part.strip())
    return {
        "provider": "Snowflake Cortex Agent",
        "status": body.get("status", "completed"),
        "answer": answer or "Cortex Agent returned a structured response without a plain-text block.",
        "sql": _find_sql(body),
        "rows": [],
        "conversation_id": str((body.get("metadata") or {}).get("thread_id", "")),
        "warnings": body.get("warnings", []),
        "note": (
            "Cortex used the tools configured on its Snowflake agent object. "
            "The uploaded Colab pipeline is not automatically copied into Snowflake."
        ),
    }


def run_provider_analysis(provider: str, question: str) -> dict[str, Any]:
    if provider == "Databricks Genie":
        return run_databricks_genie(question)
    if provider == "Snowflake Cortex":
        return run_snowflake_cortex(question)
    raise ValueError(f"Unsupported native data-agent provider: {provider}")
