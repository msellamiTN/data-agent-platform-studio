"""Live acceptance checks for the local Transformers + CrewAI runtime.

These checks intentionally load and call the configured model. They are not part
of the lightweight deterministic CI suite.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_SMOKE_PROMPT = "Reply with exactly the word READY."


def run_runtime_acceptance(
    llm: Any,
    *,
    prompt: str = DEFAULT_SMOKE_PROMPT,
    include_crewai: bool = True,
) -> dict[str, Any]:
    """Run real local inference and, optionally, a real CrewAI tool invocation.

    The caller supplies the shared LLM adapter so a notebook can warm the model
    once and reuse the same cached adapter when the Gradio app starts.
    """
    if not prompt or not prompt.strip():
        raise ValueError("The runtime smoke-test prompt must not be empty.")

    import torch

    inference_started = time.perf_counter()
    response = llm.call(prompt.strip())
    inference_seconds = time.perf_counter() - inference_started
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("The local model returned an empty inference response.")

    model = getattr(llm, "_hf_model", None)
    if model is None:
        raise RuntimeError("The local model call returned without loading model weights.")
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise RuntimeError("The loaded model exposes no parameters to inspect.")

    parameter_device = str(parameter.device)
    parameter_dtype = str(parameter.dtype)
    selected_device = llm.runtime_device
    if selected_device == "cuda" and not parameter_device.startswith("cuda"):
        raise RuntimeError(
            f"CUDA was selected but model parameters are on {parameter_device}."
        )
    if selected_device == "cpu" and parameter_device != "cpu":
        raise RuntimeError(
            f"CPU was selected but model parameters are on {parameter_device}."
        )
    expected_dtype = "torch.float16" if selected_device == "cuda" else "torch.float32"
    if parameter_dtype != expected_dtype:
        raise RuntimeError(
            f"{selected_device} runtime expected {expected_dtype}, got {parameter_dtype}."
        )

    report: dict[str, Any] = {
        "status": "passed",
        "model_id": getattr(llm, "model", "unknown"),
        "torch_version": torch.__version__,
        "selected_device": selected_device,
        "parameter_device": parameter_device,
        "parameter_dtype": parameter_dtype,
        "cuda_available": bool(torch.cuda.is_available()),
        "inference_seconds": round(inference_seconds, 3),
        "model_response": response.strip(),
        "crewai_tool_invocations": None,
    }
    if selected_device == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated(0) / (1024 * 1024), 1
        )

    if include_crewai:
        report.update(_run_crewai_tool_acceptance(llm))

    return report


def _run_crewai_tool_acceptance(llm: Any) -> dict[str, Any]:
    """Require CrewAI to execute a real local Python tool through the LLM loop."""
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool

    invocations: list[str] = []

    def runtime_acceptance_probe() -> str:
        """Return a fixed marker proving the registered tool actually ran."""
        invocations.append("called")
        return "RUNTIME_ACCEPTANCE_TOOL_OK"

    probe_tool = tool("runtime_acceptance_probe")(runtime_acceptance_probe)
    agent = Agent(
        role="Runtime Acceptance Agent",
        goal="Call runtime_acceptance_probe exactly once and report its returned marker.",
        backstory=(
            "You are validating a local agent runtime. You must use the supplied "
            "tool; never claim it ran unless you have received its exact result."
        ),
        llm=llm,
        tools=[probe_tool],
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )
    task = Task(
        description=(
            "Invoke runtime_acceptance_probe exactly once. After receiving the "
            "tool result, report the exact marker returned by the tool."
        ),
        expected_output="The exact marker RUNTIME_ACCEPTANCE_TOOL_OK.",
        agent=agent,
    )

    started = time.perf_counter()
    crew_result = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    ).kickoff()
    elapsed = time.perf_counter() - started

    if len(invocations) != 1:
        raise RuntimeError(
            "CrewAI tool acceptance failed: expected exactly one invocation, "
            f"observed {len(invocations)}."
        )
    result_text = str(crew_result).strip()
    if "RUNTIME_ACCEPTANCE_TOOL_OK" not in result_text:
        raise RuntimeError(
            "CrewAI invoked the tool, but its final result did not preserve the "
            "tool's acceptance marker."
        )

    return {
        "crewai_tool_invocations": len(invocations),
        "crewai_seconds": round(elapsed, 3),
        "crewai_result": result_text[:500],
    }
