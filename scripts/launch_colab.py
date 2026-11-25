#!/usr/bin/env python
"""Run Colab acceptance and launch Gradio in one isolated Studio process."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    from studio.acceptance import run_runtime_acceptance
    from studio.llm import get_local_llm

    # Load and validate the shared adapter once. The app imports the same
    # module/cache below, so both CrewAI agents reuse the validated model.
    llm = get_local_llm()
    report = run_runtime_acceptance(llm, include_crewai=True)
    print("LIVE RUNTIME ACCEPTANCE", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)

    from app import demo

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
