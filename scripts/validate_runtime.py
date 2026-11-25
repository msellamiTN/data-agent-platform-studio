#!/usr/bin/env python
"""Load the configured local model and validate actual inference + CrewAI tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a live local Hugging Face inference check and a CrewAI tool-call "
            "check. This downloads the model on first use if it is not cached."
        )
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model", help="Optional Hugging Face model ID override.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Generation budget for the short smoke-test prompt (default: 48).",
    )
    parser.add_argument(
        "--skip-crewai",
        action="store_true",
        help="Validate model loading/inference only; do not run a CrewAI tool.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly the word READY.",
        help="Short direct inference prompt.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_new_tokens < 1:
        print(json.dumps({"status": "failed", "error": "--max-new-tokens must be positive."}))
        return 2
    if not args.prompt.strip():
        print(json.dumps({"status": "failed", "error": "--prompt must not be empty."}))
        return 2

    if args.device:
        os.environ["HF_DEVICE"] = args.device
    if args.model:
        os.environ["HF_MODEL_ID"] = args.model
    os.environ["HF_MAX_NEW_TOKENS"] = str(args.max_new_tokens)

    try:
        from studio.acceptance import run_runtime_acceptance
        from studio.llm import get_local_llm

        report = run_runtime_acceptance(
            get_local_llm(),
            prompt=args.prompt,
            include_crewai=not args.skip_crewai,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
