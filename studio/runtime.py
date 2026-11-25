"""Runtime and environment-setting helpers independent from ML imports."""

import math


def resolve_runtime_device(requested: str | None, cuda_available: bool) -> str:
    """Resolve auto/cpu/cuda and reject impossible explicit CUDA requests."""
    choice = (requested or "auto").strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("HF_DEVICE must be one of: auto, cpu, cuda.")
    if choice == "cuda" and not cuda_available:
        raise RuntimeError("HF_DEVICE=cuda was requested, but CUDA is not available.")
    if choice == "auto":
        return "cuda" if cuda_available else "cpu"
    return choice


def resolve_cpu_threads(requested: str | None, available: int | None = None) -> int:
    """Return a positive CPU thread count; reject malformed explicit settings."""
    default = max(1, available or 1)
    if requested is None or not requested.strip():
        return default
    try:
        threads = int(requested)
    except ValueError as exc:
        raise ValueError("HF_CPU_THREADS must be a positive integer.") from exc
    if threads < 1:
        raise ValueError("HF_CPU_THREADS must be a positive integer.")
    return threads


def positive_int_setting(value: str | None, name: str, default: int) -> int:
    """Parse a strictly positive integer environment setting."""
    try:
        parsed = default if value is None or not value.strip() else int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def positive_float_setting(value: str | None, name: str, default: float) -> float:
    """Parse a finite, strictly positive floating-point environment setting."""
    try:
        parsed = default if value is None or not value.strip() else float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return parsed



def validate_prompt_token_count(token_count: int, maximum: int) -> None:
    """Reject an oversized model prompt rather than silently truncating policy."""
    if maximum < 1:
        raise ValueError("MAX_INPUT_TOKENS must be a positive integer.")
    if token_count < 0:
        raise ValueError("Prompt token count cannot be negative.")
    if token_count > maximum:
        raise ValueError(
            f"Agent prompt contains {token_count:,} tokens; the configured limit is "
            f"{maximum:,}. Reduce the column count or business context and retry."
        )
