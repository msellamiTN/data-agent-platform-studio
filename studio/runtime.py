"""Runtime selection helpers kept independent from heavyweight ML imports."""


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
