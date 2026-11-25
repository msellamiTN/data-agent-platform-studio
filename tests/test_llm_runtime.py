import pytest

from studio.runtime import resolve_runtime_device


def test_auto_prefers_cuda_when_available():
    assert resolve_runtime_device("auto", cuda_available=True) == "cuda"


def test_auto_falls_back_to_cpu_without_cuda():
    assert resolve_runtime_device("auto", cuda_available=False) == "cpu"


def test_explicit_cpu_is_respected_even_when_cuda_exists():
    assert resolve_runtime_device("cpu", cuda_available=True) == "cpu"


def test_explicit_cuda_requires_cuda():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_runtime_device("cuda", cuda_available=False)


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="HF_DEVICE"):
        resolve_runtime_device("mps", cuda_available=True)
