import pytest

from studio.runtime import resolve_cpu_threads, resolve_runtime_device


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


def test_cpu_threads_default_to_available_count():
    assert resolve_cpu_threads(None, available=8) == 8


def test_cpu_threads_accept_positive_integer():
    assert resolve_cpu_threads("4", available=8) == 4


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_cpu_threads_reject_invalid_values(value):
    with pytest.raises(ValueError, match="HF_CPU_THREADS"):
        resolve_cpu_threads(value, available=8)
