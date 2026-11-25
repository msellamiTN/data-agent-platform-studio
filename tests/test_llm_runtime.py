import pytest

from studio.runtime import (
    positive_float_setting,
    positive_int_setting,
    resolve_cpu_threads,
    resolve_runtime_device,
)


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



def test_positive_integer_setting_uses_default():
    assert positive_int_setting(None, "MAX_UPLOAD_MB", 25) == 25


def test_positive_integer_setting_accepts_valid_value():
    assert positive_int_setting("32", "MAX_UPLOAD_MB", 25) == 32


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_positive_integer_setting_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="MAX_UPLOAD_MB"):
        positive_int_setting(value, "MAX_UPLOAD_MB", 25)


def test_positive_float_setting_accepts_decimal_value():
    assert positive_float_setting("2.5", "MAX_SQL_EXECUTION_SECONDS", 5.0) == 2.5


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_positive_float_setting_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="MAX_SQL_EXECUTION_SECONDS"):
        positive_float_setting(value, "MAX_SQL_EXECUTION_SECONDS", 5.0)



def test_prompt_token_count_accepts_exact_limit():
    from studio.runtime import validate_prompt_token_count
    validate_prompt_token_count(8192, 8192)


def test_prompt_token_count_rejects_oversized_prompt():
    from studio.runtime import validate_prompt_token_count
    with pytest.raises(ValueError, match="configured limit is 8,192"):
        validate_prompt_token_count(8193, 8192)


def test_prompt_token_count_rejects_invalid_limit():
    from studio.runtime import validate_prompt_token_count
    with pytest.raises(ValueError, match="MAX_INPUT_TOKENS"):
        validate_prompt_token_count(1, 0)
