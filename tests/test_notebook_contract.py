import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sources(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_colab_notebook_uses_compatible_hf_stack_and_preflights_imports():
    notebook_path = ROOT / "notebooks" / "Colab_T4_Gradio_Studio.ipynb"
    sources = _sources(notebook_path)
    all_source = "\n".join(sources)

    assert "git pull --ff-only origin feature/colab-crewai-dual-agent-studio" in all_source
    assert "requirements-colab.txt" in all_source
    assert "pip check" in all_source
    assert "from PIL import" in all_source
    assert "from crewai import" in all_source
    assert "import gradio" in all_source
    assert "torch.cuda.is_available()" in all_source
    assert "HF_DEVICE" in all_source
    assert "run_runtime_acceptance(get_local_llm())" in all_source
    assert "from transformers import AutoModelForCausalLM, AutoTokenizer" in all_source
    assert "transformers.utils import download_url" not in all_source
    assert "runtime.restart_runtime()" not in all_source
    assert "Runtime > Restart session" in all_source
    assert "pip install torch" not in all_source.casefold()
    assert "Version('2.6.0')" in all_source


def test_colab_dependency_profile_preserves_torch_and_repairs_observed_conflicts():
    colab_requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    core_requirements = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    full_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    hf_compat = (ROOT / "requirements-hf-compat.txt").read_text(encoding="utf-8")
    colab_runtime = (ROOT / "requirements-colab-runtime.txt").read_text(encoding="utf-8")

    assert "-r requirements-core.txt" in colab_requirements
    assert "-r requirements-colab-runtime.txt" in colab_requirements
    assert "torch" not in colab_requirements.casefold()
    assert "torch" not in core_requirements.casefold()
    assert "-r requirements-core.txt" in full_requirements
    assert "torch>=2.6.0,<3.0" in full_requirements
    assert "-r requirements-hf-compat.txt" in core_requirements
    assert "transformers==5.24.1" in hf_compat
    assert "huggingface-hub==1.32.0" in hf_compat
    for requirement in (
        "numpy<2.5",
        "jedi>=0.16",
        "opentelemetry-api==1.42.1",
        "opentelemetry-sdk==1.42.1",
        "rich>=12.4.4,<14",
        "fsspec==2025.12.0",
        "oauthlib>=3.2.0,<4",
    ):
        assert requirement in colab_runtime
