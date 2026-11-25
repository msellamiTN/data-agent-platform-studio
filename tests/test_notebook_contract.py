import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return notebook


def _sources(notebook):
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_colab_notebook_uses_isolated_venv_and_live_launcher():
    notebook = _notebook(ROOT / "notebooks" / "Colab_T4_Gradio_Studio.ipynb")
    all_source = "\n".join(_sources(notebook))

    assert "git pull --ff-only origin feature/colab-crewai-dual-agent-studio" in all_source
    assert "requirements-colab.txt" in all_source
    assert "scripts/bootstrap_colab.py" in all_source
    assert ".venv/bin/python scripts/launch_colab.py" in all_source
    assert "run_runtime_acceptance(get_local_llm())" not in all_source
    assert "from transformers import AutoModelForCausalLM, AutoTokenizer" in all_source
    assert "transformers.utils import download_url" not in all_source
    assert "runtime.restart_runtime()" not in all_source
    assert "pip check" not in all_source
    assert "pip install torch" not in all_source.casefold()


def test_colab_requirements_do_not_override_global_runtime_packages():
    colab_requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    core_requirements = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    full_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    hf_compat = (ROOT / "requirements-hf-compat.txt").read_text(encoding="utf-8")

    assert colab_requirements.strip() == "-r requirements-core.txt"
    assert "requirements-colab-runtime.txt" not in colab_requirements
    assert "torch" not in colab_requirements.casefold()
    assert "torch" not in core_requirements.casefold()
    assert "-r requirements-core.txt" in full_requirements
    assert "torch>=2.6.0,<3.0" in full_requirements
    assert "-r requirements-hf-compat.txt" in core_requirements
    assert "transformers==5.24.1" in hf_compat
    assert "huggingface-hub==1.32.0" in hf_compat
    assert not (ROOT / "requirements-colab-runtime.txt").exists()


def test_colab_launcher_reuses_acceptance_model_for_gradio_app():
    source = (ROOT / "scripts" / "launch_colab.py").read_text(encoding="utf-8")
    assert "llm = get_local_llm()" in source
    assert "run_runtime_acceptance(llm, include_crewai=True)" in source
    assert source.index("run_runtime_acceptance(llm") < source.index("from app import demo")
    assert "demo.queue(default_concurrency_limit=1).launch(" in source
    assert "share=True" in source


def test_colab_bootstrap_isolates_studio_and_reuses_base_torch():
    source = (ROOT / "scripts" / "bootstrap_colab.py").read_text(encoding="utf-8")
    assert '"--system-site-packages"' in source
    assert '"--no-deps"' in source
    assert '"--python"' in source
    assert '"requirements-colab.txt"' in source
    assert 'Path(torch.__file__).resolve()' in source
    assert 'Path(__import__("os").environ["DATA2AI_COLAB_TORCH_FILE"]).resolve()' in source
    assert '"model_api_imports": "passed"' in source
    assert "pip check" not in source
    assert "requirements-colab-runtime.txt" not in source
