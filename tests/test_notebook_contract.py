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


def test_colab_notebook_uses_kernel_interpreter_for_bootstrap_and_launch():
    notebook = _notebook(ROOT / "notebooks" / "Colab_T4_Gradio_Studio.ipynb")
    all_source = "\n".join(_sources(notebook))

    assert "subprocess.check_call([sys.executable, \"scripts/bootstrap_colab.py\"])" in all_source
    assert "Path(\".venv/bin/python\").resolve()" in all_source
    assert "subprocess.check_call([" in all_source
    assert "scripts/launch_colab.py" in all_source
    assert "python -m venv" not in all_source
    assert "runtime.restart_runtime()" not in all_source
    assert "transformers.utils import download_url" not in all_source
    assert "pip install torch" not in all_source.casefold()


def test_colab_requirements_keep_torch_out_of_the_studio_dependency_set():
    colab_requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    core_requirements = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    full_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    hf_compat = (ROOT / "requirements-hf-compat.txt").read_text(encoding="utf-8")

    assert colab_requirements.strip() == "-r requirements-core.txt"
    assert "torch" not in colab_requirements.casefold()
    assert "torch" not in core_requirements.casefold()
    assert "-r requirements-core.txt" in full_requirements
    assert "torch>=2.6.0,<3.0" in full_requirements
    assert "-r requirements-hf-compat.txt" in core_requirements
    assert "transformers==5.24.1" in hf_compat
    assert "huggingface-hub==1.32.0" in hf_compat


def test_colab_launcher_reuses_acceptance_model_for_gradio_app():
    source = (ROOT / "scripts" / "launch_colab.py").read_text(encoding="utf-8")
    assert "llm = get_local_llm()" in source
    assert "run_runtime_acceptance(llm, include_crewai=True)" in source
    assert source.index("run_runtime_acceptance(llm") < source.index("from app import demo")
    assert "demo.queue(default_concurrency_limit=1).launch(" in source
    assert "share=True" in source
    assert "The isolated Colab Python cannot import Torch" in source


def test_colab_bootstrap_avoids_ensurepip_and_verifies_base_torch_reuse():
    source = (ROOT / "scripts" / "bootstrap_colab.py").read_text(encoding="utf-8")
    assert '"venv"' in source
    assert '"--python"' in source
    assert "include-system-site-packages = true" in source
    assert '"-m", "venv"' not in source
    assert "venv_sees_torch(torch_file)" in source
    assert "Path(torch.__file__).resolve()" in source
    assert '"DATA2AI_COLAB_TORCH_FILE"' in source
    assert '"requirements-colab.txt"' in source
    assert "pip check" not in source
    assert "pip install torch" not in source.casefold()
