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


def test_colab_notebook_uses_colab_dependency_profile_and_smoke_checks_imports():
    notebook_path = ROOT / "notebooks" / "Colab_T4_Gradio_Studio.ipynb"
    sources = _sources(notebook_path)
    all_source = "\n".join(sources)

    assert "requirements-colab.txt" in all_source
    assert "pip check" in all_source
    assert "from PIL import" in all_source
    assert "from crewai import" in all_source
    assert "import gradio" in all_source
    assert "torch.cuda.is_available()" in all_source
    assert "HF_DEVICE" in all_source
    assert "pip install torch" not in all_source.casefold()


def test_colab_dependency_profile_does_not_override_torch():
    colab_requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    core_requirements = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    full_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "-r requirements-core.txt" in colab_requirements
    assert "torch" not in colab_requirements.casefold()
    assert "torch" not in core_requirements.casefold()
    assert "-r requirements-core.txt" in full_requirements
    assert "torch>=" in full_requirements
