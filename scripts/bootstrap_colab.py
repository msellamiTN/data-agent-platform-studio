#!/usr/bin/env python
"""Bootstrap isolated Colab dependencies without ensurepip or Torch replacement."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VPY = VENV / "bin" / "python"


def run(command, env=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def find_uv():
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "uv"])
    candidates = [
        shutil.which("uv"),
        str(Path(sysconfig.get_path("scripts")) / "uv"),
        str(Path(sys.prefix) / "bin" / "uv"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("uv was installed but its executable could not be located.")


def venv_sees_torch(torch_file):
    if not VPY.is_file():
        return False
    result = subprocess.run(
        [str(VPY), "-c", "import torch; print(torch.__file__)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return (
        result.returncode == 0
        and Path(result.stdout.strip()).resolve() == torch_file
    )


def ensure_base_site_packages_visible(torch_file):
    """Expose the notebook kernel's package directories to this venv.

    Most Colab images expose them through pyvenv.cfg. The explicit .pth fallback
    also handles kernels whose Torch package is loaded from an additional
    site-packages/dist-packages directory.
    """
    probe = subprocess.run(
        [
            str(VPY),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    venv_site = Path(probe.stdout.strip())
    venv_site.mkdir(parents=True, exist_ok=True)

    base_sites = {
        str(Path(entry).resolve())
        for entry in sys.path
        if entry
        and ("site-packages" in entry or "dist-packages" in entry)
        and Path(entry).is_dir()
    }
    # Torch is the anchor package; always include the directory containing it.
    base_sites.add(str(torch_file.parent.parent.resolve()))
    pth = venv_site / "data2ai-colab-base-site-packages.pth"
    pth.write_text("\\n".join(sorted(base_sites)) + "\\n", encoding="utf-8")


def create_venv(uv, torch_file):
    cfg = VENV / "pyvenv.cfg"
    configured = (
        cfg.is_file()
        and "include-system-site-packages = true" in cfg.read_text().lower()
    )
    if configured and venv_sees_torch(torch_file):
        return
    if VENV.exists():
        shutil.rmtree(VENV)

    # uv creates the venv without invoking Python's ensurepip module.
    run([uv, "venv", "--python", sys.executable, str(VENV)])

    if not cfg.is_file():
        raise RuntimeError("uv did not create pyvenv.cfg.")
    text = cfg.read_text(encoding="utf-8")
    if "include-system-site-packages = false" in text:
        text = text.replace(
            "include-system-site-packages = false",
            "include-system-site-packages = true",
        )
    elif "include-system-site-packages" not in text:
        text += "\ninclude-system-site-packages = true\n"
    cfg.write_text(text, encoding="utf-8")

    if not venv_sees_torch(torch_file):
        ensure_base_site_packages_visible(torch_file)

    if not venv_sees_torch(torch_file):
        probe = subprocess.run(
            [str(VPY), "-c", "import sys; print(sys.executable); print(sys.path)"],
            cwd=ROOT, capture_output=True, text=True,
        )
        raise RuntimeError(
            "The venv cannot import the same PyTorch installation as the active "
            "Colab kernel. Refusing to install a different Torch/CUDA wheel. "
            f"Kernel Python: {sys.executable}; kernel Torch: {torch_file}; "
            f"venv diagnostic: {probe.stdout}\n{probe.stderr}"
        )


def verify(torch_file, base_torch_version):
    code = """
import json, os, platform, sys
from pathlib import Path
from importlib.metadata import version
import torch, transformers, huggingface_hub, gradio, crewai, accelerate, pandas, plotly, PIL
from PIL import Image, ImageColor
from transformers import AutoModelForCausalLM, AutoTokenizer
from packaging.version import Version

expected = {
    "transformers": "5.24.1",
    "huggingface-hub": "1.32.0",
    "Pillow": "11.3.0",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise RuntimeError(f"Expected {package}=={wanted}, found {actual}")
if Version(torch.__version__.split("+")[0]) < Version("2.6.0"):
    raise RuntimeError(f"Colab PyTorch >= 2.6 is required, found {torch.__version__}")
expected_torch = Path(os.environ["DATA2AI_COLAB_TORCH_FILE"]).resolve()
actual_torch = Path(torch.__file__).resolve()
if actual_torch != expected_torch:
    raise RuntimeError(f"Expected Colab Torch {expected_torch}, imported {actual_torch}")
print(json.dumps({
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_file": str(actual_torch),
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "gradio": gradio.__version__,
    "crewai": version("crewai"),
    "accelerate": accelerate.__version__,
    "pandas": pandas.__version__,
    "plotly": plotly.__version__,
    "pillow": PIL.__version__,
    "model_api_imports": "passed",
}, indent=2))
"""
    env = os.environ.copy()
    env["DATA2AI_COLAB_TORCH_FILE"] = str(torch_file)
    run([str(VPY), "-c", code], env=env)
    return {
        "venv_python": str(VPY),
        "base_python": sys.executable,
        "base_torch_file": str(torch_file),
        "base_torch_version": base_torch_version,
    }


def main():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"Active Colab kernel {sys.executable} cannot import torch: "
            f"{type(exc).__name__}: {exc}. Check 'import torch' in a notebook cell."
        ) from exc

    torch_file = Path(torch.__file__).resolve()
    uv = find_uv()
    create_venv(uv, torch_file)
    run([
        uv, "pip", "install", "--python", str(VPY),
        "-r", str(ROOT / "requirements-colab.txt"),
    ])
    report = verify(torch_file, torch.__version__)
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    os.environ.setdefault("HF_DEVICE", "auto" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("HF_CPU_THREADS", "4")
    print("DATA2AI COLAB BOOTSTRAP PASSED", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
