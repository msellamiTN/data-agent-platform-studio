#!/usr/bin/env python
"""Bootstrap an isolated Colab environment while retaining Colab's CUDA PyTorch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=ROOT, env=env)


def _create_or_repair_venv(base_torch_file: Path) -> None:
    config = VENV_DIR / "pyvenv.cfg"
    valid = (
        VENV_PYTHON.is_file()
        and config.is_file()
        and "include-system-site-packages = true" in config.read_text().lower()
    )
    if valid:
        probe = subprocess.run(
            [
                str(VENV_PYTHON),
                "-c",
                "import torch; print(torch.__file__)",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        valid = (
            probe.returncode == 0
            and probe.stdout.strip()
            and Path(probe.stdout.strip()).resolve() == base_torch_file
        )
    if valid:
        return
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)
    _run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(VENV_DIR),
        ]
    )


def _uv_executable() -> str:
    # uv is installed without dependencies into the Colab base Python; this does
    # not cause pip to upgrade or downgrade the preinstalled Colab package set.
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "uv",
        ]
    )
    scripts_dir = Path(sysconfig_scripts_dir())
    candidate = scripts_dir / ("uv.exe" if os.name == "nt" else "uv")
    executable = shutil.which("uv") or (str(candidate) if candidate.is_file() else None)
    if not executable:
        raise RuntimeError("uv installation completed, but the executable was not found.")
    return executable


def sysconfig_scripts_dir() -> str:
    import sysconfig

    return sysconfig.get_path("scripts")


def bootstrap() -> dict[str, Any]:
    # Import PyTorch in the base kernel first: this is the CUDA-enabled Colab
    # distribution we intend to reuse from the system-site-packages venv.
    import torch

    base_torch_file = Path(torch.__file__).resolve()
    base_torch_version = torch.__version__
    _create_or_repair_venv(base_torch_file)

    uv = _uv_executable()
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(VENV_PYTHON),
            "-r",
            str(ROOT / "requirements-colab.txt"),
        ]
    )

    probe_code = r'''
import json
import platform
from pathlib import Path
from importlib.metadata import version
import torch
import transformers
import huggingface_hub
import gradio
import crewai
import accelerate
import pandas
import plotly
import PIL
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

base_torch_file = Path(__import__("os").environ["DATA2AI_COLAB_TORCH_FILE"]).resolve()
loaded_torch_file = Path(torch.__file__).resolve()
if loaded_torch_file != base_torch_file:
    raise RuntimeError(
        "The isolated environment did not reuse Colab's PyTorch build: "
        f"expected {base_torch_file}, loaded {loaded_torch_file}"
    )

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_file": str(loaded_torch_file),
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
'''
    child_env = os.environ.copy()
    child_env["DATA2AI_COLAB_TORCH_FILE"] = str(base_torch_file)
    _run([str(VENV_PYTHON), "-c", probe_code], env=child_env)

    os.environ.setdefault(
        "HF_DEVICE", "auto" if torch.cuda.is_available() else "cpu"
    )
    os.environ.setdefault("HF_CPU_THREADS", "4")
    return {
        "venv_python": str(VENV_PYTHON),
        "base_torch_file": str(base_torch_file),
        "base_torch_version": base_torch_version,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def main() -> int:
    report = bootstrap()
    print("DATA2AI COLAB BOOTSTRAP", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
