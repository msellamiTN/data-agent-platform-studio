import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_acceptance_cli_help_does_not_import_heavy_runtime():
    result = subprocess.run(
        [sys.executable, "scripts/validate_runtime.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "CrewAI tool-call check" in result.stdout
    assert "--skip-crewai" in result.stdout


def test_runtime_acceptance_checks_actual_model_and_crewai_tool_execution():
    source = (ROOT / "studio" / "acceptance.py").read_text(encoding="utf-8")
    assert "llm.call(prompt.strip())" in source
    assert "parameter.device" in source
    assert "parameter.dtype" in source
    assert "len(invocations) != 1" in source
    assert "RUNTIME_ACCEPTANCE_TOOL_OK" in source
