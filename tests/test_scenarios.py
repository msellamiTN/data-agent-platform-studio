import builtins
import sys
import types
from pathlib import Path

import pandas as pd

import studio.etl as etl_module
import studio.scenarios as scenarios
import studio.sql as sql_module


def _configure_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    pipeline_root = workspace / "pipeline"
    current_dir = pipeline_root / "current"
    manifest_path = current_dir / "manifest.json"
    monkeypatch.setattr(scenarios, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(etl_module, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(etl_module, "PIPELINE_ROOT", pipeline_root)
    monkeypatch.setattr(etl_module, "CURRENT_PIPELINE_DIR", current_dir)
    monkeypatch.setattr(etl_module, "PIPELINE_MANIFEST", manifest_path)
    monkeypatch.setattr(sql_module, "PIPELINE_ROOT", pipeline_root)
    monkeypatch.setattr(sql_module, "PIPELINE_MANIFEST", manifest_path)
    workspace.mkdir(parents=True)
    return workspace


def _publish_title_dataset(workspace):
    source = workspace / "BookAuteurs.csv"
    pd.DataFrame(
        {
            "titre": ["Book A", "Book B", "Book C"],
            "Prix": [19.9, 29.9, 34.9],
            "Nom": ["Smith", "Smith", "Jones"],
            "prenom": ["John", "John", "Emily"],
        }
    ).to_csv(source, index=False)
    return etl_module.run_csv_pipeline(source)


def _install_fake_crewai(monkeypatch, query_to_issue):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.tools = kwargs["tools"]

    class FakeCrew:
        def __init__(self, agents, tasks, process, verbose):
            self.agents = agents

        def kickoff(self):
            self.agents[0].tools[0]()
            self.agents[0].tools[1](query_to_issue)
            return "Model narrative intentionally does not define the verified answer."

    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    crewai = types.ModuleType("crewai")
    crewai.__path__ = []
    crewai.Agent = FakeAgent
    crewai.Crew = FakeCrew
    crewai.Process = types.SimpleNamespace(sequential="sequential")
    crewai.Task = FakeTask
    tools_module = types.ModuleType("crewai.tools")
    tools_module.tool = lambda name: (lambda function: function)
    crewai.tools = tools_module
    monkeypatch.setitem(sys.modules, "crewai", crewai)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools_module)
    monkeypatch.setattr(scenarios, "_make_agent_llm", lambda: object())


def test_engineering_and_count_analysis_survive_crewai_import_failure(tmp_path, monkeypatch):
    workspace = _configure_workspace(tmp_path, monkeypatch)
    source = workspace / "BookAuteurs.csv"
    pd.DataFrame(
        {
            "titre": ["Book A", "Book B", "Book C"],
            "Prix": [19.9, 29.9, 34.9],
            "Nom": ["Smith", "Smith", "Jones"],
            "prenom": ["John", "John", "Emily"],
        }
    ).to_csv(source, index=False)

    original_import = builtins.__import__

    def fail_crewai(name, *args, **kwargs):
        if name == "crewai" or name.startswith("crewai."):
            raise ImportError("simulated PIL._typing dependency failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_crewai)
    engineering = scenarios.run_data_engineering(str(source))
    assert engineering["execution_mode"] == "python_fallback"
    assert engineering["report"]["rows_after"] == 3
    assert engineering["report"]["status"] == "READY"
    assert "PIL._typing" in engineering["crew_error"]

    analysis = scenarios.run_data_analysis("how many titre?")
    assert analysis["execution_mode"] == "validated_query_plan"
    assert analysis["result_kind"] == "verified_aggregate"
    assert analysis["sql"] == 'SELECT COUNT("titre") AS "value_count" FROM silver_data'
    assert analysis["rows"] == [{"value_count": 3}]
    assert 'contains 3 non-null value(s)' in analysis["summary"]
    assert analysis["orchestration_status"] == "not_completed"


def test_wrong_model_preview_is_overridden_by_count_plan(tmp_path, monkeypatch):
    workspace = _configure_workspace(tmp_path, monkeypatch)
    _publish_title_dataset(workspace)
    _install_fake_crewai(monkeypatch, "SELECT * FROM silver_data LIMIT 25")

    analysis = scenarios.run_data_analysis("how many titre?")
    assert analysis["execution_mode"] == "validated_query_plan"
    assert analysis["result_kind"] == "verified_aggregate"
    assert analysis["sql"] == 'SELECT COUNT("titre") AS "value_count" FROM silver_data'
    assert analysis["rows"] == [{"value_count": 3}]
    assert analysis["summary"] == 'Column "titre" contains 3 non-null value(s).'
    assert analysis["orchestration_status"] == "completed"



def test_analysis_rejects_overlong_question_before_pipeline_lookup(monkeypatch):
    monkeypatch.setattr(scenarios, "MAX_ANALYSIS_QUESTION_CHARS", 8)
    try:
        scenarios.run_data_analysis("this question is too long")
    except ValueError as exc:
        assert "8-character limit" in str(exc)
    else:
        raise AssertionError("Expected an overlong question to be rejected")
