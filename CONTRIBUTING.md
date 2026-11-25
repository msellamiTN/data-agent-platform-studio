# Contributing

## Development environment

Use Python 3.11 or a supported Google Colab runtime. Create an isolated environment and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run the deterministic test suite and source compilation before opening a pull request:

```bash
python -m compileall -q app.py studio tests
PYTHONPATH=. python -m pytest -q
```

The deterministic CI suite does not download the Hugging Face model. A change to the model adapter, dependency set, CrewAI integration or Colab notebook also requires a dependency smoke test and a real CPU/GPU runtime acceptance result. Do not report a model or Colab test as passed unless it was executed.

## Architecture invariants

- The model proposes and explains; bounded Python tools perform data operations and provide authoritative facts.
- Do not give agents shell, arbitrary Python, database write, or unbounded filesystem tools.
- Both agent workspaces must use the same active manifest and immutable run artifacts.
- Never replace the active manifest until the new pipeline artifacts are complete.
- Analysis must use the active run's Gold database and canonical `silver_data` table.
- All SQL paths must preserve the read-only URI, authorizer, query timeout and row cap.
- Never present a preview as the answer to an aggregate question.
- Use actual tool outputs for numeric summaries; do not trust model-generated numeric prose.
- Treat uploaded data, column values and business context as untrusted input.

## User interface

- Follow the tokens and component patterns in `app.py` and `docs/UX-DESIGN.md`.
- Preserve explicit loading, no-data, success, warning and failure states.
- Show the active dataset and pipeline ID in the Analysis workspace.
- Clear results tied to a previous pipeline when a new pipeline is published.
- Do not describe a deterministic query plan as CrewAI execution.
- Check responsive layouts and keyboard interaction when changing visible components.

## Dependency changes

- Review direct and transitive dependency constraints before upgrading.
- Run `pip check` in the target notebook environment.
- Verify the exact imports used by the application before importing Torch or starting Gradio.
- Record compatibility blockers and security trade-offs; do not silently downgrade a package to make installation pass.
- Update the Colab notebook and README when install steps or supported versions change.

## Pull request evidence

A pull request should state:
- the behaviour changed and the relevant issue or requirement;
- tests added or updated and their result;
- the exact commit SHA and CI run for deterministic validation;
- separate evidence for any CPU, CUDA, model-download, CrewAI, browser or Colab test;
- known limitations and whether the PR is still a prototype-only change.

## Data and secrets

Never commit user/customer datasets, generated workspaces, model weights, tokens, credentials or private logs. Use synthetic data in examples and tests. See [SECURITY.md](SECURITY.md).
