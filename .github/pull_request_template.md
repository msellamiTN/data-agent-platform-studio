## Summary
<!-- What changed and why? -->

## Scope
- [ ] Core correctness / data contract
- [ ] Agent or model behaviour
- [ ] SQL/tool security
- [ ] UI / accessibility / responsive design
- [ ] Dependencies / runtime / notebook
- [ ] Documentation

## Validation evidence
- [ ] `python -m compileall -q app.py studio tests`
- [ ] `PYTHONPATH=. python -m pytest -q`
- [ ] GitHub Actions run linked below
- [ ] CPU runtime tested (include runtime, Python and device)
- [ ] CUDA runtime tested (include GPU and precision)
- [ ] CrewAI/model tool execution tested (include model revision)
- [ ] Browser/Colab UX tested (include browser and runtime)

**CI run:** 

**Tested commit SHA:**

## Security and compatibility
- [ ] No new unbounded tools or filesystem access
- [ ] SQL remains read-only, table-scoped and bounded
- [ ] Uploaded values remain treated as untrusted data
- [ ] Dependency compatibility and security impact reviewed
- [ ] No credentials, customer data or model artifacts committed

## Limitations / follow-up
<!-- State untested paths and blockers. Do not mark unexecuted checks as passed. -->
