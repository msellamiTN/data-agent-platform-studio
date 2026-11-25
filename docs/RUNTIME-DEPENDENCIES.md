# Hugging Face runtime compatibility

## Supported compatibility set

The Studio targets:

- `transformers==5.24.1`
- `huggingface-hub==1.32.0`
- PyTorch `>=2.6.0`

Transformers 5.24.1 and current Diffusers both require the Hub 1.x line; Diffusers 0.42.1 specifies `huggingface-hub>=1.32.0,<3.0`. The model loader uses the Transformers 5 `dtype=` argument.

## Why `download_url` may fail

The old notebook preflight incorrectly imported the removed legacy helper `download_url` from `transformers.utils`. The Studio itself does not use it. The preflight now tests the public `AutoTokenizer` and `AutoModelForCausalLM` APIs. A separate third-party library may still request that legacy helper; EasyDeL is one example. Do not add EasyDeL to this Studio runtime unless a feature requires it and its compatibility is tested.

## Why the base Colab dependency warnings are not repaired globally

The reported Colab image contains unrelated preinstalled packages with mutually incompatible constraints. Examples from the supplied log include Google ADK requiring older OpenTelemetry while CrewAI requires OpenTelemetry 1.45.2 or newer, and Tweepy requiring OAuthlib below 4 while CrewAI requires OAuthlib 4 or newer. Downgrading those shared libraries to silence one package's warning can break the agent runtime.

The previous `requirements-colab-runtime.txt` attempted to pin shared packages to satisfy Google ADK, BigFrames, GCSFS, Datasets, Numba and Tweepy. That was the wrong approach for a Studio that uses current CrewAI. The file has been removed. The Colab profile now includes only the Studio's own dependencies.

## Colab isolation model

The notebook delegates setup to `scripts/bootstrap_colab.py`:

- It runs the bootstrap with the same `sys.executable` used by the active notebook kernel, not a shell-resolved `python` executable.
- It uses `uv venv --python <kernel-interpreter>` instead of `python -m venv`; the latter invokes `ensurepip`, which is missing from some Colab Python images and caused the reported `CalledProcessError`.
- It sets `include-system-site-packages = true` in the newly created `pyvenv.cfg` and verifies that `.venv/bin/python` imports the exact same `torch.__file__` as the base Colab kernel.
- If the same PyTorch build is not visible, bootstrap stops with a diagnostic before installing dependencies or launching the app. It will not silently install a different Torch/CUDA wheel.
- Studio dependencies, including Transformers 5.24.1, Hub 1.32.0, CrewAI and Gradio, are installed in `.venv` with `uv`; unrelated base packages are not downgraded.
- A global `pip check` is intentionally not used because it reports conflicts among unrelated packages preinstalled in the Colab base image, rather than checking only the Studio dependency graph.

The notebook then executes `scripts/launch_colab.py` with the explicit `.venv/bin/python` path. The launcher performs direct model inference and one real CrewAI tool invocation before importing the Gradio application. Both steps occur in the same Python process, so the app reuses the already-loaded model adapter.

## Recovery after the earlier notebook cell

The previous notebook attempted to call `google.colab.runtime.restart_runtime()`, which does not exist. The new notebook does not call this method. It also does not run `python -m venv`; use the repository's `uv` bootstrap instead.

Open the updated notebook and run all cells from the top. Cell 1 fetches the latest branch. Cell 2 must show `DATA2AI COLAB BOOTSTRAP PASSED` before continuing. The bootstrap must report a venv Python path and the exact base Torch file. Cell 6 launches using the explicit venv interpreter. The old kernel's imported Transformers modules do not affect the isolated launch process.

## Acceptance boundary

The environment preflight proves that the required packages and public model APIs import in the isolated runtime, and that Colab's Torch installation is visible. The live launch step separately verifies actual model generation and a real CrewAI tool invocation. GPU/T4 and browser acceptance still require execution in a real Colab session.
