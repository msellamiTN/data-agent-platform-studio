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

The notebook creates `.venv` with `--system-site-packages`:

- PyTorch remains the Colab-provided CUDA-enabled build and is visible from the venv.
- Studio dependencies, including Transformers 5.24.1, Hub 1.32.0, CrewAI and Gradio, are installed in the venv.
- The notebook does not downgrade or replace unrelated base packages.
- A global `pip check` is intentionally not used: it reports conflicts among unrelated packages preinstalled in the Colab base image, not just the Studio dependency graph.

The notebook runs explicit import/version checks inside the venv. It then executes `scripts/launch_colab.py`, which runs direct model inference and one real CrewAI tool invocation before importing the Gradio application. Both steps occur in the same Python process, so the app reuses the already-loaded model adapter.

## Recovery after the earlier notebook cell

The previous notebook attempted to call `google.colab.runtime.restart_runtime()`, which does not exist. The new notebook does not call this method or modify packages in the active notebook kernel.

Open the updated notebook and run all cells from the top. It installs into `.venv` and launches the app in that venv. The old kernel's imported Transformers modules do not affect the isolated launch process.

## Acceptance boundary

The environment preflight proves that the required packages and public model APIs import in the isolated runtime, and that Colab's Torch installation is visible. The live launch step separately verifies actual model generation and a real CrewAI tool invocation. GPU/T4 and browser acceptance still require execution in a real Colab session.
