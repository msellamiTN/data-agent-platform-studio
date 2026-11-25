# Hugging Face runtime compatibility

## Supported compatibility set

The Colab and local profiles use this tested dependency target:

- `transformers==5.24.1`
- `huggingface-hub==1.32.0`

Transformers 5.24.1 requires Hugging Face Hub 1.31 or later and below 3.0. The installed Colab `diffusers==0.42.1` requires Hub 1.32.0 or later and below 3.0. Hub 1.32.0 satisfies both requirements. The Studio now uses the Transformers 5 `dtype=` model-loading argument and requires PyTorch 2.6 or later.

The previous pin `huggingface-hub==0.36.2` was incompatible with the current Colab Diffusers installation. Do not restore it for this runtime.

## Why `download_url` may fail

The error:

```text
ImportError: cannot import name 'download_url' from 'transformers.utils'
```

is associated with dependent code that still imports this legacy symbol. EasyDeL contains such an import. The Data Agent Platform Studio does not use EasyDeL and does not need the `download_url` helper. The Studio now uses the public `AutoTokenizer` and `AutoModelForCausalLM` interfaces instead; the notebook does not test or import the removed legacy helper.

Do not add EasyDeL to this environment unless a separate feature requires it and its compatibility with the selected Transformers release has been verified.

## Colab package conflicts

The Colab dependency profile explicitly reconciles the conflicts reported in the runtime log:

- `huggingface-hub==1.32.0`: satisfies Diffusers 0.42.1 and Transformers 5.24.1.
- `numpy<2.5`: satisfies Numba 0.64.0.
- `jedi>=0.16`: satisfies the installed IPython 7.34.0 requirement.
- `opentelemetry-api==1.42.1` and `opentelemetry-sdk==1.42.1`: satisfy Google ADK 2.7.1.
- `rich>=12.4.4,<14`: satisfies BigFrames 2.51.0.
- `fsspec==2025.12.0`: satisfies GCSFS 2025.12.0 and Datasets' upper bound.
- `oauthlib>=3.2.0,<4`: satisfies Tweepy 4.17.0.

These additional constraints live in `requirements-colab-runtime.txt`; they are not imposed on ordinary local installations.

## Clean Colab recovery

The notebook checks whether any package whose files it will replace has already been imported. If so, it stops before installing anything and tells you to use **Runtime → Restart session**. This avoids mixing package files on disk with stale modules already loaded in memory.

For the runtime shown in the error report:

1. Select **Runtime → Restart session** in Colab.
2. Reopen the notebook from the repository branch and run all cells from the top.
3. The notebook installs `requirements-colab.txt`, runs `pip check`, then imports Transformers' public tokenizer/model classes.
4. Confirm the dependency smoke-check prints Transformers 5.24.1 and Hub 1.32.0 before continuing to the live model/CrewAI acceptance cell.

The notebook does not call `google.colab.runtime.restart_runtime()`; that method is not available in the Colab runtime module. Kernel restart is an explicit user action so package installation is not silently interrupted.

## Acceptance boundary

A successful package/import preflight proves that the pinned libraries and public APIs can be imported. It does not prove model download, tokenizer/model loading, GPU allocation, generation, or CrewAI tool execution. Those are checked separately by the live runtime acceptance cell.
