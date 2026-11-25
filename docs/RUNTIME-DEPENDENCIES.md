# Hugging Face runtime compatibility

## Supported compatibility set

The Colab and local profiles intentionally use:

- `transformers==4.57.6`
- `huggingface-hub==0.36.2`

Transformers 4.x declares a Hub 0.x dependency range. Transformers 5.x moves to the Hub 1.x line and removes some legacy utility exports. The application currently targets the Transformers 4.x API.

## Why `download_url` may fail

The error:

```text
ImportError: cannot import name 'download_url' from 'transformers.utils'
```

is commonly produced by a dependent package that still imports that legacy symbol. The EasyDeL bridge, for example, imports `download_url` from `transformers.utils`. The symbol is exported by Transformers 4.57.6, but must not be assumed to exist in Transformers 5.x.

The Data Agent Platform Studio does not use EasyDeL. Its local inference path is PyTorch + Transformers + CrewAI. Do not install or upgrade Transformers/Hugging Face Hub independently with unconstrained `-U` commands: that can override the project's selected compatibility set.

## Clean Colab recovery

Run this in a fresh Colab runtime, before importing Transformers, CrewAI, EasyDeL, or the Studio:

```python
import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-cache-dir", "--force-reinstall",
    "transformers==4.57.6",
    "huggingface-hub==0.36.2",
])
subprocess.check_call([sys.executable, "-m", "pip", "check"])
```

Then restart the runtime. Re-run the notebook from the beginning. Its dependency preflight explicitly imports `download_url` and both package versions, so an incompatible runtime fails before the Studio launches.

For this Studio, avoid importing EasyDeL unless another task explicitly requires it. If EasyDeL is required in a separate notebook, install it in a separate environment or validate its declared Transformers compatibility before combining the stacks.

## Acceptance boundary

A successful import preflight proves the package API is available. It does not prove model download, tokenizer/model loading, GPU allocation, generation, or CrewAI tool execution. Those are checked separately by the live runtime acceptance cell.
