from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from pydantic import PrivateAttr
from crewai import BaseLLM

from .config import MAX_NEW_TOKENS, MODEL_ID


def resolve_device(requested: str | None = None) -> str:
    """Resolve auto/cpu/cuda without silently ignoring an explicit CUDA request."""
    choice = (requested or os.getenv("HF_DEVICE", "auto")).strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("HF_DEVICE must be one of: auto, cpu, cuda.")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HF_DEVICE=cuda was requested, but CUDA is not available.")
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


class HuggingFaceLocalLLM(BaseLLM):
    """Shared local Transformers model for CrewAI, running on CPU or CUDA.

    CPU uses FP32 for broad compatibility. CUDA uses FP16. This adapter
    intentionally disables native function calling; CrewAI uses its text tool loop.
    """

    _tokenizer: Any = PrivateAttr(default=None)
    _hf_model: Any = PrivateAttr(default=None)
    _max_new_tokens: int = PrivateAttr(default=MAX_NEW_TOKENS)
    _device: str = PrivateAttr(default="cpu")

    def __init__(
        self,
        model_id: str = MODEL_ID,
        temperature: float = 0.0,
        max_new_tokens: int = MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> None:
        super().__init__(model=model_id, temperature=temperature)
        self._max_new_tokens = max_new_tokens
        self._device = resolve_device(device)

    @property
    def runtime_device(self) -> str:
        return self._device

    def _ensure_loaded(self) -> None:
        if self._hf_model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        if self._device == "cuda":
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            # CPU path deliberately avoids device_map="auto" and half precision.
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self._hf_model.to("cpu")
            # Small local models can otherwise oversubscribe hosted CPU runtimes.
            cpu_threads = int(os.getenv("HF_CPU_THREADS", str(max(1, os.cpu_count() or 1))))
            torch.set_num_threads(max(1, cpu_threads))

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @staticmethod
    def _normalise_messages(messages: Any) -> list[dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        normalised: list[dict[str, str]] = []
        for item in messages or []:
            if isinstance(item, dict):
                role, content = str(item.get("role", "user")), item.get("content", "")
            else:
                role = str(getattr(item, "role", "user"))
                content = getattr(item, "content", str(item))
            if isinstance(content, list):
                parts = [
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                ]
                content = "\n".join(part for part in parts if part)
            text = str(content).strip()
            if text:
                normalised.append({"role": role, "content": text})
        return normalised or [{"role": "user", "content": "Continue."}]

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del tools, callbacks, available_functions, from_task, from_agent, response_model, kwargs
        self._ensure_loaded()
        chat = self._normalise_messages(messages)
        try:
            prompt = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(
                f"{item['role'].upper()}: {item['content']}" for item in chat
            ) + "\nASSISTANT:"

        encoded = self._tokenizer(prompt, return_tensors="pt")
        model_device = next(self._hf_model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self._hf_model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        answer = self._tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        ).strip()
        if stop:
            for token in stop:
                if token and token in answer:
                    answer = answer.split(token, 1)[0].rstrip()
        return answer or "Je n'ai pas pu produire de réponse exploitable."

    def supports_function_calling(self) -> bool:
        return False


@lru_cache(maxsize=1)
def get_local_llm() -> HuggingFaceLocalLLM:
    """Create one lazy adapter; both agents share its loaded model weights."""
    return HuggingFaceLocalLLM()
