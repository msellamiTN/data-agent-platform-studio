from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from pydantic import PrivateAttr
from crewai import BaseLLM

from .config import MAX_NEW_TOKENS, MODEL_ID


class HuggingFaceLocalLLM(BaseLLM):
    """Small local Transformers model exposed through CrewAI's BaseLLM API.

    This adapter intentionally disables native function calling. CrewAI uses
    its text-based tool loop; application tools remain deterministic Python.
    """

    _tokenizer: Any = PrivateAttr(default=None)
    _hf_model: Any = PrivateAttr(default=None)
    _max_new_tokens: int = PrivateAttr(default=MAX_NEW_TOKENS)

    def __init__(
        self,
        model_id: str = MODEL_ID,
        temperature: float = 0.0,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        super().__init__(model=model_id, temperature=temperature)
        self._max_new_tokens = max_new_tokens

    def _ensure_loaded(self) -> None:
        if self._hf_model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            use_fast=True,
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @staticmethod
    def _normalise_messages(messages: Any) -> list[dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        normalised: list[dict[str, str]] = []
        for item in messages or []:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = item.get("content", "")
            else:
                role = str(getattr(item, "role", "user"))
                content = getattr(item, "content", str(item))

            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
                content = "\n".join(p for p in parts if p)

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
                chat,
                tokenize=False,
                add_generation_prompt=True,
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
            generated[0][prompt_length:],
            skip_special_tokens=True,
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
    """Create one lazy model adapter; weights load on the first agent call."""
    return HuggingFaceLocalLLM()
