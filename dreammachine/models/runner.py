"""Batched inference with Hugging Face transformers (optionally 4-bit, optionally with a LoRA adapter).

On an 8 GB laptop GPU a 1-2B model runs in bf16 without quantisation. Use
`load_in_4bit=True` only for 3B models. vLLM is faster, but it does not run on
native Windows (use WSL2), so it is not required here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .prompts import DEFAULT_PROMPT, format_prompt


def check_4bit_support(setting: str = "load_in_4bit") -> None:
    """Fail early with a clear message if 4-bit (bitsandbytes NF4) cannot run here.
    There is no silent fallback to bf16: precision is part of the experiment (PLAN.md 9.4)."""
    import torch

    hint = (f"Set `{setting}: false` instead: models up to ~1.7B fit in 8 GB in bf16 "
            "(plain LoRA for training).")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{setting} needs a CUDA GPU, but torch.cuda.is_available() is False. {hint}")
    try:
        import bitsandbytes.functional as bnb_f

        bnb_f.quantize_4bit(torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16), quant_type="nf4")
    except Exception as e:  # ImportError, or a bitsandbytes build without CUDA kernels
        raise RuntimeError(f"{setting} is set, but bitsandbytes cannot run 4-bit on this GPU "
                           f"({type(e).__name__}: {e}). {hint}") from e


class Runner(Protocol):
    def generate(self, questions: list[str]) -> list[list[str]]:
        """For each question, return n_samples completions."""


@dataclass
class GenConfig:
    max_new_tokens: int = 384
    n_samples: int = 1
    temperature: float = 0.0   # 0 = greedy
    top_p: float = 0.95
    batch_size: int = 16       # generated sequences per forward batch (questions x n_samples)
    seed: int = 0
    prompt_version: str = DEFAULT_PROMPT   # see models/prompts.py::PROMPTS


class HFRunner:
    def __init__(self, model: str, adapter: str | None = None, load_in_4bit: bool = False,
                 gen: GenConfig | None = None, device_map: str | None = "auto", dtype: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.gen = gen or GenConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model)
        self.tokenizer.padding_side = "left"  # decoder-only batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {}
        if load_in_4bit:
            check_4bit_support("eval.load_in_4bit")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        if dtype == "auto":
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available() and device_map:
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            if not load_in_4bit:
                # Fold the LoRA weights into the bf16 model: 1.84x faster generation, same accuracy (measured
                # 2026-10-07, Qwen3-0.6B, 96 GSM8K dev items: 240 s vs 441 s, 0.531 vs 0.521). Not done in 4-bit,
                # where merging into quantised weights is lossy.
                self.model = self.model.merge_and_unload()
        self.model.eval()
        self._torch = torch

    @classmethod
    def from_objects(cls, model, tokenizer, gen: GenConfig | None = None) -> "HFRunner":
        """Wrap an already-loaded model and tokenizer (tests and in-process pipelines)."""
        import torch

        self = cls.__new__(cls)
        self.model, self.tokenizer, self.gen, self._torch = model.eval(), tokenizer, gen or GenConfig(), torch
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self

    def generate(self, questions: list[str]) -> list[list[str]]:
        torch = self._torch
        g = self.gen
        torch.manual_seed(g.seed)
        prompts = [format_prompt(q, self.tokenizer, g.prompt_version) for q in questions]
        out: list[list[str]] = []
        self.last_truncated = []  # per question, per sample: hit max_new_tokens without a stop token
        stop_ids = self._stop_ids()
        # batch_size counts generated *sequences* (questions x n_samples), so memory stays bounded
        # when sampling several answers per question (diagnose uses n_samples 2-3).
        per_batch = max(1, g.batch_size // g.n_samples)
        n_batches = math.ceil(len(prompts) / per_batch)
        for b in range(n_batches):
            batch = prompts[b * per_batch:(b + 1) * per_batch]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            sample = g.temperature > 0
            with torch.no_grad():
                ids = self.model.generate(
                    **enc, max_new_tokens=g.max_new_tokens, do_sample=sample,
                    temperature=g.temperature if sample else None, top_p=g.top_p if sample else None,
                    num_return_sequences=g.n_samples, pad_token_id=self.tokenizer.pad_token_id)
            new = ids[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(new, skip_special_tokens=True)
            full = new.shape[1] >= g.max_new_tokens
            cut = [bool(full and int(row[-1]) not in stop_ids) for row in new]
            for i in range(len(batch)):
                out.append(texts[i * g.n_samples:(i + 1) * g.n_samples])
                self.last_truncated.append(cut[i * g.n_samples:(i + 1) * g.n_samples])
        return out

    def _stop_ids(self) -> set[int]:
        """Token ids that end a generation (EOS ids from the generation config and tokenizer, plus padding)."""
        ids: set[int] = set()
        for v in (getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
                  self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
            if v is not None:
                ids.update(v if isinstance(v, (list, tuple)) else [v])
        return ids


class EchoRunner:
    """Deterministic fake model for tests: answers from a lookup, else a fixed wrong answer."""

    def __init__(self, answers: dict[str, str] | None = None, default: str = "#### 0", n_samples: int = 1):
        self.answers, self.default, self.n_samples = answers or {}, default, n_samples

    def generate(self, questions: list[str]) -> list[list[str]]:
        return [[self.answers.get(q, self.default)] * self.n_samples for q in questions]
