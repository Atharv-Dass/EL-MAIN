"""Prompt versions (PLAN.md §8.5, D10)."""

import hashlib
from pathlib import Path

import pytest

from dreammachine.experiments import pipeline as P
from dreammachine.models.prompts import (
    DEFAULT_PROMPT, INSTRUCTION, PROMPTS, QWEN_BOXED, V2_700, build_messages, format_completion, format_prompt,
    get_prompt,
)

V2_700_TEXT = ("Solve the math problem step by step. At the end, write a separate final line containing only the "
               "numeric answer, with no units, dollar sign, commas, bold formatting, or extra text. "
               "Example final line: 18.")
V2_700_SHA256 = "43dc849634b055bbe14d232793d876824882cfc5fc5f1ad5f88437c61689aa96"


def test_v2_700_is_byte_exact():
    assert PROMPTS["v2_700"] == V2_700 == V2_700_TEXT
    assert hashlib.sha256(PROMPTS["v2_700"].encode("utf-8")).hexdigest() == V2_700_SHA256
    assert len(PROMPTS["v2_700"]) == 204 and not PROMPTS["v2_700"].endswith((" ", "\n"))


def test_v2_700_matches_project_memory():
    memory = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text(encoding="utf-8")
    assert memory.split(">>>`", 1)[1].split("`<<<", 1)[0] == PROMPTS["v2_700"]


def test_default_is_qwen_boxed_and_dm_v1_unchanged():
    assert DEFAULT_PROMPT == "qwen_boxed"                       # chosen by the user, 2026-10-05 (D10)
    assert build_messages("Q?") == [{"role": "user", "content": f"Q?\n{QWEN_BOXED}"}]
    assert PROMPTS["dm_v1"] == INSTRUCTION
    assert build_messages("Q?", "dm_v1") == [{"role": "user", "content": f"{INSTRUCTION}\n\nProblem: Q?"}]
    assert format_prompt("Q?", prompt_version="dm_v1") == f"{INSTRUCTION}\n\nProblem: Q?\nSolution:\n"


def test_training_answer_format_follows_the_prompt():
    sol = "Tom has 3 + 4 = 7 apples.\n#### 7"
    assert format_completion(sol, "qwen_boxed") == "Tom has 3 + 4 = 7 apples.\n\\boxed{7}"
    assert format_completion(sol, "v2_700") == "Tom has 3 + 4 = 7 apples.\n7"
    assert format_completion(sol, "dm_v1") == sol
    assert format_completion("no final marker", "qwen_boxed") == "no final marker"
    from dreammachine.diagnosis.extract import extract_answer, lastline_format_ok
    assert extract_answer(format_completion(sol, "qwen_boxed")) == 7          # the classifier reads \boxed{}
    assert lastline_format_ok(format_completion(sol, "v2_700"))


def test_placements():
    assert build_messages("Q?", "v2_700") == [{"role": "system", "content": V2_700}, {"role": "user", "content": "Q?"}]
    assert build_messages("Q?", "qwen_boxed") == [{"role": "user", "content": f"Q?\n{QWEN_BOXED}"}]
    # without a system role, v2_700 falls back to the user turn exactly like dm_v1
    assert build_messages("Q?", "v2_700", system_role=False) == [
        {"role": "user", "content": f"{V2_700}\n\nProblem: Q?"}]


def test_unknown_prompt_version():
    with pytest.raises(KeyError, match="available"):
        get_prompt("v9")


class _NoSystemTokenizer:
    """Mimics a chat template that rejects the system role (e.g. Gemma 2)."""
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_system_fallback_when_template_rejects_it():
    out = format_prompt("Q?", _NoSystemTokenizer(), "v2_700")
    assert out == f"user:{V2_700}\n\nProblem: Q?"


def test_chat_template_renders_system(tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    a, b = format_prompt("Q?", tok, "dm_v1"), format_prompt("Q?", tok, "v2_700")
    assert a.startswith("<|user|>") and b.startswith(f"<|system|>{V2_700}<|end|><|user|>Q?")


def test_config_carries_one_prompt_version(tmp_path):
    cfg = P.Config(name="t", output_dir=str(tmp_path), prompt_version="v2_700")
    assert cfg.gen().prompt_version == "v2_700"
    assert P.Config(name="t", output_dir=str(tmp_path)).gen().prompt_version == "qwen_boxed"
    cfg.generation = {"prompt_version": "dm_v1"}
    with pytest.raises(ValueError, match="top level"):
        cfg.gen()


def test_training_uses_the_prompt_version(tiny_model_dir):
    from transformers import AutoTokenizer

    from dreammachine.data import TrainExample
    from dreammachine.train import tokenize_example

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ex = TrainExample(prompt="Asha has 5 apples.", completion="#### 5", source="t", id="1")
    a = tokenize_example(tok, ex, 512, "dm_v1")
    b = tokenize_example(tok, ex, 512, "v2_700")
    n_b = len(tok(format_prompt(ex.prompt, tok, "v2_700"), add_special_tokens=False)["input_ids"])
    assert a["input_ids"] != b["input_ids"] and all(l == -100 for l in b["labels"][:n_b])
