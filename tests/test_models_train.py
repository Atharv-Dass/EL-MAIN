"""CPU smoke tests for inference and QLoRA-style training on a tiny random model.

These prove the plumbing (prompt format, label masking, adapter save/load,
batched generation), not model quality. On the GPU the same code runs with
load_in_4bit=True.
"""

import pytest

from dreammachine.data import TrainExample, clean_solution
from dreammachine.generator import GenSpec, generate_many
from dreammachine.models import GenConfig, HFRunner, format_prompt
from dreammachine.train import TrainConfig, tokenize_example, train


def test_format_prompt_without_tokenizer():
    p = format_prompt("What is 2 + 3?")
    assert "Problem: What is 2 + 3?" in p and p.endswith("Solution:\n")


def test_tokenize_masks_prompt(tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ex = TrainExample(prompt="Asha has 5 apples.", completion="5 + 1 = 6\n#### 6", source="t", id="1")
    t = tokenize_example(tok, ex, max_len=512)
    n_prompt = len(tok(format_prompt(ex.prompt, tok), add_special_tokens=False)["input_ids"])
    assert all(l == -100 for l in t["labels"][:n_prompt])
    assert t["labels"][n_prompt:] == t["input_ids"][n_prompt:]
    assert t["input_ids"][-1] == tok.eos_token_id
    assert tokenize_example(tok, ex, max_len=5) is None  # prompt alone exceeds the limit


def test_runner_generates_batched(tiny_model_dir):
    runner = HFRunner(str(tiny_model_dir), gen=GenConfig(max_new_tokens=8, batch_size=3, n_samples=2,
                                                         temperature=0.7))
    out = runner.generate(["Asha has 5 apples.", "Ravi has 7 books.", "Mei has 3 eggs.", "Sam has 2."])
    assert len(out) == 4 and all(len(s) == 2 and all(isinstance(x, str) for x in s) for s in out)


def test_train_saves_adapter_and_resumes_reload(tiny_model_dir, tmp_path):
    items = generate_many(GenSpec(steps=2, digits=1), 16, seed=0)
    exs = [TrainExample(it.question, clean_solution(it.solution), "syn", it.id) for it in items]
    cfg = TrainConfig(base_model=str(tiny_model_dir), output_dir=str(tmp_path / "run"), load_in_4bit=False,
                      max_steps=4, per_device_batch_size=4, grad_accum=1, save_steps=2, logging_steps=1,
                      target_modules=["q_proj", "v_proj"])
    adapter = train(cfg, exs)
    assert (adapter / "adapter_config.json").exists()
    manifest = (tmp_path / "run" / "train_manifest.json").read_text(encoding="utf-8")
    assert '"n_examples": 16' in manifest
    runner = HFRunner(str(tiny_model_dir), adapter=str(adapter), gen=GenConfig(max_new_tokens=4))
    assert len(runner.generate(["Asha has 5 apples."])) == 1


def test_4bit_without_cuda_gives_clear_error(monkeypatch):
    torch = pytest.importorskip("torch")
    from dreammachine.models.runner import check_4bit_support

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"train\.load_in_4bit: false"):
        check_4bit_support("train.load_in_4bit")


@pytest.mark.slow
def test_overfit_one_example_end_to_end(tiny_model_dir, tmp_path):
    """Train on one problem until memorised, then evaluate it through the real
    diagnosis path. CORRECT here proves train/eval prompt formats agree."""
    from dreammachine.data import from_items
    from dreammachine.experiments.evaluate import evaluate

    item = generate_many(GenSpec(steps=1, digits=1, op_weights=(1, 0, 0, 0)), 1, seed=3)[0]
    ex = TrainExample(item.question, clean_solution(item.solution), "syn", item.id)
    cfg = TrainConfig(base_model=str(tiny_model_dir), output_dir=str(tmp_path / "overfit"), load_in_4bit=False,
                      max_steps=150, per_device_batch_size=1, grad_accum=1, learning_rate=1e-2, lora_r=32,
                      lora_alpha=64, lora_dropout=0.0, warmup_ratio=0.0, save_steps=1000, logging_steps=50,
                      target_modules="all-linear", gradient_checkpointing=False)
    adapter = train(cfg, [ex])
    runner = HFRunner(str(tiny_model_dir), adapter=str(adapter), gen=GenConfig(max_new_tokens=48))
    records = evaluate(runner, from_items([item]))
    assert records[0].correct, records[0].text
