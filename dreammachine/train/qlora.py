"""QLoRA supervised fine-tuning (NF4 base + LoRA r=16, alpha=32, dropout=0.05).

Sized for an 8 GB laptop GPU (RTX 4060): a 1-2B model with batch 4 x accum 4,
seq 512 and gradient checkpointing. Loss is computed on completion tokens only.
Training resumes automatically from the last checkpoint in `output_dir`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.mixer import TrainExample
from ..models.prompts import format_prompt


@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    learning_rate: float = 2e-4
    num_epochs: float = 2.0
    max_steps: int = -1
    per_device_batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    seed: int = 0
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    extra: dict = field(default_factory=dict)


def tokenize_example(tokenizer, ex: TrainExample, max_len: int) -> dict | None:
    """Prompt tokens get label -100, so only the completion is learned.
    Returns None if truncation would remove the entire completion."""
    prompt = format_prompt(ex.prompt, tokenizer)
    completion = ex.completion.strip() + (tokenizer.eos_token or "")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(p_ids) >= max_len:
        return None
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class _Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, value: int) -> "torch.Tensor":
            return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

        return {"input_ids": pad("input_ids", self.pad_id), "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100)}


def _load_model(cfg: TrainConfig):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    cuda = torch.cuda.is_available()
    kwargs: dict = {}
    if cfg.load_in_4bit:
        if not cuda:
            raise RuntimeError("load_in_4bit needs a CUDA GPU; set load_in_4bit: false for CPU smoke runs")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    kwargs["dtype"] = torch.bfloat16 if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.config.use_cache = False
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


def train(cfg: TrainConfig, examples: list[TrainExample]) -> Path:
    """Fine-tune and return the adapter directory (output_dir/adapter)."""
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data, skipped = [], 0
    for ex in examples:
        t = tokenize_example(tokenizer, ex, cfg.max_seq_len)
        if t is None:
            skipped += 1
        else:
            data.append(t)
    if not data:
        raise ValueError("no training examples fit in max_seq_len")

    model = _load_model(cfg)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    # transformers v5 removed `warmup_ratio`; convert it to steps (works on v4 and v5).
    steps_per_epoch = math.ceil(len(data) / (cfg.per_device_batch_size * cfg.grad_accum))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(steps_per_epoch * cfg.num_epochs)
    warmup_steps = int(round(cfg.warmup_ratio * total_steps))
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=bf16,
        fp16=cuda and not bf16,
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        seed=cfg.seed,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=False,  # already enabled on the model where requested
        dataloader_pin_memory=cuda,
        **cfg.extra,
    )
    trainer = Trainer(model=model, args=args, train_dataset=data,
                      data_collator=_Collator(tokenizer.pad_token_id))
    resume = any(out.glob("checkpoint-*"))
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    manifest = {
        "config": asdict(cfg),
        "n_examples": len(data),
        "n_skipped_too_long": skipped,
        "n_label_tokens": int(sum(sum(1 for l in d["labels"] if l != -100) for d in data)),
        "train_metrics": result.metrics,
        "resumed": resume,
        "wall_seconds": time.time() - t0,
    }
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    return adapter_dir
