"""QLoRA supervised fine-tuning (NF4 base + LoRA r=16, alpha=32, dropout=0.05).

Sized for an 8 GB laptop GPU (RTX 4060): a 1-2B model with batch 4 x accum 4,
seq 512 and gradient checkpointing. Loss is computed on completion tokens only.
Training resumes automatically from the last checkpoint in `output_dir`.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.mixer import TrainExample
from ..models.prompts import DEFAULT_PROMPT, format_completion, format_prompt


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
    prompt_version: str = DEFAULT_PROMPT   # must equal the evaluation prompt (PLAN.md D10)
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    keep_checkpoints: bool = False   # delete checkpoint-* after the final adapter is saved
    extra: dict = field(default_factory=dict)


def tokenize_example(tokenizer, ex: TrainExample, max_len: int, prompt_version: str = DEFAULT_PROMPT) -> dict | None:
    """Prompt tokens get label -100, so only the completion is learned.
    Returns None if truncation would remove the entire completion."""
    prompt = format_prompt(ex.prompt, tokenizer, prompt_version)
    completion = format_completion(ex.completion.strip(), prompt_version) + (tokenizer.eos_token or "")
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
        from ..models.runner import check_4bit_support

        check_4bit_support("train.load_in_4bit")
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


def complete_checkpoints(out: Path) -> list[Path]:
    """checkpoint-* folders that finished writing (they contain trainer_state.json), oldest first.
    Incomplete ones (cut off mid-write by a crash or a hard kill) are deleted: resuming from them would fail."""
    done = []
    for ck in out.glob("checkpoint-*"):
        if not ck.is_dir():
            continue
        if (ck / "trainer_state.json").exists():
            done.append(ck)
        else:
            shutil.rmtree(ck, ignore_errors=True)

    def step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except ValueError:
            return -1
    return sorted(done, key=step)


def _job_callback(total_steps: int, on_progress, should_cancel, logging_steps: int):
    """TrainerCallback: report (step, total, loss) and stop cleanly at a step boundary on cancel (PLAN.md S5, 7.3)."""
    from transformers import TrainerCallback

    class _JobCallback(TrainerCallback):
        cancelled = False

        def on_train_begin(self, args, state, control, **kw):
            if on_progress:
                on_progress(state.global_step, state.max_steps or total_steps, None)

        def on_log(self, args, state, control, logs=None, **kw):
            if on_progress and logs and "loss" in logs:
                on_progress(state.global_step, state.max_steps or total_steps, float(logs["loss"]))

        def on_step_end(self, args, state, control, **kw):
            if should_cancel and state.global_step % max(1, logging_steps) == 0 and should_cancel():
                self.cancelled = True
                control.should_save = True            # keep a complete checkpoint so the job can resume
                control.should_training_stop = True
            return control

    return _JobCallback()


def train(cfg: TrainConfig, examples: list[TrainExample], on_progress=None, should_cancel=None) -> Path:
    """Fine-tune and return the adapter directory (output_dir/adapter).

    on_progress(step, total_steps, loss|None) is called at the start and at every logging step;
    should_cancel() is polled every `logging_steps`: if True, a checkpoint is saved, training stops at that step
    boundary and errors.Cancelled is raised (no adapter is written; a later run resumes from the checkpoint).
    """
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    from ..errors import Cancelled

    t_start = time.time()
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data, skipped = [], 0
    for ex in examples:
        t = tokenize_example(tokenizer, ex, cfg.max_seq_len, cfg.prompt_version)
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
    # Windows: DataLoader worker processes are spawned (not forked) and gain nothing for
    # pre-tokenised in-memory data, so keep loading in the main process. `extra` may override.
    extra = {"dataloader_num_workers": 0, **cfg.extra}
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
        **extra,
    )
    callback = _job_callback(total_steps, on_progress, should_cancel, cfg.logging_steps)
    trainer = Trainer(model=model, args=args, train_dataset=data,
                      data_collator=_Collator(tokenizer.pad_token_id), callbacks=[callback])
    checkpoints = complete_checkpoints(out)
    resume = str(checkpoints[-1]) if checkpoints else None
    load_seconds = time.time() - t_start
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    if callback.cancelled:
        raise Cancelled(f"training stopped at step {trainer.state.global_step} (checkpoint kept for resume)")
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    manifest = {
        "config": asdict(cfg),
        "n_examples": len(data),
        "n_skipped_too_long": skipped,
        "n_label_tokens": int(sum(sum(1 for l in d["labels"] if l != -100) for d in data)),
        "train_metrics": result.metrics,
        "resumed": resume is not None,
        "resumed_from": Path(resume).name if resume else None,
        "load_seconds": load_seconds,
        "wall_seconds": time.time() - t0,
        "loss_curve": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h],
    }
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not cfg.keep_checkpoints:     # only needed to resume an unfinished run (PLAN.md S5 cleanup)
        for ck in out.glob("checkpoint-*"):
            shutil.rmtree(ck, ignore_errors=True)
    return adapter_dir
