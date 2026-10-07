"""Shared fixtures. `tiny_model_dir` builds a randomly initialised ~100k-parameter Llama
with a locally trained byte-level BPE tokenizer, so the runner and trainer can be
smoke-tested on CPU with no network access (Hugging Face Hub is not needed)."""

from __future__ import annotations

import pytest

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from dreammachine.generator import GenSpec, generate_many
    from dreammachine.models.prompts import INSTRUCTION

    corpus = [INSTRUCTION] + [f"{it.question}\n{it.solution}" for it in generate_many(GenSpec(steps=3), 300, seed=0)]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<pad>", "<eos>", "<|user|>", "<|assistant|>", "<|end|>"]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=800, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<pad>")
    hf_tok.chat_template = CHAT_TEMPLATE

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      pad_token_id=hf_tok.pad_token_id, eos_token_id=hf_tok.eos_token_id,
                      bos_token_id=None)
    model = LlamaForCausalLM(cfg)
    out = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(out)
    hf_tok.save_pretrained(out)
    return out
