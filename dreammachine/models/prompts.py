"""One prompt format, shared by evaluation and training.

Using the identical format in both places matters: if training and evaluation
prompts differ, part of any measured gain is just format adaptation.
"""

from __future__ import annotations

INSTRUCTION = (
    "Solve the math word problem step by step. Write every calculation as an equation, "
    "for example 12 + 7 = 19. On the last line write '#### ' followed by the final number only."
)


def build_messages(question: str) -> list[dict[str, str]]:
    # System prompts are not supported by every chat template (e.g. Gemma), so the
    # instruction goes inside the user turn.
    return [{"role": "user", "content": f"{INSTRUCTION}\n\nProblem: {question}"}]


def format_prompt(question: str, tokenizer=None) -> str:
    """Render the prompt with the model's chat template when it has one."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            # Qwen3: disable thinking mode; other templates ignore unknown variables.
            return tokenizer.apply_chat_template(build_messages(question), enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(build_messages(question), **kwargs)
    return f"{INSTRUCTION}\n\nProblem: {question}\nSolution:\n"
