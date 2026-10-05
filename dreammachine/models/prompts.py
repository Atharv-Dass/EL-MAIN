"""Prompt versions, shared by evaluation and training (PLAN.md D10, §8.5).

Using the identical format in both places matters: if training and evaluation
prompts differ, part of any measured gain is just format adaptation. One project uses
one `prompt_version` everywhere; the others exist to compare prompts.

    dm_v1       ours (default). Instruction + "Problem: ..." in the user turn.
    v2_700      the friend's prompt (text received 2026-10-04), as the SYSTEM message; user turn = question.
    qwen_boxed  the math prompt recommended in the Qwen3 model card ("Best Practices"), after the question.
"""

from __future__ import annotations

INSTRUCTION = (
    "Solve the math word problem step by step. Write every calculation as an equation, "
    "for example 12 + 7 = 19. On the last line write '#### ' followed by the final number only."
)

# Byte-for-byte as received; pinned by tests (string and sha256). Never edit.
V2_700 = (
    "Solve the math problem step by step. At the end, write a separate final line containing only the "
    "numeric answer, with no units, dollar sign, commas, bold formatting, or extra text. Example final line: 18."
)

QWEN_BOXED = "Please reason step by step, and put your final answer within \\boxed{}."

PROMPTS: dict[str, str] = {"dm_v1": INSTRUCTION, "v2_700": V2_700, "qwen_boxed": QWEN_BOXED}
DEFAULT_PROMPT = "dm_v1"

# Where the prompt text goes: before the problem in the user turn, as the system message, or after the question.
_PLACEMENT = {"dm_v1": "user_prefix", "v2_700": "system", "qwen_boxed": "user_suffix"}


def get_prompt(prompt_version: str) -> str:
    try:
        return PROMPTS[prompt_version]
    except KeyError:
        raise KeyError(f"unknown prompt_version {prompt_version!r}; available: {sorted(PROMPTS)}") from None


def _user_prefix(text: str, question: str) -> str:
    return f"{text}\n\nProblem: {question}"


def build_messages(question: str, prompt_version: str = DEFAULT_PROMPT,
                   system_role: bool = True) -> list[dict[str, str]]:
    """Chat messages for one problem. `system_role=False` is the fallback for chat templates without a
    system role (e.g. Gemma): the prompt then goes into the user turn exactly like dm_v1's."""
    text = get_prompt(prompt_version)
    placement = _PLACEMENT[prompt_version]
    if placement == "system":
        if system_role:
            return [{"role": "system", "content": text}, {"role": "user", "content": question}]
        return [{"role": "user", "content": _user_prefix(text, question)}]
    if placement == "user_suffix":
        return [{"role": "user", "content": f"{question}\n{text}"}]
    # System prompts are not supported by every chat template (e.g. Gemma), so dm_v1 uses the user turn.
    return [{"role": "user", "content": _user_prefix(text, question)}]


def _apply(tokenizer, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Qwen3: disable thinking mode; other templates ignore unknown variables.
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def format_prompt(question: str, tokenizer=None, prompt_version: str = DEFAULT_PROMPT) -> str:
    """Render the prompt with the model's chat template when it has one."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = build_messages(question, prompt_version)
        if messages[0]["role"] != "system":
            return _apply(tokenizer, messages)
        try:
            return _apply(tokenizer, messages)
        except Exception as e:  # jinja2 TemplateError, e.g. "System role not supported"
            if "system" not in str(e).lower():
                raise
            return _apply(tokenizer, build_messages(question, prompt_version, system_role=False))
    # No chat template: the user-turn text, then a completion cue.
    return build_messages(question, prompt_version, system_role=False)[0]["content"] + "\nSolution:\n"
