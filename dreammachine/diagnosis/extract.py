"""Final-answer extraction from free-form model output."""

from __future__ import annotations

import re
from fractions import Fraction

_NUM = r"-?\$?\s?\d[\d,]*(?:\.\d+)?"
_THINK = re.compile(r"<think>.*?</think>", re.S)
_HASH = re.compile(rf"####\s*({_NUM})")
_BOXED = re.compile(rf"\\boxed\{{\s*({_NUM})\s*\}}")
_ANSWER_IS = re.compile(rf"answer\s*(?:is|:|=)\s*[:\s]*\**\s*({_NUM})", re.I)
_ANY_NUM = re.compile(_NUM)


def parse_number(s: str) -> Fraction | None:
    """'$1,234.50' -> Fraction(2469, 2). Returns None if it is not a number."""
    t = s.replace("$", "").replace(",", "").replace(" ", "").strip()
    if t.endswith("."):
        t = t[:-1]
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def strip_reasoning_tags(text: str) -> str:
    """Remove <think> blocks (e.g. Qwen3 thinking mode) so they cannot be mistaken for the answer."""
    return _THINK.sub(" ", text)


def extract_answer(text: str) -> Fraction | None:
    """Priority: '#### n' > '\\boxed{n}' > 'the answer is n' > last number in the text."""
    text = strip_reasoning_tags(text)
    for pattern in (_HASH, _BOXED, _ANSWER_IS):
        hits = pattern.findall(text)
        if hits:
            return parse_number(hits[-1])
    # Drop GSM8K-style calculator annotations before the last-number fallback.
    plain = re.sub(r"<<[^<>]*>>", " ", text)
    hits = _ANY_NUM.findall(plain)
    return parse_number(hits[-1]) if hits else None


_BARE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def last_line(text: str) -> str:
    """The last non-empty line, after removing <think> blocks."""
    lines = [ln.strip() for ln in strip_reasoning_tags(text).splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_lastline(text: str) -> Fraction | None:
    """Final-line answer for prompts like v2_700 ("a final line containing only the numeric answer").
    Tolerant: ignores bold markers (**), dollar signs, thousands commas and a trailing period, and takes
    the *last* number on the last non-empty line, so "3 * 6 = 18" gives 18."""
    line = last_line(text).replace("**", "").replace("$", "").strip().rstrip(".").strip()
    hits = _ANY_NUM.findall(line)
    return parse_number(hits[-1]) if hits else None


def lastline_format_ok(text: str) -> bool:
    """True only if the last non-empty line is strictly a bare number (e.g. "18", "-2.5"):
    no units, $, commas, bold, trailing period or other text. Recorded next to the tolerant value."""
    return bool(_BARE_NUMBER.fullmatch(last_line(text)))


def extract_lastline_v0(text: str) -> Fraction | None:
    """The friend's Colab behaviour, kept only to measure disagreement: the *first* number on the last
    line, so "3 * 6 = 18" gives 3 (the bug PLAN.md §8.5 fixes in `extract_lastline`)."""
    hits = _ANY_NUM.findall(last_line(text))
    return parse_number(hits[0]) if hits else None


def answers_match(pred: Fraction | None, gold: Fraction | int | str, tol: Fraction = Fraction(1, 10**6)) -> bool:
    if pred is None:
        return False
    g = gold if isinstance(gold, Fraction) else parse_number(str(gold))
    return g is not None and abs(pred - g) <= tol
