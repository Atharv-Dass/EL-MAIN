"""Code-written solution style (option b of docs/RUNNING.md §10; dev-set test approved by the user 2026-10-06).

Fine-tuning on terse solutions ("Hana now has 3 + 7 = 10." / "#### 10") made the model copy that style and lowered
GSM8K accuracy. `steps_style` rewrites a solution, by code and deterministically, into a longer step-by-step form
closer to how the model itself answers: an opening line, the goal, one numbered step per solution line that first
quotes the problem sentence it uses, and a closing sentence. Every equation, number and the final `#### n` line are
kept exactly, so answers and equation checks are unchanged. Works on synthetic and real GSM8K solutions alike
(a GSM8K step whose sentence cannot be found is written without a quote).

Selected per config: `data.solution_style: terse` (default, unchanged data) or `steps`.
"""

from __future__ import annotations

import re

from .mixer import TrainExample

SOLUTION_STYLES = ("terse", "steps")

_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_RESULT = re.compile(r"=\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _norm(n: str) -> str:
    n = n.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def _numbers(text: str) -> list[str]:
    return [_norm(m.group()) for m in _NUM.finditer(text)]


def _operands(line: str) -> tuple[list[str], list[str]]:
    """Numbers used on a line, and the results it computes (the numbers right after an '=')."""
    results = [_norm(m.group(1)) for m in _RESULT.finditer(line)]
    nums = _numbers(line)
    ops = list(nums)
    for r in results:
        if r in ops:
            ops.remove(r)
    return ops, results


def _split_question(question: str) -> tuple[list[str], str | None]:
    sents = [s.strip() for s in _SENT.split(question.strip()) if s.strip()]
    facts = [s for s in sents if "?" not in s]
    asked = " ".join(s for s in sents if "?" in s) or None
    return facts, asked


def steps_style(question: str, solution: str) -> str:
    """Rewrite a `... #### n` solution into the longer step-by-step style (see module doc)."""
    lines = [l.strip() for l in solution.strip().split("\n") if l.strip()]
    if not lines or not lines[-1].startswith("####"):
        return solution
    final, body = lines[-1], lines[:-1]
    answer = final[4:].strip()
    facts, asked = _split_question(question)
    out = ["Let's solve this step by step."]
    if asked:
        out.append(f"We need to find: {asked}")
    seen_results: set[str] = set()
    pos = 0
    for k, line in enumerate(body, 1):
        ops, results = _operands(line)
        # Numbers computed by earlier steps are not looked up in the problem text (they may also appear there by
        # chance) unless the step uses nothing else.
        want = [n for n in ops if n not in seen_results] or ops
        quotes: list[str] = []
        j = pos
        while want and j < len(facts):
            hit = [n for n in want if n in _numbers(facts[j])]
            if hit:
                quotes.append(facts[j])
                for n in hit:
                    want.remove(n)
                pos = j + 1
            j += 1
        seen_results.update(results)
        head = f"Step {k}: " + " ".join(quotes) if quotes else f"Step {k}:"
        out.append(f"**{head}**\n{line}")
    out.append(f"So the answer is {answer}.")
    out.append(final)
    return "\n\n".join(out[:-1]) + "\n" + out[-1]


def restyle(rows: list[TrainExample], style: str) -> list[TrainExample]:
    if style not in SOLUTION_STYLES:
        raise ValueError(f"unknown data.solution_style {style!r}; available: {SOLUTION_STYLES}")
    if style == "terse":
        return list(rows)
    return [TrainExample(r.prompt, steps_style(r.prompt, r.completion), r.source, r.id) for r in rows]
