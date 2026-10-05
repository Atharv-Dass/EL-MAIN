"""Final-line extractors for v2_700-style answers (PLAN.md §8.5)."""

from fractions import Fraction

import pytest

from dreammachine.data import from_items
from dreammachine.diagnosis.extract import extract_lastline, extract_lastline_v0, last_line, lastline_format_ok
from dreammachine.experiments.evaluate import evaluate, summarize
from dreammachine.generator import GenSpec, generate_many
from dreammachine.models.runner import EchoRunner


@pytest.mark.parametrize("text, tolerant, v0, ok", [
    ("Step 1: 3 + 3 = 6\n18", 18, 18, True),
    ("He has 3 boxes of 6.\n3 * 6 = 18", 18, 3, False),     # the friend's bug: first number on the line
    ("Total is 18 eggs.\n18.", 18, 18, False),               # trailing period: value kept, not a bare number
    ("**$1,234**", 1234, 1234, False),                       # bold, dollar, thousands comma
    ("Final answer: 42 apples.", 42, 42, False),
    ("-2.5", Fraction(-5, 2), Fraction(-5, 2), True),
    ("18\n\n   \n", 18, 18, True),                            # trailing blank lines are ignored
    ("<think>so 99</think>\n7", 7, 7, True),                  # thinking text is not the last line
    ("1,234", 1234, 1234, False),
    ("no number here", None, None, False),
    ("", None, None, False),
])
def test_lastline_extractors(text, tolerant, v0, ok):
    assert extract_lastline(text) == (None if tolerant is None else Fraction(tolerant))
    assert extract_lastline_v0(text) == (None if v0 is None else Fraction(v0))
    assert lastline_format_ok(text) is ok


def test_last_line():
    assert last_line("a\n\n b \n") == "b" and last_line("  \n") == ""


def test_evaluate_records_lastline_fields():
    items = generate_many(GenSpec(steps=2, digits=1), 2, seed=0)
    exs = from_items(items)
    answers = {exs[0].question: f"work\n{exs[0].answer}", exs[1].question: "1 + 1 = 2\n#### 2"}
    recs = evaluate(EchoRunner(answers), exs)
    d0, d1 = recs[0].diagnosis, recs[1].diagnosis
    assert d0["lastline"] == exs[0].answer and d0["format_ok"] is True
    assert d1["lastline"] == "2" and d1["lastline_v0"] == "2" and d1["format_ok"] is False
    assert summarize(recs)["format_ok_share"] == 0.5
    assert "truncated" not in d0  # EchoRunner does not report truncation
