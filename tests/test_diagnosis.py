import random
from fractions import Fraction

import pytest

from dreammachine.diagnosis import (
    ErrorType, ReferenceTrace, classify, cohen_kappa, confusion_matrix, error_distribution,
    extract_answer, parse_equations, stratified_sample,
)
from dreammachine.generator import GenSpec, generate, generate_many

# Two GSM8K train examples (reference solutions with calculator annotations).
GSM_Q1 = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
          "in May. How many clips did Natalia sell altogether in April and May?")
GSM_A1 = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
          "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72")
GSM_Q2 = ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
          "babysitting. How much did she earn?")
GSM_A2 = ("Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
          "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10")


# ------------------------------------------------------------------ extract
@pytest.mark.parametrize("text,expected", [
    ("so the total is 5.\n#### 72", 72),
    ("#### 1,234", 1234),
    ("The answer is $18.50.", Fraction(37, 2)),
    ("Therefore \\boxed{42}", 42),
    ("<think>maybe 3</think> We get 7 apples.", 7),
    ("She has 3 + 4 = <<3+4=7>>7 now", 7),
    ("no numbers here", None),
    ("#### -5", -5),
])
def test_extract_answer(text, expected):
    got = extract_answer(text)
    assert got == (None if expected is None else Fraction(expected))


# -------------------------------------------------------------- cot parsing
def test_parse_equations_correct_and_slip():
    eqs = parse_equations("First 48 * 7 = 336. Then 336 - 20 = 306.")
    assert [e.correct for e in eqs] == [True, False]
    assert eqs[1].lhs_value == 316 and eqs[1].rhs_value == 306
    assert eqs[0].operands == [48, 7] and eqs[0].ops == ["*"]


def test_parse_equations_normalisation():
    eqs = parse_equations("It costs $1,200 × 3 = $3,600 and 12 x 4 = 48 and 10 ÷ 4 = 2.5")
    assert [(e.lhs_value, e.rhs_value, e.correct) for e in eqs] == [
        (3600, 3600, True), (48, 48, True), (Fraction(5, 2), Fraction(5, 2), True)]


def test_parse_equations_chained_and_numbered():
    eqs = parse_equations("1. 25 * 9 = 225 + 5 = 230\n2. 5 + 3 = 8")
    assert [(str(e.lhs_value), str(e.rhs_value)) for e in eqs] == [("225", "225"), ("230", "230"), ("8", "8")]


def test_parse_equations_rounding_and_parens():
    eqs = parse_equations("(10 + 2) / 7 = 1.71 and 10 / 3 = 3.4")
    assert eqs[0].correct and eqs[0].lhs == "(10 + 2) / 7"
    assert not eqs[1].correct


def test_parse_equations_ignores_non_equations():
    assert parse_equations("In 2024 she was 12 years old. x = 5 is unknown.") == []


# --------------------------------------------------- GSM8K reference traces
@pytest.mark.parametrize("q,a,answer,n_steps", [(GSM_Q1, GSM_A1, 72, 2), (GSM_Q2, GSM_A2, 10, 2)])
def test_gsm8k_reference_classifies_as_correct(q, a, answer, n_steps):
    trace = ReferenceTrace.from_gsm8k(q, a)
    assert trace.answer == answer and len(trace.steps) == n_steps
    d = classify(a, trace)
    assert d.error_type is ErrorType.CORRECT
    assert d.n_slips == 0 and d.progress == 1.0


def test_gsm8k_distractor_detection():
    trace = ReferenceTrace.from_gsm8k(GSM_Q2, GSM_A2)
    assert Fraction(50) in trace.relevant_numbers and Fraction(12) in trace.relevant_numbers


# ------------------------------------------------ synthetic item taxonomy
@pytest.fixture
def item():
    spec = GenSpec(steps=3, digits=2, op_weights=(0.5, 0.5, 0, 0), n_distractors=1)
    return generate(spec, random.Random(5))


def test_reference_solution_is_correct(item):
    trace = ReferenceTrace.from_item(item)
    d = classify(item.solution, trace)
    assert d.error_type is ErrorType.CORRECT and d.progress == 1.0


def test_all_generated_solutions_classify_correct():
    for it in generate_many(GenSpec(steps=5, digits=3, n_distractors=2, merge=True), 300, seed=1):
        d = classify(it.solution, ReferenceTrace.from_item(it))
        assert d.error_type is ErrorType.CORRECT and d.n_slips == 0 and d.progress == 1.0


def _spaced(expr: str) -> str:
    for sym in "+-*/":
        expr = expr.replace(sym, f" {sym} ")
    return expr


def test_arithmetic_slip(item):
    trace = ReferenceTrace.from_item(item)
    first = trace.steps[0]
    wrong = first.value + 1
    d = classify(f"{_spaced(first.expr)} = {wrong}\n#### {wrong}", trace)
    assert d.error_type is ErrorType.ARITHMETIC_SLIP
    assert d.slip_index == 0 and d.divergence_step == 0


def test_distractor_use(item):
    trace = ReferenceTrace.from_item(item)
    m = item.distractor_numbers[0]
    start = int(sorted(trace.relevant_numbers)[0])
    resp = f"{start} + {m} = {start + m}\n#### {start + m}"
    assert classify(resp, trace).error_type is ErrorType.DISTRACTOR_USE


def test_plan_error_and_divergence(item):
    trace = ReferenceTrace.from_item(item)
    first = trace.steps[0]
    d = classify(f"{_spaced(first.expr)} = {first.value}\n#### {first.value}", trace)
    assert first.value != trace.answer
    assert d.error_type is ErrorType.PLAN_ERROR
    assert d.divergence_step == 1 and 0 < d.progress < 1


def test_format_and_unverifiable(item):
    trace = ReferenceTrace.from_item(item)
    assert classify("I am not sure.", trace).error_type is ErrorType.FORMAT_ERROR
    assert classify(f"#### {item.answer + 3}", trace).error_type is ErrorType.UNVERIFIABLE
    last = trace.steps[-1]
    resp = f"{_spaced(last.expr)} = {last.value}\nSo the answer is 1."
    assert classify(resp, trace).error_type is ErrorType.FORMAT_ERROR


def test_error_distribution():
    from dreammachine.diagnosis import Diagnosis
    ds = [Diagnosis(ErrorType.CORRECT, "1", "1", 0, 0), Diagnosis(ErrorType.PLAN_ERROR, "2", "1", 1, 0),
          Diagnosis(ErrorType.ARITHMETIC_SLIP, "2", "1", 1, 1)]
    assert error_distribution(ds) == {"ARITHMETIC_SLIP": 0.5, "PLAN_ERROR": 0.5}


# ---------------------------------------------------------------- agreement
def test_cohen_kappa():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    # po = 0.5, pe = 0.5 -> kappa 0
    assert cohen_kappa(list("abab"), list("aabb")) == pytest.approx(0.0)
    a = ["x"] * 20 + ["y"] * 30 + ["x"] * 5 + ["y"] * 45
    b = ["x"] * 20 + ["x"] * 30 + ["y"] * 5 + ["y"] * 45
    # classic textbook: po=0.65, pe=0.5*0.25... computed directly
    po = 65 / 100
    pa_x, pb_x = 25 / 100, 50 / 100
    pe = pa_x * pb_x + (1 - pa_x) * (1 - pb_x)
    assert cohen_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))
    assert confusion_matrix(a, b)["y"]["x"] == 30


def test_stratified_sample_balances_rare_types():
    recs = [("common", i) for i in range(100)] + [("rare", i) for i in range(5)]
    s = stratified_sample(recs, key=lambda r: r[0], n=10, seed=0)
    assert sum(r[0] == "rare" for r in s) == 5 and len(s) == 10
