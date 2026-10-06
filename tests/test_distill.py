"""Self-distillation of solution texts (dreammachine.data.distill, PLAN.md D14): CPU plumbing with a fake runner."""

from dreammachine.data.distill import DistillConfig, accept, distill_arm, rebalance
from dreammachine.data.mixer import TrainExample, total_tokens

LONG_OK = "First, 3 + 4 = 7 apples.\nThen 7 * 2 = 14, " + "so we double it carefully. " * 6 + "\n\\boxed{14}"
WRONG_ANSWER = "3 + 4 = 7\n7 * 2 = 14\n\\boxed{15}"
WRONG_STEP = "3 + 4 = 8\n8 * 2 = 14\n\\boxed{14}"
NO_WORK = "The answer is \\boxed{14}"


def _row(i: int, source: str) -> TrainExample:
    return TrainExample(f"question {i} with some words", "3 + 4 = 7\n7 * 2 = 14\n#### 14", source, f"{source}-{i}")


class FakeRunner:
    def __init__(self, answers: list[str], truncated: list[bool] | None = None):
        self.answers, self.cut, self.calls = answers, truncated or [False] * len(answers), 0

    def generate(self, questions):
        self.calls += 1
        self.last_truncated = [list(self.cut) for _ in questions]
        return [list(self.answers) for _ in questions]


def test_accept_rules():
    assert accept(LONG_OK, "14")
    assert not accept(LONG_OK, "14", truncated=True)
    assert not accept(WRONG_ANSWER, "14")      # wrong final answer
    assert not accept(WRONG_STEP, "14")        # right answer, wrong written step
    assert not accept(NO_WORK, "14")           # no equation to check


def test_distill_picks_first_accepted_sample_and_keeps_original_otherwise():
    rows = [_row(i, "gsm8k") for i in range(4)] + [_row(i, "synthetic:train") for i in range(4)]
    big = 10_000
    out, st = distill_arm(rows, FakeRunner([WRONG_STEP, LONG_OK]), DistillConfig(samples=2, chunk=3), big, 1, 0)
    assert st.n_items == 8 and st.n_distilled == 8
    assert {r.completion for r in out} == {LONG_OK}
    assert st.by_source == {"real": [4, 4], "synthetic": [4, 4]}
    out, st = distill_arm(rows, FakeRunner([WRONG_STEP, WRONG_ANSWER]), DistillConfig(chunk=3), big, 1, 0)
    assert st.n_kept_original == 8 and sorted(r.id for r in out) == sorted(r.id for r in rows)
    out, st = distill_arm(rows, FakeRunner([LONG_OK], truncated=[True]), DistillConfig(chunk=3), big, 1, 0)
    assert st.n_distilled == 0                 # cut-off answers are never kept


def test_distill_keeps_budget_and_ratio_and_stops_early():
    rows = [_row(i, "synthetic:train" if i % 4 else "gsm8k") for i in range(400)]
    budget, ratio = 2000, 3
    runner = FakeRunner([LONG_OK])
    out, st = distill_arm(rows, runner, DistillConfig(samples=1, chunk=16), budget, ratio, 0)
    assert total_tokens(out) <= budget
    real = total_tokens([r for r in out if r.source == "gsm8k"])
    syn = total_tokens([r for r in out if r.source.startswith("synthetic")])
    assert real <= budget // 4 and syn <= budget * 3 // 4
    assert real > 0.8 * budget / 4 and syn > 0.8 * budget * 3 / 4
    assert st.n_items < len(rows) and runner.calls < len(rows) / 16   # did not distill rows that cannot fit


def test_rebalance_shuffles_deterministically():
    rows = [_row(i, "synthetic:train" if i % 2 else "gsm8k") for i in range(40)]
    a, b = rebalance(rows, 200, 1, seed=3), rebalance(rows, 200, 1, seed=3)
    assert [r.id for r in a] == [r.id for r in b] and total_tokens(a) <= 200


# ---------------------------------------------------------------- code-written step style (option b)
def test_steps_style_keeps_equations_and_answer_and_quotes_the_problem():
    from dreammachine.data.style import restyle, steps_style
    from dreammachine.diagnosis.cot_parse import parse_equations

    q = ("Hana has 5 cookies in the bakery. Lucia bakes 9 muffins. Hana eats 2 of the cookies. "
         "Hana receives 3 cookies from a friend. How many cookies does Hana end up with?")
    sol = "Hana then has 5 - 2 = 3 left.\nHana now has 3 + 3 = 6.\n#### 6"
    out = steps_style(q, sol)
    assert out.endswith("\n#### 6") and out.startswith("Let's solve this step by step.")
    assert "We need to find: How many cookies does Hana end up with?" in out
    assert "**Step 1: Hana has 5 cookies in the bakery. Hana eats 2 of the cookies.**" in out   # distractor skipped
    assert "**Step 2: Hana receives 3 cookies from a friend.**" in out      # the computed 3 is not looked up
    assert [(e.lhs, e.correct) for e in parse_equations(out)] == [(e.lhs, e.correct) for e in parse_equations(sol)]
    assert steps_style(q, "no final line") == "no final line"
    row = TrainExample(q, sol, "synthetic:train", "x")
    assert restyle([row], "terse") == [row] and restyle([row], "steps")[0].completion == out


def test_unknown_solution_style_is_rejected():
    import pytest

    from dreammachine.data.style import restyle

    with pytest.raises(ValueError):
        restyle([], "fancy")
