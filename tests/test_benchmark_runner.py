"""Standalone benchmark CLI, resume, prompt-version separation and the importer (PLAN.md §8.4-8.5).
CPU only: EchoRunner and local JSONL fixtures, no network."""

import json

import pytest

from dreammachine.benchmarks import registry
from dreammachine.benchmarks.__main__ import main
from dreammachine.benchmarks.importer import import_results
from dreammachine.benchmarks.runner import output_path, read_rows, run_benchmark
from dreammachine.data import Example, save_jsonl
from dreammachine.models.runner import EchoRunner, GenConfig
from dreammachine.store import Store


def _gsm_like(n=6):
    """Tiny GSM8K-format set: gold = (i+3) + (i+4), with a <<calculator>> reference trace."""
    out = []
    for i in range(n):
        a, b = i + 3, i + 4
        out.append(Example(id=f"t-{i}", source="gsm8k", question=f"Tom has {a} apples and buys {b} more. How many?",
                           answer=str(a + b), solution=f"{a}+{b} = <<{a}+{b}={a + b}>>{a + b}\n#### {a + b}"))
    return out


@pytest.fixture
def data(tmp_path):
    exs = _gsm_like()
    path = save_jsonl(exs, tmp_path / "set.jsonl")
    return exs, f"jsonl:{path}", tmp_path


class CountingEcho(EchoRunner):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.asked = []

    def generate(self, questions):
        self.asked += questions
        return super().generate(questions)


def _answers(exs, right=(0, 2, 4)):
    return {e.question: (f"{e.solution.splitlines()[0].split('=')[0]}= {e.answer}\n#### {e.answer}" if i in right
                         else "#### 0") for i, e in enumerate(exs)}


def test_cli_run_scores_stores_and_records_provenance(data):
    exs, name, tmp = data
    db, out = str(tmp / "dm.db"), str(tmp / "bench")
    res = main(["run", "--model", "echo-model", "--benchmark", name, "--db", db, "--out", out],
               runner_factory=lambda m, g, q: EchoRunner(_answers(exs)))
    r = res[name]
    assert r[name] == 0.5 and r["n_examples"] == 6 and r["format_ok_share"] == 0.0
    rows = read_rows(output_path(out, "echo-model", name, "qwen_boxed"))
    assert len(rows) == 6
    for key in ("model_id", "prompt_version", "gen_settings", "batch_secs", "git_commit", "packages", "gold"):
        assert key in rows[0], key
    assert rows[0]["prompt_version"] == "qwen_boxed" and rows[0]["gen_settings"]["max_new_tokens"] == 512
    store = Store(db)
    run = store.get_run(r["run_id"])
    assert run["kind"] == "eval" and run["config"]["prompt_version"] == "qwen_boxed"
    assert len(store.get_responses(r["run_id"])) == 6


def test_resume_skips_done_rows_and_survives_a_cut_line(data):
    exs, name, tmp = data
    bench, gen = registry.get(name), GenConfig()
    runner = CountingEcho(_answers(exs))
    run_benchmark(bench, bench.load(limit=3), runner, "m", gen, tmp / "o", log=lambda s: None)
    run_benchmark(bench, bench.load(), runner, "m", gen, tmp / "o", log=lambda s: None)
    assert len(runner.asked) == 6 and len(set(runner.asked)) == 6          # the first 3 were not asked again
    path = output_path(tmp / "o", "m", name, "qwen_boxed")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:25], encoding="utf-8")   # sudden stop mid-write
    res = run_benchmark(bench, bench.load(), runner, "m", gen, tmp / "o", log=lambda s: None)
    assert len(runner.asked) == 7 and runner.asked[-1] == exs[5].question   # only the cut answer is redone
    assert len(res["records"]) == 6 and res["summary"]["accuracy"] == 0.5


def test_prompt_versions_never_mix(data):
    exs, name, tmp = data
    bench = registry.get(name)
    a = run_benchmark(bench, exs, EchoRunner(_answers(exs)), "m", GenConfig(), tmp / "o", log=lambda s: None)
    b = run_benchmark(bench, exs, EchoRunner(_answers(exs)), "m", GenConfig(prompt_version="v2_700"), tmp / "o",
                      log=lambda s: None)
    assert a["path"] != b["path"] and "v2_700" in b["path"]
    assert {r["prompt_version"] for r in read_rows(output_path(tmp / "o", "m", name, "v2_700"))} == {"v2_700"}
    with pytest.raises(ValueError, match="different"):   # same file, other generation settings
        run_benchmark(bench, exs, EchoRunner(), "m", GenConfig(max_new_tokens=64), tmp / "o", log=lambda s: None)


def _friend_file(path):
    rows = [
        {"idx": 0, "response": "3 + 4 = 7\n7"},                       # all three extractors agree
        {"idx": 1, "response": "4 * 1 = 4\n4 + 5 = 9"},               # v0 takes the first number (4)
        {"idx": 2, "response": "So the total is 11.\n**11**"},        # tolerant: bold is ignored
        {"idx": 3, "response": "#### 13\nThanks!"},                    # no number on the last line
        {"idx": 0, "response": "3 + 4 = 7\n7"},                       # resumable file repeated an index
        {"idx": 99, "response": "1"},                                   # not in the benchmark
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "prompt_version": "v2_700",
                                "gen_settings": {"max_new_tokens": 700}, "batch_secs": 1.5}) + "\n")
    return path


def test_importer_rescores_with_three_extractors(data):
    exs, name, tmp = data
    res = import_results(_friend_file(tmp / "friend.jsonl"), name)
    rep, meta = res["report"], res["meta"]
    assert (rep["id_field"], rep["text_field"]) == ("idx", "response")
    assert rep["n_scored"] == 4 and rep["n_duplicates"] == 1 and rep["n_unmatched"] == 1
    assert rep["accuracy"] == {"extract_answer": 1.0, "extract_lastline": 0.75, "extract_lastline_v0": 0.5}
    assert rep["disagreements"]["extract_lastline vs extract_lastline_v0"]["examples"] == ["t-1"]
    assert rep["disagreements"]["extract_answer vs extract_lastline"]["examples"] == ["t-3"]
    assert meta == {"model": "Qwen/Qwen2.5-1.5B-Instruct", "prompt_version": "v2_700",
                    "gen_settings": {"max_new_tokens": 700}}
    assert rep["format_ok_share"] == 0.25
    assert {r.example_id: r.error_type for r in res["records"]}["t-3"] == "CORRECT"


def test_importer_cli_stores_a_run(data):
    exs, name, tmp = data
    db = str(tmp / "dm.db")
    out = main(["import-jsonl", str(_friend_file(tmp / "f.jsonl")), "--benchmark", name, "--db", db])
    store = Store(db)
    assert store.get_run(out["run_id"])["config"]["source"] == "benchmarks.import-jsonl"
    assert store.get_artifact(out["run_id"], "import_report")["n_scored"] == 4
    assert len(store.get_responses(out["run_id"])) == 4


def test_importer_field_detection(data, tmp_path):
    exs, name, _ = data
    odd = tmp_path / "odd.jsonl"
    odd.write_text(json.dumps({"idx": 0, "answer_text": "7"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answer_text"):
        import_results(odd, name)
    assert import_results(odd, name, text_field="answer_text")["report"]["accuracy"]["extract_lastline"] == 1.0
