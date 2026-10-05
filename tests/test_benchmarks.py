"""Benchmark registry, built-ins and plug-ins (PLAN.md §8.1-8.3). No network: GSM sets come from local JSONL."""

import sys

import pytest

from dreammachine import benchmarks as B
from dreammachine.benchmarks import registry
from dreammachine.data import Example, from_items, load_jsonl, save_jsonl
from dreammachine.experiments import pipeline as P
from dreammachine.generator import GenSpec, generate_many
from dreammachine.generator.probes import factorial_probe_set

SMALL = {"steps": [2, 4], "digits": [1, 2], "distractors": [0], "op_profiles": ["addsub", "mixed"]}


def _gsm_file(path, n, source="gsm8k"):
    exs = from_items(generate_many(GenSpec(steps=2, digits=2), n, seed=5))
    save_jsonl([Example(id=f"{source}-{i}", source=source, question=e.question, answer=e.answer,
                        solution=e.solution) for i, e in enumerate(exs)], path)
    return path


@pytest.fixture
def cfg(tmp_path):
    return P.Config(name="t", output_dir=str(tmp_path / "out"),
                    eval_data={"gsm8k_test": str(_gsm_file(tmp_path / "g.jsonl", 12)),
                               "gsm_symbolic_main": str(_gsm_file(tmp_path / "s.jsonl", 7, "gsm_symbolic:main"))},
                    eval={"gsm8k_limit": 5, "gsm_symbolic": ["main"], "probe_per_cell": 1, "grid": SMALL})


def _legacy_eval_sets(cfg):
    """eval_sets as it was before the registry (B1), frozen here to prove the refactor changes nothing."""
    e, sets = cfg.eval, {}
    lim = e.get("gsm8k_limit")
    sets["gsm8k"] = P._examples(cfg, "gsm8k_test", lambda: None)[:lim]
    for v in e.get("gsm_symbolic", []):
        sets[f"gsm_symbolic:{v}"] = P._examples(cfg, f"gsm_symbolic_{v}", lambda: None)
    per = e.get("probe_per_cell", 2)
    sets["probes_train_families"] = from_items(factorial_probe_set(cfg.grid(e), per_cell=per, seed=777))
    sets["probes_heldout_families"] = from_items(
        factorial_probe_set(cfg.grid(e), per_cell=per, seed=778, split="heldout"))
    return sets


def test_builtins_registered_in_order():
    names = [b.name for b in B.available()]
    assert names[:6] == ["gsm8k", "gsm_symbolic:main", "gsm_symbolic:p1", "gsm_symbolic:p2",
                         "probes_train_families", "probes_heldout_families"]
    info = B.get("gsm8k").info()
    assert info == {"name": "gsm8k", "kind": "external", "size": 1319,
                    "description": "GSM8K test set (1319 grade-school word problems)."}
    assert B.get("probes_heldout_families").kind == "synthetic" and B.get("probes_heldout_families").size is None


def test_eval_sets_identical_to_before_registry(cfg):
    new, old = P.eval_sets(cfg), _legacy_eval_sets(cfg)
    assert list(new) == list(old)
    for name in old:
        assert [e.to_dict() for e in new[name]] == [e.to_dict() for e in old[name]], name
    assert len(new["gsm8k"]) == 5


def test_eval_benchmarks_list_and_jsonl(cfg, tmp_path):
    extra = _gsm_file(tmp_path / "mine.jsonl", 3, "mine")
    cfg.eval["benchmarks"] = ["probes_heldout_families", f"jsonl:{extra}"]
    sets = P.eval_sets(cfg)
    assert list(sets) == ["probes_heldout_families", f"jsonl:{extra}"]
    assert [e.id for e in sets[f"jsonl:{extra}"]] == [e.id for e in load_jsonl(extra)]


def test_unknown_and_reserved_names():
    with pytest.raises(KeyError, match="available"):
        B.get("nope")
    with pytest.raises(KeyError):
        B.get("jsonl:")
    with pytest.raises(ValueError, match="already registered"):
        B.register(B.get("gsm8k"))
    with pytest.raises(ValueError, match="reserved"):
        B.register(B.Benchmark("jsonl:x", "d", "external", lambda **k: []))


def test_probe_limit_is_a_seeded_subset():
    b = B.get("probes_train_families")
    full = b.load(grid=P.Config(name="t", output_dir=".").grid({"grid": SMALL}), per_cell=2)
    sub = b.load(limit=5, seed=3, grid=P.Config(name="t", output_dir=".").grid({"grid": SMALL}), per_cell=2)
    again = b.load(limit=5, seed=3, grid=P.Config(name="t", output_dir=".").grid({"grid": SMALL}), per_cell=2)
    ids = [e.id for e in full]
    assert [e.id for e in sub] == [e.id for e in again] and len(sub) == 5
    assert [ids.index(e.id) for e in sub] == sorted(ids.index(e.id) for e in sub)   # original order kept


def test_default_score_is_the_shared_taxonomy():
    ex = from_items(generate_many(GenSpec(steps=2, digits=1), 1, seed=1))[0]
    b = B.get("probes_train_families")
    assert b.score(ex, ex.solution).error_type.value == "CORRECT"
    assert b.score(ex, "no idea").error_type.value == "FORMAT_ERROR"


def test_plugin_file_registers_a_benchmark(tmp_path):
    plugin = tmp_path / "toy_bench.py"
    plugin.write_text(
        "from dreammachine.benchmarks import Benchmark, register\n"
        "from dreammachine.data import Example\n"
        "def _load(limit=None, seed=0):\n"
        "    return [Example(id=f'toy-{i}', source='toy', question=f'What is {i} + 1?', answer=str(i + 1),\n"
        "                    solution=f'{i} + 1 = <<{i}+1={i + 1}>>{i + 1}\\n#### {i + 1}') for i in range(4)][:limit]\n"
        "register(Benchmark('toy', 'A toy plug-in.', 'external', _load, size=4))\n", encoding="utf-8")
    try:
        assert registry.load_plugins([plugin]) == ["dreammachine.benchmarks.contrib.toy_bench"]
        toy = B.get("toy")
        assert len(toy.load(limit=2)) == 2 and toy.score(toy.load()[1], "1 + 1 = 2\n#### 2").error_type.value == "CORRECT"
        assert "toy" in [b.name for b in B.available()]
    finally:
        B.unregister("toy")
        sys.modules.pop("dreammachine.benchmarks.contrib.toy_bench", None)


def test_model_catalog(tmp_path):
    from dreammachine.models.catalog import load_models, model_ids

    ids = model_ids()
    assert {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-1.5B-Instruct"} <= set(ids)   # D12 + friend's model
    for path in ("configs/main.yaml", "configs/smoke.yaml"):
        c = P.Config.load(path)
        assert c.model in ids and set(c.candidates) <= set(ids), path
    gated = {m["id"]: m["gated"] for m in load_models()}
    assert gated["meta-llama/Llama-3.2-1B-Instruct"] and not gated["Qwen/Qwen3-0.6B"]
    dup = tmp_path / "m.yaml"
    dup.write_text("models:\n  - {id: a, display_name: A}\n  - {id: a, display_name: A2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        load_models(dup)


def test_gsm_symbolic_limit_spreads_across_templates(tmp_path):
    # B8: the dataset groups rows by template, so "the first 50" was 50 copies of one problem
    from dreammachine.benchmarks.builtin import spread_templates
    from dreammachine.data.loaders import _from_gsm_rows

    rows = [{"id": t, "instance": k, "original_id": 100 + t, "question": f"q{t}-{k}", "answer": f"#### {t}"}
            for t in range(4) for k in range(5)]
    exs = _from_gsm_rows(rows, "gsm_symbolic:main", "gsmsym-main")
    save_jsonl(exs, tmp_path / "sym.jsonl")
    got = B.get("gsm_symbolic:main").load(limit=6, path=str(tmp_path / "sym.jsonl"))
    assert [e.id for e in got] == ["gsmsym-main-0-0", "gsmsym-main-1-0", "gsmsym-main-2-0", "gsmsym-main-3-0",
                                   "gsmsym-main-0-1", "gsmsym-main-1-1"]
    assert len(spread_templates(exs)) == 20 and sorted(e.id for e in spread_templates(exs)) == sorted(e.id for e in exs)
    gsm = [Example(id=f"g{i}", source="gsm8k", question="q", answer="1") for i in range(3)]
    assert spread_templates(gsm) == gsm                       # no template metadata: order unchanged
    assert B.get("gsm_symbolic:main").version == 2 and B.get("gsm8k").version == 1
