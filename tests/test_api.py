"""API v1 (PLAN.md milestone B7): every endpoint of docs/API_CONTRACT.md against real stored jobs.

A completed explore pipeline and a completed benchmark run are built once per module (simulated model with a
planted weakness + the tiny CPU model for training), then the API is exercised over HTTP with TestClient."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dreammachine.api.app import create_app, parse_components, read_log_chunk  # noqa: E402
from dreammachine.data import Example, from_items, save_jsonl  # noqa: E402
from dreammachine.experiments import pipeline as P  # noqa: E402
from dreammachine.generator import GenSpec, generate_many  # noqa: E402
from dreammachine.jobs import create_job_from_request, run_job  # noqa: E402
from dreammachine.store import Store  # noqa: E402
from test_pipeline import SimulatedRunner  # noqa: E402

BENCH = ["gsm8k", "probes_train_families", "probes_heldout_families"]
SMALL = {"steps": [2, 4, 6], "digits": [1, 2, 3], "distractors": [0, 1], "op_profiles": ["addsub", "mixed"]}
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture(scope="module")
def world(tmp_path_factory, tiny_model_dir):
    tmp = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    oracle: dict[str, Example] = {}
    real_from_items = P.from_items

    def recording(items):
        exs = real_from_items(items)
        oracle.update({e.question: e for e in exs})
        return exs

    from dreammachine.benchmarks import builtin
    mp.setattr(P, "from_items", recording)
    mp.setattr(builtin, "from_items", recording)

    def fake_gsm(n, seed):
        exs = from_items(generate_many(GenSpec(steps=3, digits=2), n, seed=seed))
        out = [Example(id=f"g{seed}-{i}", source="gsm8k", question=e.question, answer=e.answer, solution=e.solution)
               for i, e in enumerate(exs)]
        oracle.update({e.question: e for e in out})
        return out

    save_jsonl(fake_gsm(400, 1), tmp / "gsm_train.jsonl")
    save_jsonl(fake_gsm(40, 2), tmp / "gsm_test.jsonl")

    def preset(**over):
        c = P.Config(
            name="test", output_dir=str(tmp / "unused"), model=str(tiny_model_dir), seeds=[0],
            real_data=str(tmp / "gsm_train.jsonl"), eval_data={"gsm8k_test": str(tmp / "gsm_test.jsonl")},
            diagnose={"probe_per_cell": 6, "n_samples": 3, "bootstrap": 40, "natural_pool": 1500, "annotate": 10,
                      "grid": SMALL},
            data={"pool_size": 6000, "natural_size": 2000, "token_budget": 15000, "ratios": [3], "main_ratio": 3,
                  "grid": {"steps": [2, 3, 4, 5, 6], "digits": [1, 2, 3, 4]}},
            train={"load_in_4bit": False, "max_steps": 2, "per_device_batch_size": 2, "grad_accum": 1,
                   "target_modules": ["q_proj", "v_proj"], "save_steps": 1000, "logging_steps": 1},
            eval={"benchmark_limit": 20, "gsm_symbolic": [], "probe_per_cell": 1, "grid": SMALL,
                  "load_in_4bit": False},
            preflight={"min_free_gb": 0.01}, gpu={"required": False, "memory_fraction": 0})
        for k, v in over.items():
            setattr(c, k, v)
        return c

    presets = {"explore": preset(),
               "paper": preset(seeds=[0, 1, 2], data={**preset().data, "ratios": [1, 3, 9]})}
    store = Store(tmp / "dm.db")
    factory = lambda model, adapter, gen: SimulatedRunner(oracle, seed=len(adapter or ""), n_samples=gen.n_samples)
    common = dict(presets=presets, known_models=[], runs_root=tmp / "pipelines", allow_local_models=True)
    pl = create_job_from_request(store, "pipeline", {"mode": "explore", "model": str(tiny_model_dir),
                                                     "benchmarks": BENCH, "name": "e2e"}, **common)
    assert run_job(store, pl, factory, log=lambda s: None) == "done"
    br = create_job_from_request(store, "benchmark_run", {"model": str(tiny_model_dir), "benchmarks": BENCH,
                                                          "name": "bench"}, **common)
    assert run_job(store, br, factory, log=lambda s: None) == "done"
    (Path(store.get_job(pl)["output_dir"]) / "logs").mkdir(exist_ok=True)
    # raw bytes: the log endpoint returns the file exactly (write_text would turn \n into \r\n on Windows)
    (Path(store.get_job(pl)["output_dir"]) / "logs" / "job.log").write_bytes(("step 0 ok — Δ ±\n" * 3).encode("utf-8"))
    client = TestClient(create_app(store=store, **common))
    yield {"client": client, "store": store, "pl": pl, "br": br, "model": str(tiny_model_dir)}
    mp.undo()


def _ok(r, status=200):
    assert r.status_code == status, r.text
    return r.json()


def _err(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and "message" in body["error"]
    return body["error"]


# ------------------------------------------------------------ system and catalog
def test_health_and_unknown_route(world):
    c = world["client"]
    assert _ok(c.get("/api/v1/health")) == {"status": "ok", "version": "0.1.0", "api_version": "0.2.0"}
    _err(c.get("/api/v1/nope"), 404, "not_found")
    _err(c.get("/health"), 404, "not_found")                    # the old unprefixed routes are gone


def test_system(world):
    s = _ok(world["client"].get("/api/v1/system"))
    assert set(s) == {"gpu", "worker", "versions", "platform", "disk"}
    assert s["worker"]["alive"] is False and s["disk"]["free_gb"] > 0 and s["disk"]["min_free_gb"] == 0.01
    assert s["versions"]["python"].count(".") == 2 and isinstance(s["gpu"]["available"], bool)


def test_meta(world):
    m = _ok(world["client"].get("/api/v1/meta"))
    assert len(m["features"]) == 7 and len(m["error_types"]) == 6 and len(m["stages"]) == 8
    assert [a["name"] for a in m["arms"]] == ["base", "real_only", "untargeted", "matched_control", "targeted"]
    carry = next(f for f in m["features"] if f["name"] == "n_carry")
    assert carry["label"] == "Carries and borrows" and m["families"]["warehouse"]["split"] == "heldout"
    assert {"name": "diagnose", "label": "Find weaknesses"} in m["stages"]


def test_models_benchmarks_presets_components(world):
    c = world["client"]
    ids = [m["id"] for m in _ok(c.get("/api/v1/models"))]
    assert "Qwen/Qwen3-0.6B" in ids and "Qwen/Qwen3-1.7B" in ids
    b = _ok(c.get("/api/v1/benchmarks"))
    assert {"name": "gsm8k", "kind": "external", "size": 1319,
            "description": "GSM8K test set (1319 grade-school word problems)."} in b
    p = _ok(c.get("/api/v1/presets"))
    assert p["explore"]["defaults"]["arms"] == ["targeted", "matched_control"] and p["explore"]["defaults"]["ratio"] == 3
    assert p["explore"]["allowed"] == {"ratio": [1, 3, 9], "seeds": [0, 1, 2], "max_seeds": 3, "min_benchmark_limit": 10}
    assert p["paper"]["n_steps"] == 42 and p["paper"]["settable"] == ["name", "model"]
    comps = _ok(c.get("/api/v1/components"))
    assert any(x["id"] == "C9" and x["path"].endswith("lltm.py") for x in comps)


def test_parse_components_missing_file(tmp_path):
    assert parse_components(tmp_path / "none.md") == []


# ------------------------------------------------------------------- pipelines
def test_create_pipeline_and_queue(world):
    c = world["client"]
    p = _ok(c.post("/api/v1/pipelines", json={"mode": "explore", "model": world["model"], "benchmarks": BENCH,
                                              "name": "queued-one"}), 202)
    assert p["status"] == "queued" and p["kind"] == "pipeline" and "worker_not_running" in p["warnings"]
    assert p["queue_position"] >= 1 and p["target"] is None and p["resolved"]["ratios"] == [3.0]
    assert ISO.match(p["created_at"]) and p["started_at"] is None and len(p["steps"]) == 10
    q = _ok(c.get("/api/v1/queue"))
    assert q["current"] is None and any(x["job_id"] == p["id"] for x in q["queued"])
    _ok(c.post(f"/api/v1/pipelines/{p['id']}/cancel"))         # keep the queue clean for other tests


def test_create_pipeline_errors(world):
    c = world["client"]
    _err(c.post("/api/v1/pipelines", json={"mode": "paper", "model": world["model"], "seeds": [0]}), 422,
         "paper_mode_locked")
    _err(c.post("/api/v1/pipelines", json={"mode": "explore", "model": "nope/model"}), 422, "unknown_model")
    _err(c.post("/api/v1/pipelines", json={"mode": "explore", "model": world["model"], "benchmarks": ["x"]}), 422,
         "unknown_benchmark")
    _err(c.post("/api/v1/pipelines", json={"mode": "explore", "model": world["model"],
                                           "target": {"strategy": "feature", "feature": "colour"}}), 422,
         "invalid_feature")
    e = _err(c.post("/api/v1/pipelines", json=[1, 2]), 422, "validation_error")
    assert e["details"]["fields"]
    _err(c.post("/api/v1/pipelines", json={"mode": "explore", "model": world["model"], "reuse_from": "nope"}), 422,
         "reuse_mismatch")


def test_list_and_get_pipeline(world):
    c, pl = world["client"], world["pl"]
    page = _ok(c.get("/api/v1/pipelines", params={"limit": 50}))
    assert page["total"] >= 1 and page["limit"] == 50 and any(x["id"] == pl for x in page["items"])
    item = next(x for x in page["items"] if x["id"] == pl)
    assert item["model"] == world["model"] and item["paper_eligible"] is False
    assert _ok(c.get("/api/v1/pipelines", params={"mode": "paper"}))["items"] == []
    _err(c.get("/api/v1/pipelines", params={"limit": 0}), 422, "validation_error")
    p = _ok(c.get(f"/api/v1/pipelines/{pl}"))
    assert p["status"] == "done" and p["progress"] == {"done_steps": 10, "total_steps": 10, "fraction": 1.0}
    assert p["target"]["feature"] == "n_carry" and p["paper_eligible"] is False and p["eta_s"] is None
    assert p["output_bytes"] > 0 and p["current_step"] is None and p["error"] is None
    tr = next(s for s in p["steps"] if s["key"] == "train:targeted_r3_s0")
    assert tr["label"] == "Train: targeted, ratio 3, seed 0" and tr["progress"]["unit"] == "steps"
    assert tr["duration_s"] > 0 and ISO.match(tr["finished_at"]) and set(tr["metrics"]) == {"load_s", "run_s", "gpu"}
    _err(c.get(f"/api/v1/pipelines/{world['br']}"), 404, "not_found")      # a benchmark-run id on a pipeline path


def test_cancel_resume_delete_rules(world):
    c, store = world["client"], world["store"]
    p = _ok(c.post("/api/v1/pipelines", json={"mode": "explore", "model": world["model"], "benchmarks": BENCH}), 202)
    assert _ok(c.post(f"/api/v1/pipelines/{p['id']}/cancel"))["status"] == "cancelled"
    _err(c.post(f"/api/v1/pipelines/{p['id']}/cancel"), 409, "conflict")
    assert _ok(c.post(f"/api/v1/pipelines/{p['id']}/resume"))["status"] == "queued"
    _err(c.post(f"/api/v1/pipelines/{p['id']}/resume"), 409, "conflict")
    _err(c.delete(f"/api/v1/pipelines/{p['id']}"), 409, "conflict")         # still queued
    _ok(c.post(f"/api/v1/pipelines/{p['id']}/cancel"))
    assert _ok(c.delete(f"/api/v1/pipelines/{p['id']}")) == {"deleted": True, "freed_bytes": 0}
    _err(c.get(f"/api/v1/pipelines/{p['id']}"), 404, "not_found")
    paper = _ok(c.post("/api/v1/pipelines", json={"mode": "paper", "model": world["model"]}), 202)
    _ok(c.post(f"/api/v1/pipelines/{paper['id']}/cancel"))
    _err(c.delete(f"/api/v1/pipelines/{paper['id']}"), 409, "paper_protected")
    assert len(paper["steps"]) == 42


def test_baseline_and_diagnosis(world):
    c, pl = world["client"], world["pl"]
    b = _ok(c.get(f"/api/v1/pipelines/{pl}/baseline"))
    assert b["model"] == world["model"] and b["reused_from"] is None and len(b["run_ids"]) == 1
    g = next(x for x in b["benchmarks"] if x["name"] == "gsm8k")
    assert g["kind"] == "external" and g["n"] == 20 and 0 <= g["accuracy"] <= 1
    assert not g["error_distribution"] or abs(sum(g["error_distribution"].values()) - 1) < 1e-9
    d = _ok(c.get(f"/api/v1/pipelines/{pl}/diagnosis"))
    assert d["weaknesses"][0]["rank"] == 1 and d["weaknesses"][0]["feature"] == "n_carry"
    assert d["weaknesses"][0]["label"] == "Carries and borrows" and d["fit"]["n_items"] > 0
    assert d["target"]["feature"] == "n_carry" and d["target"]["significant"] is True
    queued = _ok(c.post("/api/v1/pipelines", json={"mode": "explore", "model": world["model"], "benchmarks": BENCH}), 202)
    _err(c.get(f"/api/v1/pipelines/{queued['id']}/baseline"), 409, "not_ready")
    _err(c.get(f"/api/v1/pipelines/{queued['id']}/diagnosis"), 409, "not_ready")
    _err(c.get(f"/api/v1/pipelines/{queued['id']}/data"), 409, "not_ready")
    _err(c.get(f"/api/v1/pipelines/{queued['id']}/results"), 409, "not_ready")
    _err(c.get(f"/api/v1/pipelines/{queued['id']}/report"), 409, "not_ready")
    _ok(c.post(f"/api/v1/pipelines/{queued['id']}/cancel"))


def test_data_samples_download(world):
    c, pl = world["client"], world["pl"]
    d = _ok(c.get(f"/api/v1/pipelines/{pl}/data"))
    assert d["target"] == "n_carry" and {a["key"] for a in d["arms"]} == {"targeted_r3_s0", "matched_control_r3_s0"}
    a = next(x for x in d["arms"] if x["key"] == "targeted_r3_s0")
    assert (a["arm"], a["ratio"], a["seed"]) == ("targeted", 3.0, 0) and a["n"] == a["n_synthetic"] + a["n_real"]
    assert d["matching"][0]["seed"] == 0 and d["matching"][0]["n_requested"] > 0
    syn = _ok(c.get(f"/api/v1/pipelines/{pl}/data/targeted_r3_s0/samples", params={"limit": 2, "source": "synthetic"}))
    assert syn["total"] == a["n_synthetic"] and len(syn["items"]) == 2 and syn["limit"] == 2
    assert set(syn["items"][0]["features"]) >= {"n_carry", "steps"} and syn["items"][0]["source"].startswith("synthetic")
    real = _ok(c.get(f"/api/v1/pipelines/{pl}/data/targeted_r3_s0/samples", params={"source": "real"}))
    assert real["total"] == a["n_real"] and real["items"][0]["features"] is None
    _err(c.get(f"/api/v1/pipelines/{pl}/data/targeted_r9_s0/samples"), 404, "not_found")
    _err(c.get(f"/api/v1/pipelines/{pl}/data/..%2F..%2Fdm/samples"), 404, "not_found")
    _err(c.get(f"/api/v1/pipelines/{pl}/data/targeted_r3_s0/samples", params={"limit": 51}), 422, "validation_error")
    r = c.get(f"/api/v1/pipelines/{pl}/data/targeted_r3_s0/download")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    assert "attachment" in r.headers["content-disposition"] and len(r.text.strip().splitlines()) == a["n"]


def test_training_results_report_log(world):
    c, pl = world["client"], world["pl"]
    tr = _ok(c.get(f"/api/v1/pipelines/{pl}/training"))
    assert {t["key"] for t in tr} == {"targeted_r3_s0", "matched_control_r3_s0"}
    t = tr[0]
    assert t["status"] == "done" and t["step"] == t["total_steps"] == 2 and len(t["loss_curve"]) == 2
    assert t["final_loss"] == t["loss_curve"][-1]["loss"] and t["duration_s"] > 0
    res = _ok(c.get(f"/api/v1/pipelines/{pl}/results"))
    assert res["complete"] is True and res["primary"]["status"] == "proposed" and res["paper_eligible"] is False
    rep = c.get(f"/api/v1/pipelines/{pl}/report")
    assert rep.status_code == 200 and rep.headers["content-type"].startswith("text/markdown")
    assert rep.text.startswith("# DreamMachine results") and "EXPLORE — not paper evidence" in rep.text
    lg = _ok(c.get(f"/api/v1/pipelines/{pl}/log", params={"offset": 0}))
    assert lg["text"] == "step 0 ok — Δ ±\n" * 3 and lg["eof"] is True
    assert lg["next_offset"] == len(("step 0 ok — Δ ±\n" * 3).encode("utf-8"))
    assert _ok(c.get(f"/api/v1/pipelines/{pl}/log", params={"offset": lg["next_offset"]})) == {
        "text": "", "next_offset": lg["next_offset"], "eof": True}


def test_log_chunks_never_split_a_character(tmp_path):
    text = "àé—Δ±✓ log line 💡\n" * 50
    p = tmp_path / "job.log"
    p.write_bytes(text.encode("utf-8"))
    out, off, rounds = "", 0, 0
    while True:
        ch = read_log_chunk(p, off, max_bytes=7)
        assert "�" not in ch["text"]
        out, off, rounds = out + ch["text"], ch["next_offset"], rounds + 1
        if ch["eof"]:
            break
    assert out == text and rounds > 50
    assert read_log_chunk(tmp_path / "missing.log", 0) == {"text": "", "next_offset": 0, "eof": True}


# -------------------------------------------------------------- benchmark runs
def test_benchmark_runs(world):
    c, br = world["client"], world["br"]
    new = _ok(c.post("/api/v1/benchmark-runs", json={"model": world["model"], "benchmarks": ["gsm8k"],
                                                     "include_diagnosis": False}), 202)
    assert new["kind"] == "benchmark_run" and new["mode"] == "explore" and "worker_not_running" in new["warnings"]
    assert [s["key"] for s in new["steps"]] == ["preflight", "baseline"] and "target" not in new
    _err(c.post("/api/v1/benchmark-runs", json={"model": world["model"], "arms": ["targeted"]}), 422,
         "validation_error")
    page = _ok(c.get("/api/v1/benchmark-runs"))
    assert any(x["id"] == br for x in page["items"]) and "paper_eligible" not in page["items"][0]
    run = _ok(c.get(f"/api/v1/benchmark-runs/{br}"))
    assert run["status"] == "done" and run["resolved"]["arms"] == []
    assert _ok(c.post(f"/api/v1/benchmark-runs/{new['id']}/cancel"))["status"] == "cancelled"
    assert _ok(c.delete(f"/api/v1/benchmark-runs/{new['id']}"))["deleted"] is True
    assert len(_ok(c.get(f"/api/v1/benchmark-runs/{br}/baseline"))["benchmarks"]) == 3
    assert _ok(c.get(f"/api/v1/benchmark-runs/{br}/diagnosis"))["target"] is None   # no choose_target in a benchmark run
    assert _ok(c.get(f"/api/v1/benchmark-runs/{br}/log"))["eof"] is True
    _err(c.get(f"/api/v1/benchmark-runs/{world['pl']}"), 404, "not_found")   # a pipeline id on a benchmark-run path


# ----------------------------------------------------------- runs and compare
def test_runs_responses_artifacts_compare(world):
    c, store, pl = world["client"], world["store"], world["pl"]
    page = _ok(c.get("/api/v1/runs", params={"kind": "eval", "job_id": pl}))
    assert page["total"] == 3 and all(r["kind"] == "eval" and r["job_id"] == pl for r in page["items"])
    assert all(ISO.match(r["created_at"]) and r["mode"] == "explore" for r in page["items"])
    base_id = _ok(c.get(f"/api/v1/pipelines/{pl}/baseline"))["run_ids"][0]
    detail = _ok(c.get(f"/api/v1/runs/{base_id}"))
    assert "provenance" in detail["artifacts"] and "lltm" in detail["artifacts"]
    resp = _ok(c.get(f"/api/v1/runs/{base_id}/responses", params={"correct": False, "source": "gsm8k", "limit": 3}))
    assert resp["total"] == store.count_responses(base_id, correct=False, source="gsm8k") and len(resp["items"]) <= 3
    r0 = resp["items"][0]
    assert r0["correct"] is False and r0["source"] == "gsm8k" and r0["features"] == {}
    assert set(r0["diagnosis"]) == {"error_type", "predicted", "gold", "n_equations", "n_slips", "slip_index",
                                    "slip_ops", "slip_operand_digits", "divergence_step", "progress"}
    et = r0["error_type"]
    assert all(x["error_type"] == et for x in _ok(c.get(f"/api/v1/runs/{base_id}/responses",
                                                        params={"error_type": et}))["items"])
    _err(c.get(f"/api/v1/runs/{base_id}/responses", params={"error_type": "OOPS"}), 422, "validation_error")
    prov = _ok(c.get(f"/api/v1/runs/{base_id}/artifacts/provenance"))
    assert prov["mode"] == "explore" and "git" in prov
    _err(c.get(f"/api/v1/runs/{base_id}/artifacts/nope"), 404, "not_found")
    _err(c.get("/api/v1/runs/nope"), 404, "not_found")
    after = next(r["id"] for r in page["items"] if r["id"] != base_id)
    cmp_ = _ok(c.get("/api/v1/compare", params={"before": base_id, "after": after}))
    assert cmp_["before"] == base_id and {b["name"] for b in cmp_["benchmarks"]} == set(BENCH)
    diag = _ok(c.get("/api/v1/runs", params={"kind": "diagnose", "job_id": pl}))["items"][0]["id"]
    _err(c.get("/api/v1/compare", params={"before": base_id, "after": diag}), 422, "validation_error")
    _err(c.get("/api/v1/compare", params={"before": base_id, "after": "nope"}), 404, "not_found")


# ---------------------------------------------------------------------- tools
def test_tools(world):
    c = world["client"]
    items = _ok(c.post("/api/v1/tools/generate", json={"n": 3, "steps": 2, "digits": 2, "seed": 1}))
    assert len(items) == 3 and all("####" in it["solution"] for it in items)
    d = _ok(c.post("/api/v1/tools/classify", json={"response": items[0]["solution"], "item": items[0]}))
    assert d["error_type"] == "CORRECT"
    d = _ok(c.post("/api/v1/tools/classify", json={
        "question": "Natalia sold clips to 48 friends in April and half as many in May. How many in total?",
        "reference_solution": "48/2 = <<48/2=24>>24\n48+24 = <<48+24=72>>72\n#### 72",
        "response": "48 / 2 = 24\n48 + 24 = 70\n#### 70"}))
    assert d["error_type"] == "ARITHMETIC_SLIP" and d["slip_index"] == 1
    Q = [[s, 1.0] for s in range(1, 9) for _ in range(10)]
    succ = [1.0 if (i % 10) < 10 - q[0] else 0.0 for i, q in enumerate(Q)]
    r = _ok(c.post("/api/v1/tools/lltm-fit", json={"Q": Q, "successes": succ, "feature_names": ["steps", "const"]}))
    assert r["eta"][0] > 0 and r["eta"][1] is None and r["dropped"] == ["const"]
    _err(c.post("/api/v1/tools/generate", json={"split": "heldout", "family": "bakery"}), 422, "validation_error")
    _err(c.post("/api/v1/tools/generate", json={"steps": 0}), 422, "validation_error")
    _err(c.post("/api/v1/tools/classify", json={"response": "x"}), 422, "validation_error")
    _err(c.post("/api/v1/tools/lltm-fit", json={"Q": [[1.0]], "successes": [2], "trials": [1]}), 422,
         "validation_error")


def test_cors(world):
    c = world["client"]
    ok = c.options("/api/v1/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = c.options("/api/v1/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers
