import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dreammachine.api.app import create_app, parse_components  # noqa: E402
from dreammachine.experiments.evaluate import ResponseRecord  # noqa: E402
from dreammachine.store import Store  # noqa: E402


@pytest.fixture
def client_and_store():
    store = Store(":memory:")
    return TestClient(create_app(store=store)), store


def test_health_meta_components(client_and_store):
    c, _ = client_and_store
    assert c.get("/health").json()["status"] == "ok"
    meta = c.get("/meta").json()
    assert "n_carry" in meta["features"] and meta["families"]["warehouse"]["split"] == "heldout"
    comps = c.get("/components").json()
    assert any(x["id"] == "C9" and x["path"].endswith("lltm.py") for x in comps)


def test_generate_then_diagnose(client_and_store):
    c, _ = client_and_store
    items = c.post("/generate", json={"n": 3, "steps": 2, "digits": 2, "seed": 1}).json()
    assert len(items) == 3 and all("####" in it["solution"] for it in items)
    d = c.post("/diagnose", json={"response": items[0]["solution"], "item": items[0]}).json()
    assert d["error_type"] == "CORRECT"
    d = c.post("/diagnose", json={"response": "#### 0", "item": items[0]}).json()
    assert d["error_type"] == "UNVERIFIABLE"


def test_diagnose_gsm8k_format(client_and_store):
    c, _ = client_and_store
    body = {"question": "Natalia sold clips to 48 friends in April and half as many in May. How many in total?",
            "reference_solution": "48/2 = <<48/2=24>>24\n48+24 = <<48+24=72>>72\n#### 72",
            "response": "48 / 2 = 24\n48 + 24 = 70\n#### 70"}
    d = c.post("/diagnose", json=body).json()
    assert d["error_type"] == "ARITHMETIC_SLIP" and d["slip_index"] == 1


def test_validation_errors(client_and_store):
    c, _ = client_and_store
    assert c.post("/generate", json={"split": "heldout", "family": "bakery"}).status_code == 422
    assert c.post("/generate", json={"steps": 0}).status_code == 422
    assert c.post("/diagnose", json={"response": "x"}).status_code == 422
    assert c.post("/lltm/fit", json={"Q": [[1.0]], "successes": [2], "trials": [1]}).status_code == 422


def test_lltm_fit_endpoint(client_and_store):
    c, _ = client_and_store
    Q = [[s, 1.0] for s in range(1, 9) for _ in range(10)]
    succ = [1.0 if (i % 10) < 10 - q[0] else 0.0 for i, q in enumerate(Q)]
    r = c.post("/lltm/fit", json={"Q": Q, "successes": succ, "feature_names": ["steps", "const"]}).json()
    assert r["eta"][0] > 0 and r["eta"][1] is None and r["dropped"] == ["const"]


def test_runs_endpoints(client_and_store):
    c, store = client_and_store
    run = store.create_run("x", "eval", {"arm": "base"})
    store.add_responses(run, [ResponseRecord("e1", "gsm8k", 0, "#### 1", True, "CORRECT", {"a": 1}, {})])
    store.put_artifact(run, "lltm", {"eta": [float("nan"), 1.0]})
    store.finish_run(run, metrics={"accuracy": 1.0})
    assert c.get("/runs", params={"kind": "eval"}).json()[0]["id"] == run
    assert c.get(f"/runs/{run}").json()["artifacts"] == ["lltm"]
    assert c.get(f"/runs/{run}/responses").json()[0]["correct"] is True
    assert c.get(f"/runs/{run}/artifacts/lltm").json() == {"eta": [None, 1.0]}
    assert c.get("/runs/nope").status_code == 404
    assert c.get(f"/runs/{run}/artifacts/nope").status_code == 404


def test_parse_components_missing_file(tmp_path):
    assert parse_components(tmp_path / "none.md") == []
