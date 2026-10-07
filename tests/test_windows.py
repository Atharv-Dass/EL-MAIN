"""Windows hardening (PLAN.md B0).

On native Windows, Python's default text encoding is the ANSI code page (cp1252), so any
`open`/`read_text`/`write_text` without an explicit encoding breaks on characters such as
'Δ' (report.md). These tests force that default on every OS, so the bug reproduces on Linux too.
"""

import ast
import builtins
import io
from pathlib import Path

import pytest

from dreammachine.experiments import pipeline as P
from dreammachine.experiments.evaluate import ResponseRecord
from dreammachine.store import Store

PKG = Path(__file__).resolve().parents[1] / "dreammachine"


@pytest.fixture
def cp1252_default(monkeypatch):
    """Make every text-mode open() without an explicit encoding use cp1252 (the Windows default)."""
    real_open = io.open

    def fake_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr(builtins, "open", fake_open)
    # In UTF-8 mode io.text_encoding(None) returns "utf-8"; force the locale path instead.
    monkeypatch.setattr(io, "text_encoding", lambda enc, stacklevel=2: "locale" if enc is None else enc)


def _eval_run(store: Store, arm: str, ratio, seed, correct: list[bool]) -> None:
    run = store.create_run(f"t:eval:{arm}", "eval", {"arm": arm, "ratio": ratio, "seed": seed})
    store.add_responses(run, [ResponseRecord(f"e{i}", "gsm8k", 0, "#### 1", c, "CORRECT" if c else "PLAN_ERROR",
                                             {}, {}) for i, c in enumerate(correct)])
    store.finish_run(run, metrics={"gsm8k": sum(correct) / len(correct), "ratio": ratio, "seed": seed})


def test_report_md_survives_cp1252_default(tmp_path, cp1252_default):
    cfg = P.Config(name="t", output_dir=str(tmp_path / "out"), data={"main_ratio": 3})
    store = Store(":memory:")
    _eval_run(store, "base", None, None, [True, False, False, True] * 5)
    _eval_run(store, "targeted", 3.0, 0, [True, True, False, True] * 5)
    _eval_run(store, "matched_control", 3.0, 0, [True, False, True, True] * 5)
    P.report(cfg, store)
    text = (tmp_path / "out" / "report.md").read_bytes().decode("utf-8")
    assert "Δ accuracy" in text and "±" in text


def test_config_load_reads_utf8(tmp_path, cp1252_default):
    p = tmp_path / "c.yaml"
    p.write_bytes("name: Δ-run\noutput_dir: out\n".encode("utf-8"))
    assert P.Config.load(p).name == "Δ-run"


def _calls_without_encoding(path: Path) -> list[str]:
    """Text-mode open/read_text/write_text calls that do not pass encoding=..."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name not in ("open", "read_text", "write_text"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        mode = next((a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)), "")
        mode = next((k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)),
                    mode)
        if "b" in mode:
            continue
        bad.append(f"{path.relative_to(PKG.parent)}:{node.lineno}")
    return bad


def test_every_text_file_access_names_its_encoding():
    files = sorted(PKG.rglob("*.py")) + sorted((PKG.parent / "tests").glob("*.py"))
    bad = [hit for f in files for hit in _calls_without_encoding(f)]
    assert not bad, "missing encoding='utf-8': " + ", ".join(bad)
