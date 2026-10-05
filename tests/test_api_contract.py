"""The API must match docs/API_CONTRACT.md exactly (PLAN.md §11):
1. every (method, path) in the contract's endpoint table exists in the app, and every /api/v1 route is in the table;
2. docs/openapi.json equals the app's OpenAPI document (it was regenerated);
3. the contract version equals the app's api_version."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from dreammachine.api.app import API_VERSION, create_app  # noqa: E402
from dreammachine.store import Store  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
CONTRACT = (DOCS / "API_CONTRACT.md").read_text(encoding="utf-8")


def _table_endpoints() -> set[tuple[str, str]]:
    block = CONTRACT.split("<!-- ENDPOINTS:START -->", 1)[1].split("<!-- ENDPOINTS:END -->", 1)[0]
    rows = re.findall(r"^\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*(\S+)\s*\|", block, re.M)
    return {(m, p) for m, p in rows}


def _app_endpoints() -> set[tuple[str, str]]:
    spec = create_app(store=Store(":memory:")).openapi()
    return {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops if m in ("get", "post", "put", "patch",
                                                                                    "delete")}


def test_every_contract_endpoint_exists_and_nothing_else():
    table, app = _table_endpoints(), _app_endpoints()
    assert len(table) == 39
    assert table - app == set(), f"in the contract but not in the app: {sorted(table - app)}"
    extra = {e for e in app - table if e[1].startswith("/api/v1")}
    assert extra == set(), f"in the app but not in the contract: {sorted(extra)}"
    assert all(p.startswith("/api/v1") for _, p in app), "every route must live under /api/v1"


def test_openapi_json_is_regenerated():
    on_disk = json.loads((DOCS / "openapi.json").read_text(encoding="utf-8"))
    assert on_disk == create_app(store=Store(":memory:")).openapi(), \
        "docs/openapi.json is stale: run `python -m dreammachine.api.export_openapi`"


def test_contract_version_matches_app():
    m = re.search(r"\*\*Contract version: `([0-9.]+)`\*\*", CONTRACT)
    assert m and m.group(1) == API_VERSION
    assert create_app(store=Store(":memory:")).state.api_version == API_VERSION
