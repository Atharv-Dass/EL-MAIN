"""Write the API's OpenAPI document (the machine-readable copy of docs/API_CONTRACT.md).

    python -m dreammachine.api.export_openapi            # -> docs/openapi.json

Run it after every route or response-shape change; tests/test_api_contract.py fails until the file is current.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..store.db import Store
from .app import REPO_ROOT, create_app


def openapi_document() -> dict:
    return create_app(store=Store(":memory:")).openapi()


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(prog="python -m dreammachine.api.export_openapi")
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "openapi.json"))
    args = p.parse_args(argv)
    out = Path(args.out)
    out.write_text(json.dumps(openapi_document(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
