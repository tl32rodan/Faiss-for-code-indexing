from __future__ import annotations

import json
import tempfile
from pathlib import Path

from refine import run_refine


def _load_status(knowledge_root: Path, filename: str, symbol_id: str) -> str:
    payload = json.loads((knowledge_root / filename).read_text(encoding="utf-8"))
    for entry in payload.get("symbols", []):
        if entry["symbol_id"] == symbol_id:
            return entry["status"]
    return "MISSING"


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_root = root / "src"
        knowledge_root = root / "knowledge_base"
        source_root.mkdir()
        module_path = source_root / "pipeline.py"

        module_path.write_text(
            "def process(data: list[int]) -> int:\n    return sum(data)\n",
            encoding="utf-8",
        )
        run_refine(str(source_root), str(knowledge_root))
        status = _load_status(knowledge_root, "pipeline.json", "pipeline:process")
        print(f"After create: {status}")

        module_path.write_text(
            "def process(data: list[int]) -> int:\n    return max(data)\n",
            encoding="utf-8",
        )
        run_refine(str(source_root), str(knowledge_root))
        status = _load_status(knowledge_root, "pipeline.json", "pipeline:process")
        print(f"After change: {status}")


if __name__ == "__main__":
    main()
