from __future__ import annotations

import json
import tempfile
from pathlib import Path

from refine import run_refine


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_root = root / "src"
        knowledge_root = root / "knowledge_base"
        source_root.mkdir()
        (source_root / "demo.py").write_text(
            "def greet(name: str) -> str:\n    return f'Hello, {name}'\n",
            encoding="utf-8",
        )

        updated = run_refine(str(source_root), str(knowledge_root))
        print(f"Refined symbols: {updated}")
        payload = json.loads(
            (knowledge_root / "demo.json").read_text(encoding="utf-8")
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
