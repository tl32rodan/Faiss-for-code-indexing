from __future__ import annotations

import argparse
from pathlib import Path

from src.extractors import get_extractor_for_path
from src.ingest import FileLoader
from src.knowledge_store import JSONKnowledgeStore
from src.refinery import KnowledgeRefinery


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine knowledge base metadata.")
    parser.add_argument("--source-root", default="src", help="Source directory root.")
    parser.add_argument(
        "--knowledge-root", default="knowledge_base", help="Knowledge base root."
    )
    return parser.parse_args()


def run_refine(source_root: str, knowledge_root: str) -> int:
    source_root_path = Path(source_root)
    if not source_root_path.exists():
        return 0
    loader = FileLoader(
        valid_extensions=(".py", ".pl", ".pm", ".tcl", ".csh", ".txt", ".md")
    )
    source_files = loader.scan_directory(str(source_root_path))
    if not source_files:
        return 0
    store = JSONKnowledgeStore(knowledge_root, str(source_root_path))
    updated_total = 0
    for filepath in source_files:
        extractor = get_extractor_for_path(str(source_root_path), filepath)
        refinery = KnowledgeRefinery(extractor, store)
        updated = refinery.refine_file(filepath)
        updated_total += len(updated)
    return updated_total


def main() -> None:
    args = _parse_args()
    run_refine(args.source_root, args.knowledge_root)


if __name__ == "__main__":
    main()
