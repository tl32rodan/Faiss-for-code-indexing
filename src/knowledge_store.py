from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator

from src.models import SymbolChunk


class JSONKnowledgeStore:
    def __init__(self, knowledge_root: str, source_root: str) -> None:
        self.knowledge_root = Path(knowledge_root)
        self.source_root = Path(source_root)

    def load_symbols(self, source_path: str) -> Dict[str, SymbolChunk]:
        knowledge_path = self._knowledge_path_for_source(source_path)
        if not knowledge_path.exists():
            return {}
        payload = json.loads(knowledge_path.read_text(encoding="utf-8"))
        symbols = {}
        for entry in payload.get("symbols", []):
            chunk = SymbolChunk.from_dict(entry)
            symbols[chunk.symbol_id] = chunk
        return symbols

    def save_symbols(self, source_path: str, symbols: Iterable[SymbolChunk]) -> None:
        knowledge_path = self._knowledge_path_for_source(source_path)
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_path": str(source_path),
            "symbols": [symbol.to_dict() for symbol in symbols],
        }
        knowledge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def iter_symbols(self) -> Iterator[SymbolChunk]:
        if not self.knowledge_root.exists():
            return iter(())
        for path in self.knowledge_root.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload.get("symbols", []):
                yield SymbolChunk.from_dict(entry)

    def _knowledge_path_for_source(self, source_path: str) -> Path:
        rel_path = self._relative_source_path(source_path)
        return self.knowledge_root / rel_path.with_suffix(".json")

    def _relative_source_path(self, source_path: str) -> Path:
        source_path_obj = Path(source_path)
        try:
            return source_path_obj.relative_to(self.source_root)
        except ValueError:
            resolved_root = self.source_root.resolve()
            resolved_source = source_path_obj.resolve()
            try:
                return resolved_source.relative_to(resolved_root)
            except ValueError:
                rel_path = source_path_obj
                if rel_path.is_absolute():
                    parts = [part for part in rel_path.parts if part != rel_path.anchor]
                    rel_path = Path(*parts)
                return rel_path
