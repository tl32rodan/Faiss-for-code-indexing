from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.extractors import BaseExtractor
from src.knowledge_store import JSONKnowledgeStore
from src.models import SymbolChunk


def generate_intent(code: str) -> str:
    return "TODO: LLM gen"


class KnowledgeRefinery:
    def __init__(self, extractor: BaseExtractor, store: JSONKnowledgeStore) -> None:
        self.extractor = extractor
        self.store = store

    def refine_file(self, source_path: str) -> List[SymbolChunk]:
        content = Path(source_path).read_text(encoding="utf-8")
        extracted = self.extractor.extract_symbols(source_path, content)
        existing = self.store.load_symbols(source_path)
        reconciled = self._reconcile_symbols(extracted, existing)
        self.store.save_symbols(source_path, reconciled)
        return reconciled

    def refine_files(self, source_paths: Iterable[str]) -> List[SymbolChunk]:
        updated: List[SymbolChunk] = []
        for source_path in source_paths:
            updated.extend(self.refine_file(source_path))
        return updated

    def _reconcile_symbols(
        self, extracted: Iterable[SymbolChunk], existing: dict[str, SymbolChunk]
    ) -> List[SymbolChunk]:
        reconciled: List[SymbolChunk] = []
        for symbol in extracted:
            stored = existing.get(symbol.symbol_id)
            if stored is None:
                symbol.intent = generate_intent(symbol.content)
                symbol.meta_hash = symbol.code_hash
                symbol.status = "NEW"
            else:
                symbol.intent = stored.intent
                symbol.meta_hash = stored.meta_hash
                if stored.meta_hash != symbol.code_hash:
                    symbol.status = "STALE"
                elif stored.status == "NEW":
                    symbol.status = "NEW"
                else:
                    symbol.status = "OK"
            reconciled.append(symbol)
        return reconciled
