from __future__ import annotations

from dataclasses import dataclass

from src.knowledge_store import JSONKnowledgeStore


@dataclass
class IntentManager:
    def update_intent(
        self, symbol_id: str, intent_text: str, store: JSONKnowledgeStore, source_path: str
    ) -> None:
        symbols = store.load_symbols(source_path)
        symbol = symbols.get(symbol_id)
        if symbol is None:
            raise KeyError(f"Symbol {symbol_id} not found")
        symbol.intent = intent_text
        symbol.meta_hash = symbol.code_hash
        symbol.status = "OK"
        store.save_symbols(source_path, symbols.values())
