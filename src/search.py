from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.models import SymbolChunk
from src.vector_store import FaissManager


def _format_chunks(chunks: Iterable[SymbolChunk]) -> str:
    sections = []
    for chunk in chunks:
        sections.append(
            "\n".join(
                [
                    f"Symbol: {chunk.symbol_id}",
                    f"File: {chunk.filepath}",
                    f"Kind: {chunk.symbol_kind}",
                    f"Intent: {chunk.intent}",
                    f"Status: {chunk.status}",
                    f"Start line: {chunk.start_line}",
                    "Code:",
                    chunk.content,
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


@dataclass
class CodeSearchEngine:
    vector_db: FaissManager
    embedding_model: Any

    def query(self, user_question: str, filters: Optional[Dict[str, Any]] = None) -> str:
        query_vector = self.embedding_model.encode([user_question])
        results = self.vector_db.search(query_vector, top_k=5)
        return _format_chunks(results)
