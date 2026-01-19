from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.vector_store import FaissManager


@dataclass
class IntentManager:
    save_path: Optional[str] = None

    def update_intent(
        self,
        chunk_id: str,
        intent_text: str,
        vector_db: FaissManager,
        embedding_model: Any,
    ) -> None:
        chunk = vector_db.docstore.get(chunk_id)
        if chunk is None:
            raise KeyError(f"Chunk {chunk_id} not found")
        chunk.meta_intent = intent_text
        vector_db.deactivate_chunk(chunk_id)
        vector_db.add_chunks([chunk], embedding_model)
        if self.save_path:
            vector_db.save_local(self.save_path)
