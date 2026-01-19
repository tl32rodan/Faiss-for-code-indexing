from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _compute_chunk_id(filepath: str, content: str) -> str:
    digest = hashlib.sha256()
    digest.update(filepath.encode("utf-8"))
    digest.update(b"\0")
    digest.update(content.encode("utf-8"))
    return digest.hexdigest()


@dataclass(slots=True)
class CodeChunk:
    filepath: str
    content: str
    meta_intent: str = ""
    quality_tier: str = "SILVER"
    start_line: int = 1
    id: str = ""
    def __post_init__(self) -> None:
        if not self.id:
            self.id = _compute_chunk_id(self.filepath, self.content)

    def get_embedding_content(self) -> str:
        return (
            f"# File: {self.filepath}\n"
            f"# Tier: {self.quality_tier}\n"
            f"# Intent: {self.meta_intent}\n"
            f"{self.content}"
        )
