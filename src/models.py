from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict


def compute_code_hash(content: str) -> str:
    digest = hashlib.sha256()
    digest.update(content.encode("utf-8"))
    return digest.hexdigest()


@dataclass
class SymbolChunk:
    symbol_id: str
    filepath: str
    symbol_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    content: str
    code_hash: str
    meta_hash: str = ""
    intent: str = ""
    status: str = "OK"

    def get_embedding_content(self) -> str:
        return (
            f"# Symbol: {self.symbol_id}\n"
            f"# File: {self.filepath}\n"
            f"# Kind: {self.symbol_kind}\n"
            f"# Intent: {self.intent}\n"
            f"{self.content}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> SymbolChunk:
        return cls(**payload)
