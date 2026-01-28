from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeUnit:
    uid: str
    content: str
    vector: Optional[List[float]] = None
    source_type: str = "code"
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_ids: List[str] = field(default_factory=list)

    def get_embedding_content(self) -> str:
        intent = self.metadata.get("intent", "")
        return f"{intent}\n{self.content}".strip()
