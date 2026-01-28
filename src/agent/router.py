from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List


class RouterInterface(ABC):
    @abstractmethod
    def route(self, query: str) -> List[str]:
        raise NotImplementedError


@dataclass
class KeywordRouter(RouterInterface):
    default_indices: List[str] = field(
        default_factory=lambda: ["source_code", "tests", "issues", "knowledge"]
    )
    issue_keywords: Iterable[str] = ("issue", "bug", "ticket", "jira")
    test_keywords: Iterable[str] = ("test", "pytest", "unittest", "spec")
    knowledge_keywords: Iterable[str] = ("doc", "readme", "guide", "how to")

    def route(self, query: str) -> List[str]:
        query_lower = query.lower()
        indices = set()
        if any(keyword in query_lower for keyword in self.issue_keywords):
            indices.add("issues")
        if any(keyword in query_lower for keyword in self.test_keywords):
            indices.add("tests")
        if any(keyword in query_lower for keyword in self.knowledge_keywords):
            indices.add("knowledge")
        if not indices:
            indices.add("source_code")
        return [index for index in self.default_indices if index in indices]
