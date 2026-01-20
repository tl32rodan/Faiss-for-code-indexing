from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, text: List[str], **kwargs: object) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError
