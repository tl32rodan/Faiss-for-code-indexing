from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DummyEmbeddingModel:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            seed = sum(ord(char) for char in text)
            vector = [float((seed + idx) % 10) for idx in range(self._dimension)]
            embeddings.append(vector)
        return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors: list[list[float]] = []

    def add(self, vectors: list[list[float]]) -> None:
        self.vectors.extend(vectors)

    def search(
        self, query_vector: list[list[float]], top_k: int
    ) -> tuple[list[list[float]], list[list[int]]]:
        if not self.vectors:
            return [[0.0] * top_k], [[-1] * top_k]
        query = query_vector[0]
        scores = [self._dot(query, vector) for vector in self.vectors]
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        selected = ranked[:top_k]
        score_row = [scores[idx] for idx in selected]
        index_row = selected.copy()
        while len(score_row) < top_k:
            score_row.append(0.0)
            index_row.append(-1)
        return [score_row], [index_row]

    def _dot(self, left: list[float], right: list[float]) -> float:
        return sum(
            left_value * right_value for left_value, right_value in zip(left, right)
        )
