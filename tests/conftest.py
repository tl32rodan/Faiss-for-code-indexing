from __future__ import annotations

import numpy as np


class DummyEmbeddingModel:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            values = [float(len(text) % (i + 2)) for i in range(self.dimension)]
            vectors.append(values)
        return np.array(vectors, dtype="float32")


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        for vector in vectors:
            self._vectors.append(np.array(vector, dtype="float32"))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return np.zeros((1, k), dtype="float32"), -np.ones((1, k), dtype="int64")
        matrix = np.stack(self._vectors)
        scores = matrix @ query[0]
        order = np.argsort(scores)[::-1]
        top_indices = order[:k]
        top_scores = scores[top_indices]
        if len(top_indices) < k:
            padding = k - len(top_indices)
            top_indices = np.concatenate([top_indices, -np.ones(padding, dtype=int)])
            top_scores = np.concatenate([top_scores, np.zeros(padding, dtype=float)])
        return top_scores.reshape(1, -1), top_indices.reshape(1, -1)
