from __future__ import annotations


class DummyEmbeddingModel:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            values = [float(len(text) % (i + 2)) for i in range(self.dimension)]
            vectors.append(values)
        return vectors


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors: list[list[float]] = []

    def add(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            self._vectors.append([float(value) for value in vector])

    def search(self, query: list[list[float]], k: int) -> tuple[list[list[float]], list[list[int]]]:
        if not self._vectors:
            return [[0.0] * k], [[-1] * k]
        query_vector = query[0]
        scores = [self._dot(vector, query_vector) for vector in self._vectors]
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        top_indices = order[:k]
        top_scores = [scores[idx] for idx in top_indices]
        if len(top_indices) < k:
            padding = k - len(top_indices)
            top_indices.extend([-1] * padding)
            top_scores.extend([0.0] * padding)
        return [top_scores], [top_indices]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))
