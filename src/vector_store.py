from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models import CodeChunk


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


@dataclass(slots=True)
class FaissManager:
    dimension: int
    index: Any | None = None
    docstore: dict[str, CodeChunk] = field(default_factory=dict)
    _id_map: list[str] = field(default_factory=list, init=False)
    _positions_by_id: dict[str, list[int]] = field(default_factory=dict, init=False)
    _inactive_positions: set[int] = field(default_factory=set, init=False)
    _faiss: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.index is None:
            import faiss

            self._faiss = faiss
            self.index = faiss.IndexFlatIP(self.dimension)

    def add_chunks(self, chunks: Iterable[CodeChunk], embedding_model: Any) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        embeddings = embedding_model.encode(
            [chunk.get_embedding_content() for chunk in chunk_list]
        )
        vectors = _ensure_2d(np.array(embeddings, dtype="float32"))
        self.index.add(vectors)
        start_position = len(self._id_map)
        for offset, chunk in enumerate(chunk_list):
            position = start_position + offset
            self.docstore[chunk.id] = chunk
            self._id_map.append(chunk.id)
            self._positions_by_id.setdefault(chunk.id, []).append(position)

    def deactivate_chunk(self, chunk_id: str) -> None:
        positions = self._positions_by_id.get(chunk_id, [])
        self._inactive_positions.update(positions)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[CodeChunk]:
        vector = _ensure_2d(np.array(query_vector, dtype="float32"))
        scores, indices = self.index.search(vector, top_k)
        results: list[CodeChunk] = []
        seen: set[str] = set()
        for idx in indices[0]:
            if idx < 0 or idx in self._inactive_positions:
                continue
            chunk_id = self._id_map[idx]
            if chunk_id in seen:
                continue
            chunk = self.docstore.get(chunk_id)
            if chunk is None:
                continue
            results.append(chunk)
            seen.add(chunk_id)
        return results

    def save_local(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        if self._faiss is not None:
            self._faiss.write_index(self.index, str(root / "index.faiss"))
        else:
            with (root / "index.pkl").open("wb") as handle:
                pickle.dump(self.index, handle)
        with (root / "docstore.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "docstore": self.docstore,
                    "id_map": self._id_map,
                    "positions_by_id": self._positions_by_id,
                    "inactive_positions": self._inactive_positions,
                },
                handle,
            )

    def load_local(self, path: str) -> None:
        root = Path(path)
        if self._faiss is not None and (root / "index.faiss").exists():
            self.index = self._faiss.read_index(str(root / "index.faiss"))
        elif (root / "index.pkl").exists():
            with (root / "index.pkl").open("rb") as handle:
                self.index = pickle.load(handle)
        with (root / "docstore.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        self.docstore = payload["docstore"]
        self._id_map = payload["id_map"]
        self._positions_by_id = payload["positions_by_id"]
        self._inactive_positions = payload["inactive_positions"]
