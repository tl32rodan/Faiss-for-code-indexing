from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.embeddings import BaseEmbeddingModel
from src.knowledge_store import JSONKnowledgeStore
from src.models import SymbolChunk


def _ensure_2d(values: Any) -> Any:
    if isinstance(values, list):
        if not values:
            return [[]]
        if isinstance(values[0], (int, float)):
            return [values]
    return values


@dataclass
class FaissManager:
    dimension: int
    index: Optional[Any] = None
    docstore: Dict[str, SymbolChunk] = field(default_factory=dict)
    _id_map: List[str] = field(default_factory=list, init=False)
    _positions_by_id: Dict[str, List[int]] = field(default_factory=dict, init=False)
    _inactive_positions: Set[int] = field(default_factory=set, init=False)
    _faiss: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.index is None:
            import faiss

            self._faiss = faiss
            self.index = faiss.IndexFlatIP(self.dimension)

    def add_symbols(
        self, symbols: Iterable[SymbolChunk], embedding_model: BaseEmbeddingModel
    ) -> None:
        symbol_list = list(symbols)
        if not symbol_list:
            return
        embeddings = embedding_model.encode(
            [symbol.get_embedding_content() for symbol in symbol_list]
        )
        vectors = self._prepare_vectors(embeddings)
        self.index.add(vectors)
        start_position = len(self._id_map)
        for offset, symbol in enumerate(symbol_list):
            position = start_position + offset
            self.docstore[symbol.symbol_id] = symbol
            self._id_map.append(symbol.symbol_id)
            self._positions_by_id.setdefault(symbol.symbol_id, []).append(position)

    def deactivate_symbol(self, symbol_id: str) -> None:
        positions = self._positions_by_id.get(symbol_id, [])
        self._inactive_positions.update(positions)

    def search(self, query_vector: Any, top_k: int = 5) -> List[SymbolChunk]:
        vector = self._prepare_vectors(query_vector)
        scores, indices = self.index.search(vector, top_k)
        results: List[SymbolChunk] = []
        seen: Set[str] = set()
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

    def _prepare_vectors(self, values: Any) -> Any:
        vectors = _ensure_2d(values)
        if self._faiss is None:
            return vectors
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required when using FAISS indices") from exc
        return np.array(vectors, dtype="float32")

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

    def index_exists(self, root: str) -> bool:
        path = Path(root)
        if not path.exists():
            return False
        if not (path / "docstore.pkl").exists():
            return False
        return (path / "index.faiss").exists() or (path / "index.pkl").exists()

    def index_from_store(
        self, store: JSONKnowledgeStore, embedding_model: BaseEmbeddingModel
    ) -> None:
        existing_ids = set(self.docstore.keys())
        to_add: List[SymbolChunk] = []
        for symbol in store.iter_symbols():
            if symbol.status == "STALE":
                if symbol.symbol_id in existing_ids:
                    self.deactivate_symbol(symbol.symbol_id)
                continue
            if symbol.symbol_id in existing_ids:
                continue
            to_add.append(symbol)
        if to_add:
            self.add_symbols(to_add, embedding_model)
