from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.knowledge_unit import KnowledgeUnit
from src.embeddings import BaseEmbeddingModel


def _ensure_2d(values: Any) -> Any:
    if isinstance(values, list):
        if not values:
            return [[]]
        if isinstance(values[0], (int, float)):
            return [values]
    return values


class VectorStoreBase(ABC):
    @abstractmethod
    def add(self, units: Iterable[KnowledgeUnit]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, units: Iterable[KnowledgeUnit]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, uids: Iterable[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeUnit]:
        raise NotImplementedError


@dataclass
class IdMapStore:
    mapping_path: Path
    str_to_int: Dict[str, int] = field(default_factory=dict)
    int_to_str: Dict[int, str] = field(default_factory=dict)

    def load(self) -> None:
        if not self.mapping_path.exists():
            return
        payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        self.str_to_int = {key: int(value) for key, value in payload.get("str_to_int", {}).items()}
        self.int_to_str = {int(key): value for key, value in payload.get("int_to_str", {}).items()}

    def save(self) -> None:
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "str_to_int": self.str_to_int,
            "int_to_str": {str(key): value for key, value in self.int_to_str.items()},
        }
        self.mapping_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_or_create(self, uid: str) -> int:
        existing = self.str_to_int.get(uid)
        if existing is not None:
            return existing
        next_id = max(self.int_to_str.keys(), default=-1) + 1
        self.str_to_int[uid] = next_id
        self.int_to_str[next_id] = uid
        return next_id

    def remove(self, uid: str) -> Optional[int]:
        int_id = self.str_to_int.pop(uid, None)
        if int_id is None:
            return None
        self.int_to_str.pop(int_id, None)
        return int_id

    def get_uid(self, int_id: int) -> Optional[str]:
        return self.int_to_str.get(int_id)


@dataclass
class FaissStore(VectorStoreBase):
    dimension: int
    embedding_model: BaseEmbeddingModel
    index: Optional[Any] = None
    id_map: Optional[IdMapStore] = None
    docstore: Dict[str, KnowledgeUnit] = field(default_factory=dict)
    _faiss: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.index is None:
            import faiss

            self._faiss = faiss
            self.index = self._create_index()
        if self.id_map is None:
            self.id_map = IdMapStore(Path("knowledge_base/id_map.json"))

    def add(self, units: Iterable[KnowledgeUnit]) -> None:
        unit_list = list(units)
        if not unit_list:
            return
        embeddings = self.embedding_model.encode(
            [unit.get_embedding_content() for unit in unit_list]
        )
        vectors = self._prepare_vectors(embeddings)
        ids = [self.id_map.get_or_create(unit.uid) for unit in unit_list]
        self.index.add_with_ids(vectors, self._prepare_ids(ids))
        for unit in unit_list:
            self.docstore[unit.uid] = unit
        self.id_map.save()

    def update(self, units: Iterable[KnowledgeUnit]) -> None:
        unit_list = list(units)
        if not unit_list:
            return
        for unit in unit_list:
            self.docstore[unit.uid] = unit
            self.id_map.get_or_create(unit.uid)
        try:
            ids_to_remove = [self.id_map.get_or_create(unit.uid) for unit in unit_list]
            self._remove_ids(ids_to_remove)
            self.add(unit_list)
        except RuntimeError:
            self._rebuild_index()
        self.id_map.save()

    def delete(self, uids: Iterable[str]) -> None:
        ids: List[int] = []
        for uid in uids:
            int_id = self.id_map.remove(uid)
            if int_id is not None:
                ids.append(int_id)
            self.docstore.pop(uid, None)
        if ids:
            try:
                self._remove_ids(ids)
            except RuntimeError:
                self._rebuild_index()
        self.id_map.save()

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeUnit]:
        vector = self._prepare_vectors(_ensure_2d(query_vector))
        fetch_k = max(top_k * 4, top_k)
        scores, indices = self.index.search(vector, fetch_k)
        results: List[KnowledgeUnit] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            uid = self.id_map.get_uid(int(idx))
            if uid is None:
                continue
            unit = self.docstore.get(uid)
            if unit is None:
                continue
            if filter_criteria and not self._matches_filter(unit, filter_criteria):
                continue
            results.append(unit)
            if len(results) >= top_k:
                break
        return results

    def get_by_uid(self, uid: str) -> Optional[KnowledgeUnit]:
        return self.docstore.get(uid)

    def _matches_filter(self, unit: KnowledgeUnit, criteria: Dict[str, Any]) -> bool:
        for key, value in criteria.items():
            if key == "source_type":
                if unit.source_type != value:
                    return False
                continue
            meta_value = unit.metadata.get(key)
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            else:
                if meta_value != value:
                    return False
        return True

    def _create_index(self) -> Any:
        base_index = self._faiss.IndexHNSWFlat(self.dimension, 32)
        return self._faiss.IndexIDMap(base_index)

    def _remove_ids(self, ids: List[int]) -> None:
        if not ids:
            return
        if self._faiss is None:
            selector = ids
        else:
            selector = self._faiss.IDSelectorBatch(self._prepare_ids(ids))
        self.index.remove_ids(selector)

    def _rebuild_index(self) -> None:
        self.index = self._create_index()
        if not self.docstore:
            return
        units = list(self.docstore.values())
        embeddings = self.embedding_model.encode(
            [unit.get_embedding_content() for unit in units]
        )
        vectors = self._prepare_vectors(embeddings)
        ids = [self.id_map.get_or_create(unit.uid) for unit in units]
        self.index.add_with_ids(vectors, self._prepare_ids(ids))

    def _prepare_vectors(self, values: Any) -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required when using FAISS indices") from exc
        return np.array(values, dtype="float32")

    def _prepare_ids(self, values: List[int]) -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required when using FAISS indices") from exc
        return np.array(values, dtype="int64")


@dataclass
class IndexRegistry:
    indices: Dict[str, VectorStoreBase] = field(default_factory=dict)

    def get_index(self, name: str) -> VectorStoreBase:
        if name not in self.indices:
            raise KeyError(f"Index '{name}' is not registered")
        return self.indices[name]

    def register_index(self, name: str, index: VectorStoreBase) -> None:
        self.indices[name] = index
