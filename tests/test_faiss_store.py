import tempfile
import unittest
from pathlib import Path

from src.core.knowledge_unit import KnowledgeUnit
from src.stores.faiss_store import FaissStore, IdMapStore, IndexRegistry
from tests.conftest import DummyEmbeddingModel


class TestFaissStore(unittest.TestCase):
    def test_add_search_update_delete(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "map.json"
            store = FaissStore(
                dimension=4,
                embedding_model=embedding_model,
                id_map=IdMapStore(mapping_path),
            )
            unit = KnowledgeUnit(uid="code::alpha", content="alpha", metadata={"intent": ""})
            store.add([unit])
            query_vector = embedding_model.encode(["alpha"])[0]
            results = store.search(query_vector, top_k=1)

            self.assertEqual(results[0].uid, "code::alpha")

            updated = KnowledgeUnit(uid="code::alpha", content="beta", metadata={"intent": ""})
            store.update([updated])
            query_vector = embedding_model.encode(["beta"])[0]
            results = store.search(query_vector, top_k=1)
            self.assertEqual(results[0].content, "beta")

            store.delete(["code::alpha"])
            results = store.search(query_vector, top_k=1)
            self.assertEqual(results, [])

    def test_filtering_and_registry(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "map.json"
            store = FaissStore(
                dimension=4,
                embedding_model=embedding_model,
                id_map=IdMapStore(mapping_path),
            )
            store.add(
                [
                    KnowledgeUnit(
                        uid="code::alpha",
                        content="alpha",
                        source_type="code",
                        metadata={"language": "python", "intent": ""},
                    ),
                    KnowledgeUnit(
                        uid="test::alpha",
                        content="alpha test",
                        source_type="tests",
                        metadata={"language": "python", "intent": ""},
                    ),
                ]
            )
            query_vector = embedding_model.encode(["alpha"])[0]
            results = store.search(
                query_vector,
                top_k=2,
                filter_criteria={"source_type": "code"},
            )

            self.assertEqual([unit.uid for unit in results], ["code::alpha"])

            registry = IndexRegistry()
            registry.register_index("source_code", store)
            self.assertIs(registry.get_index("source_code"), store)


if __name__ == "__main__":
    unittest.main()
