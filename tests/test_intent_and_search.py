import unittest

from src.intent import IntentManager
from src.models import CodeChunk
from src.search import CodeSearchEngine
from src.vector_store import FaissManager
from tests.conftest import DummyEmbeddingModel, FakeIndex


class TestIntentAndSearch(unittest.TestCase):
    def test_intent_update_reindexes(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = CodeChunk(filepath="/repo/src/a.py", content="alpha")
        manager.add_chunks([chunk], embedding_model)

        intent_manager = IntentManager()
        intent_manager.update_intent(chunk.id, "new intent", manager, embedding_model)

        query_vector = embedding_model.encode(["alpha"])
        results = manager.search(query_vector, top_k=2)
        self.assertEqual(results[0].meta_intent, "new intent")

    def test_code_search_engine_formats_results(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = CodeChunk(
            filepath="/repo/src/a.py",
            content="alpha",
            meta_intent="demo",
            quality_tier="GOLD",
            start_line=10,
        )
        manager.add_chunks([chunk], embedding_model)

        engine = CodeSearchEngine(vector_db=manager, embedding_model=embedding_model)
        output = engine.query("alpha")

        self.assertIn("File: /repo/src/a.py", output)
        self.assertIn("Tier: GOLD", output)
        self.assertIn("Intent: demo", output)


if __name__ == "__main__":
    unittest.main()
