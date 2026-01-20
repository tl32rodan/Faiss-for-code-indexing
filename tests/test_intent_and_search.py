import tempfile
import unittest

from src.intent import IntentManager
from src.knowledge_store import JSONKnowledgeStore
from src.models import SymbolChunk, compute_code_hash
from src.search import CodeSearchEngine
from src.vector_store import FaissManager
from tests.conftest import DummyEmbeddingModel, FakeIndex


class TestIntentAndSearch(unittest.TestCase):
    def test_intent_update_reindexes(self) -> None:
        chunk = SymbolChunk(
            symbol_id="a:alpha",
            filepath="/repo/src/a.py",
            symbol_name="alpha",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="alpha",
            code_hash=compute_code_hash("alpha"),
            meta_hash=compute_code_hash("alpha"),
            intent="old intent",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JSONKnowledgeStore(temp_dir, "/repo/src")
            store.save_symbols("/repo/src/a.py", [chunk])
            intent_manager = IntentManager()
            intent_manager.update_intent("a:alpha", "new intent", store, "/repo/src/a.py")
            updated = store.load_symbols("/repo/src/a.py")

        self.assertEqual(updated["a:alpha"].intent, "new intent")

    def test_code_search_engine_formats_results(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = SymbolChunk(
            symbol_id="a:alpha",
            filepath="/repo/src/a.py",
            symbol_name="alpha",
            symbol_kind="function",
            start_line=10,
            end_line=10,
            content="alpha",
            code_hash=compute_code_hash("alpha"),
            intent="demo",
            status="OK",
        )
        manager.add_symbols([chunk], embedding_model)

        engine = CodeSearchEngine(vector_db=manager, embedding_model=embedding_model)
        output = engine.query("alpha", top_k=1)

        self.assertIn("Symbol: a:alpha", output)
        self.assertIn("File: /repo/src/a.py", output)
        self.assertIn("Kind: function", output)
        self.assertIn("Intent: demo", output)


if __name__ == "__main__":
    unittest.main()
