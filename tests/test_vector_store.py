import tempfile
import unittest

from src.models import CodeChunk
from src.vector_store import FaissManager
from tests.conftest import DummyEmbeddingModel, FakeIndex


class TestVectorStore(unittest.TestCase):
    def test_add_and_search_chunks(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunks = [
            CodeChunk(filepath="/repo/src/a.py", content="alpha"),
            CodeChunk(filepath="/repo/src/b.py", content="beta"),
        ]
        manager.add_chunks(chunks, embedding_model)
        query_vector = embedding_model.encode(["alpha"])
        results = manager.search(query_vector, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn(results[0].filepath, {"/repo/src/a.py", "/repo/src/b.py"})

    def test_deactivate_chunk_skips_results(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = CodeChunk(filepath="/repo/src/a.py", content="alpha")
        manager.add_chunks([chunk], embedding_model)
        manager.deactivate_chunk(chunk.id)
        query_vector = embedding_model.encode(["alpha"])
        results = manager.search(query_vector, top_k=1)

        self.assertEqual(results, [])

    def test_save_and_load_roundtrip(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = CodeChunk(filepath="/repo/src/a.py", content="alpha")
        manager.add_chunks([chunk], embedding_model)

        with tempfile.TemporaryDirectory() as temp_dir:
            manager.save_local(temp_dir)
            new_manager = FaissManager(dimension=4, index=FakeIndex(4))
            new_manager.load_local(temp_dir)

        query_vector = embedding_model.encode(["alpha"])
        results = new_manager.search(query_vector, top_k=1)
        self.assertEqual(results[0].filepath, "/repo/src/a.py")


if __name__ == "__main__":
    unittest.main()
