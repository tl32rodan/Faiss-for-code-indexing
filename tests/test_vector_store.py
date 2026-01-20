import tempfile
import unittest

from src.knowledge_store import JSONKnowledgeStore
from src.models import SymbolChunk, compute_code_hash
from src.vector_store import FaissManager
from tests.conftest import DummyEmbeddingModel, FakeIndex


class TestVectorStore(unittest.TestCase):
    def test_add_and_search_chunks(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunks = [
            SymbolChunk(
                symbol_id="a:alpha",
                filepath="/repo/src/a.py",
                symbol_name="alpha",
                symbol_kind="function",
                start_line=1,
                end_line=1,
                content="alpha",
                code_hash=compute_code_hash("alpha"),
            ),
            SymbolChunk(
                symbol_id="b:beta",
                filepath="/repo/src/b.py",
                symbol_name="beta",
                symbol_kind="function",
                start_line=1,
                end_line=1,
                content="beta",
                code_hash=compute_code_hash("beta"),
            ),
        ]
        manager.add_symbols(chunks, embedding_model)
        query_vector = embedding_model.encode(["alpha"])
        results = manager.search(query_vector, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn(results[0].filepath, {"/repo/src/a.py", "/repo/src/b.py"})

    def test_deactivate_symbol_skips_results(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = SymbolChunk(
            symbol_id="a:alpha",
            filepath="/repo/src/a.py",
            symbol_name="alpha",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="alpha",
            code_hash=compute_code_hash("alpha"),
        )
        manager.add_symbols([chunk], embedding_model)
        manager.deactivate_symbol(chunk.symbol_id)
        query_vector = embedding_model.encode(["alpha"])
        results = manager.search(query_vector, top_k=1)

        self.assertEqual(results, [])

    def test_save_and_load_roundtrip(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        chunk = SymbolChunk(
            symbol_id="a:alpha",
            filepath="/repo/src/a.py",
            symbol_name="alpha",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="alpha",
            code_hash=compute_code_hash("alpha"),
        )
        manager.add_symbols([chunk], embedding_model)

        with tempfile.TemporaryDirectory() as temp_dir:
            manager.save_local(temp_dir)
            new_manager = FaissManager(dimension=4, index=FakeIndex(4))
            loaded = new_manager.load_local(temp_dir)
            self.assertTrue(loaded)

        query_vector = embedding_model.encode(["alpha"])
        results = new_manager.search(query_vector, top_k=1)
        self.assertEqual(results[0].filepath, "/repo/src/a.py")

    def test_load_local_noop_when_missing(self) -> None:
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        with tempfile.TemporaryDirectory() as temp_dir:
            loaded = manager.load_local(temp_dir)
            self.assertFalse(loaded)
            self.assertEqual(manager.docstore, {})

    def test_index_from_store_incremental_and_stale(self) -> None:
        embedding_model = DummyEmbeddingModel(dimension=4)
        manager = FaissManager(dimension=4, index=FakeIndex(4))
        existing_ok = SymbolChunk(
            symbol_id="a:alpha",
            filepath="/repo/src/a.py",
            symbol_name="alpha",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="alpha",
            code_hash=compute_code_hash("alpha"),
            status="OK",
        )
        existing_stale = SymbolChunk(
            symbol_id="b:beta",
            filepath="/repo/src/b.py",
            symbol_name="beta",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="beta",
            code_hash=compute_code_hash("beta"),
            status="OK",
        )
        manager.add_symbols([existing_ok, existing_stale], embedding_model)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = JSONKnowledgeStore(temp_dir, "/repo/src")
            new_symbol = SymbolChunk(
                symbol_id="c:gamma",
                filepath="/repo/src/c.py",
                symbol_name="gamma",
                symbol_kind="function",
                start_line=1,
                end_line=1,
                content="gamma",
                code_hash=compute_code_hash("gamma"),
                status="NEW",
            )
            store.save_symbols(
                "/repo/src/a.py",
                [
                    SymbolChunk(
                        symbol_id="a:alpha",
                        filepath="/repo/src/a.py",
                        symbol_name="alpha",
                        symbol_kind="function",
                        start_line=1,
                        end_line=1,
                        content="alpha",
                        code_hash=compute_code_hash("alpha"),
                        status="OK",
                    ),
                    SymbolChunk(
                        symbol_id="b:beta",
                        filepath="/repo/src/b.py",
                        symbol_name="beta",
                        symbol_kind="function",
                        start_line=1,
                        end_line=1,
                        content="beta",
                        code_hash=compute_code_hash("beta"),
                        status="STALE",
                    ),
                    new_symbol,
                ],
            )
            manager.index_from_store(store, embedding_model)

        self.assertIn("c:gamma", manager.docstore)
        self.assertEqual(len(manager.docstore), 3)
        stale_positions = manager._positions_by_id["b:beta"]
        for position in stale_positions:
            self.assertIn(position, manager._inactive_positions)
        self.assertEqual(len(manager._positions_by_id["a:alpha"]), 1)


if __name__ == "__main__":
    unittest.main()
