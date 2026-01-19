from pathlib import Path

from src.models import CodeChunk
from src.vector_store import FaissManager
from tests.conftest import DummyEmbeddingModel, FakeIndex


def test_add_and_search_chunks() -> None:
    embedding_model = DummyEmbeddingModel(dimension=4)
    manager = FaissManager(dimension=4, index=FakeIndex(4))
    chunks = [
        CodeChunk(filepath="/repo/src/a.py", content="alpha"),
        CodeChunk(filepath="/repo/src/b.py", content="beta"),
    ]
    manager.add_chunks(chunks, embedding_model)
    query_vector = embedding_model.encode(["alpha"])
    results = manager.search(query_vector, top_k=2)

    assert len(results) == 2
    assert results[0].filepath in {"/repo/src/a.py", "/repo/src/b.py"}


def test_deactivate_chunk_skips_results() -> None:
    embedding_model = DummyEmbeddingModel(dimension=4)
    manager = FaissManager(dimension=4, index=FakeIndex(4))
    chunk = CodeChunk(filepath="/repo/src/a.py", content="alpha")
    manager.add_chunks([chunk], embedding_model)
    manager.deactivate_chunk(chunk.id)
    query_vector = embedding_model.encode(["alpha"])
    results = manager.search(query_vector, top_k=1)

    assert results == []


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    embedding_model = DummyEmbeddingModel(dimension=4)
    manager = FaissManager(dimension=4, index=FakeIndex(4))
    chunk = CodeChunk(filepath="/repo/src/a.py", content="alpha")
    manager.add_chunks([chunk], embedding_model)

    manager.save_local(str(tmp_path))
    new_manager = FaissManager(dimension=4, index=FakeIndex(4))
    new_manager.load_local(str(tmp_path))

    query_vector = embedding_model.encode(["alpha"])
    results = new_manager.search(query_vector, top_k=1)
    assert results[0].filepath == "/repo/src/a.py"
