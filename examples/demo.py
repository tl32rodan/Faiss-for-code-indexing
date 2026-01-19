from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.models import CodeChunk
from src.search import CodeSearchEngine
from src.vector_store import FaissManager


def main() -> None:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dimension = embedding_model.get_sentence_embedding_dimension()
    vector_db = FaissManager(dimension=dimension)

    chunks = [
        CodeChunk(
            filepath="/repo/src/math.py",
            content="def add(a, b): return a + b",
            quality_tier="GOLD",
            meta_intent="Simple math helper",
            start_line=1,
        ),
        CodeChunk(
            filepath="/repo/src/io.py",
            content="def read_text(path): return Path(path).read_text()",
            quality_tier="SILVER",
            meta_intent="File reading helper",
            start_line=1,
        ),
    ]

    vector_db.add_chunks(chunks, embedding_model)
    search_engine = CodeSearchEngine(vector_db=vector_db, embedding_model=embedding_model)

    print(search_engine.query("How do I add numbers?"))


if __name__ == "__main__":
    main()
