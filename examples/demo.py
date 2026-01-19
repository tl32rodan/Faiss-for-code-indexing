from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.models import SymbolChunk, compute_code_hash
from src.search import CodeSearchEngine
from src.vector_store import FaissManager


def main() -> None:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dimension = embedding_model.get_sentence_embedding_dimension()
    vector_db = FaissManager(dimension=dimension)

    chunks = [
        SymbolChunk(
            symbol_id="math:add",
            filepath="/repo/src/math.py",
            symbol_name="add",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="def add(a, b): return a + b",
            code_hash=compute_code_hash("def add(a, b): return a + b"),
            intent="Simple math helper",
            status="OK",
        ),
        SymbolChunk(
            symbol_id="io:read_text",
            filepath="/repo/src/io.py",
            symbol_name="read_text",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="def read_text(path): return Path(path).read_text()",
            code_hash=compute_code_hash(
                "def read_text(path): return Path(path).read_text()"
            ),
            intent="File reading helper",
            status="OK",
        ),
    ]

    vector_db.add_symbols(chunks, embedding_model)
    search_engine = CodeSearchEngine(vector_db=vector_db, embedding_model=embedding_model)

    print(search_engine.query("How do I add numbers?"))


if __name__ == "__main__":
    main()
