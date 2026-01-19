# Code RAG Framework (FAISS)

This repository provides a minimal, testable scaffold for a Code RAG system that
combines code snippets with intent metadata and indexes them via FAISS.

## Key Concepts

- **CodeChunk** encapsulates a code snippet plus metadata for embedding and display.
- **Hybrid Context Embedding** is achieved by prefixing file path, quality tier, and
  intent to the code before embedding, while keeping them separate for display.
- **FAISS-backed vector store** manages chunk embeddings and metadata.

## Quick Start (Demo)

The demo creates a few in-memory chunks, indexes them, and runs a query.

```bash
python examples/demo.py
```

## Project Structure

- `src/models.py`: `CodeChunk` definition.
- `src/ingest.py`: file scanning and chunking.
- `src/vector_store.py`: FAISS manager + docstore.
- `src/intent.py`: intent update and re-embedding.
- `src/search.py`: query facade and prompt formatting.
- `examples/demo.py`: minimal end-to-end usage example.

## Development

```bash
pip install -r requirements.txt
ruff check .
python -m unittest discover -s tests
```
