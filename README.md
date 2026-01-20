# Codebase Knowledge System (FAISS)

This repository provides a modular Codebase Knowledge System that extracts
stable symbols from source code, stores intent metadata as versionable JSON
sidecars, and indexes only fresh knowledge via FAISS.

## Key Concepts

- **SymbolChunk** encapsulates a code symbol plus intent metadata for embedding and display.
- **Stable identities** are derived from AST symbols (e.g. `module:function`), so metadata
  persists across refactors.
- **JSON knowledge base** mirrors `src/` so metadata can be versioned alongside code.
- **Refinery** detects drift by comparing stored metadata hashes with current code hashes.
- **FAISS-backed vector store** indexes only symbols that are not stale.

## Quick Start (Demo)

The demo creates a few in-memory symbols, indexes them, and runs a query.

```bash
python examples/demo.py
```

## Refinery Demos

Run the refinery against a temporary `src/` tree and inspect the generated JSON:

```bash
python examples/refine_demo.py
```

Create a symbol, refine, mutate the code, and refine again to observe drift:

```bash
python examples/create_and_refine_demo.py
```

## Project Structure

- `src/models.py`: `SymbolChunk` definition.
- `src/ingest.py`: file scanning utilities.
- `src/extractors.py`: base extractor + Python AST extractor.
- `src/knowledge_store.py`: JSON knowledge base I/O.
- `src/refinery.py`: drift detection + reconciliation.
- `src/vector_store.py`: FAISS manager + docstore.
- `src/intent.py`: intent updates in JSON metadata.
- `src/search.py`: query facade and prompt formatting.
- `refine.py`: update knowledge base from `src/`.
- `examples/demo.py`: minimal end-to-end usage example.

## Development

```bash
pip install -r requirements.txt
ruff check .
python -m unittest discover -s tests
```
