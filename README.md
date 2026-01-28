# Agentic RAG Knowledge System (FAISS)

This repository provides a modular Agentic RAG Knowledge System that extracts
knowledge units from source code, tests, issues, and docs, stores intent metadata
in editable sidecar YAML files, and indexes heterogeneous knowledge via FAISS.

## Key Concepts

- **KnowledgeUnit** encapsulates any retrievable artifact (code, tests, issues, docs) with
  stable IDs, metadata, and explicit relationships.
- **Sidecar metadata** lives in `.meta.yaml` files that mirror the source tree, enabling
  "edit while reading" workflows and compatibility with non-AST languages.
- **Multi-index architecture** separates source code, tests, issues, and knowledge into
  independent vector indices for targeted retrieval.
- **CRUD-enabled FAISS store** uses integer ID mappings to support updates and deletions
  without full re-indexing.
- **Router + ReAct agent** selects indices, retrieves context, and produces final answers
  in a reasoning-and-acting loop.

## Quick Start (Demo)

The demo creates a few in-memory symbols, indexes them, and runs a query.

```bash
python examples/demo.py
```

Launch the multi-index ReAct demo (requires the configured LLM and embedding servers):

```bash
python examples/demo_gradio.py
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

- `src/core/knowledge_unit.py`: `KnowledgeUnit` definition.
- `src/agent/router.py`: router interface and keyword router.
- `src/ingest.py`: file scanning, sidecar manager, and ingestion pipeline.
- `src/extractors.py`: AST extractor + regex extractor + generic chunking.
- `src/stores/faiss_store.py`: CRUD-capable FAISS store + ID mapping + registry.
- `src/knowledge_store.py`: JSON knowledge base I/O (legacy).
- `src/refinery.py`: drift detection + reconciliation (legacy).
- `src/vector_store.py`: legacy FAISS manager + docstore.
- `refine.py`: update knowledge base from `src/`.
- `examples/demo.py`: minimal end-to-end usage example.
- `examples/demo_gradio.py`: router-driven ReAct agent demo.

## Development

```bash
pip install -r requirements.txt
ruff check .
python -m pytest
```
