from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import gradio as gr
import requests

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if TYPE_CHECKING:
    from src.vector_store import FaissManager


class LLMClient(Protocol):
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response for the provided prompt."""


class SimpleLLM:
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return (
            "**[模擬 LLM 回應]**\n"
            f"收到了你的問題：'{prompt[-50:]}...'\n"
            "根據檢索到的 Context，我認為..."
        )


class RemoteEmbeddingModel:
    def __init__(
        self,
        endpoint_url: str,
        dimension: int,
        batch_size: int = 32,
        max_workers: int = 8,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_workers = max_workers

    def encode(self, texts: list[str]) -> list[list[float]]:
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        if not batches:
            return []
        results: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch_vectors in executor.map(self._send_request, batches):
                results.extend(batch_vectors)
        return results

    def _send_request(self, batch_texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.endpoint_url, json={"inputs": batch_texts}, timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("embeddings", [])

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


def initialize_llm_and_embedding_model() -> tuple[LLMClient, RemoteEmbeddingModel]:
    print("Step 1: Initializing models...")
    endpoint_url = os.environ.get("EMBEDDING_ENDPOINT", "http://localhost:8001/embed")
    dimension = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
    batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
    max_workers = int(os.environ.get("EMBEDDING_MAX_WORKERS", "8"))
    embedding_model = RemoteEmbeddingModel(
        endpoint_url=endpoint_url,
        dimension=dimension,
        batch_size=batch_size,
        max_workers=max_workers,
    )
    llm = SimpleLLM()
    return llm, embedding_model


def setup_vector_database(embedding_model: RemoteEmbeddingModel) -> "FaissManager":
    print("Step 3: Setting up Vector DB Manager...")
    from src.vector_store import FaissManager

    dimension = embedding_model.get_sentence_embedding_dimension()
    return FaissManager(dimension=dimension)


def index_documents_if_needed(
    docs_dir: Path,
    know_dir: Path,
    vector_dir: Path,
    vector_db: "FaissManager",
    embedding_model: RemoteEmbeddingModel,
    updated_count: int,
) -> None:
    print("Step 4: Checking index status...")
    from src.knowledge_store import JSONKnowledgeStore

    store = JSONKnowledgeStore(str(know_dir), str(docs_dir))
    loaded = vector_db.load_local(str(vector_dir))
    if loaded:
        if updated_count > 0:
            print("Warning: Knowledge was updated; syncing the vector index.")
        print(f"Loading existing index from {vector_dir}...")
    else:
        print(f"No index found in {vector_dir}. Building from Knowledge Store...")

    print(f"Syncing vector DB from knowledge store (updated: {updated_count})...")
    vector_db.index_from_store(store, embedding_model)

    if updated_count > 0 or not loaded:
        print(f"Saving updated index to {vector_dir}...")
        vector_db.save_local(str(vector_dir))


def initialize_chatbot(
    system_prompt: str,
    llm: LLMClient,
    vector_db: "FaissManager",
    embedding_model: RemoteEmbeddingModel,
) -> gr.ChatInterface:
    print("Step 5: Initializing Chatbot Engine...")
    from src.search import CodeSearchEngine

    search_engine = CodeSearchEngine(vector_db=vector_db, embedding_model=embedding_model)

    def chat_function(message: str, history: list[dict[str, str]]) -> str:
        print(f"Querying: {message}")
        context_str = search_engine.query(message, top_k=3)

        if not context_str.strip():
            context_str = "No relevant code found in knowledge base."

        full_prompt = (
            "Context from codebase:\n"
            f"{context_str}\n\n"
            f"User Question: {message}\n"
            "Answer:"
        )

        return llm.generate(full_prompt, system_prompt=system_prompt)

    return gr.ChatInterface(
        fn=chat_function,
        title="Codebase RAG Demo",
        description="Ask questions about the indexed codebase.",
        examples=["How do I add symbols?", "Explain the refine process."],
    )


def main() -> None:
    from refine import run_refine

    docs_dir = Path(os.environ.get("DOCS_DIR", "source_code_link")).expanduser().resolve()
    know_dir = Path(os.environ.get("KNOW_DIR", "knowledge_base")).expanduser().resolve()
    vector_dir = Path(os.environ.get("VECTOR_DIR", "vector_db_storage")).expanduser().resolve()

    if not docs_dir.exists():
        docs_dir.mkdir(exist_ok=True, parents=True)
    know_dir.mkdir(exist_ok=True, parents=True)
    vector_dir.mkdir(exist_ok=True, parents=True)

    llm, embedding_model = initialize_llm_and_embedding_model()

    print(f"Step 2: Refining knowledge from {docs_dir} into {know_dir}...")
    updated_count = run_refine(str(docs_dir), str(know_dir))
    print(f"Refinement complete. {updated_count} symbols updated/created.")

    vector_db = setup_vector_database(embedding_model)

    index_documents_if_needed(
        docs_dir,
        know_dir,
        vector_dir,
        vector_db,
        embedding_model,
        updated_count,
    )

    system_prompt = "You are a helpful coding assistant."
    demo = initialize_chatbot(system_prompt, llm, vector_db, embedding_model)

    print("Step 6 & 7: Launching Gradio...")
    demo.launch(share=False)


if __name__ == "__main__":
    main()
