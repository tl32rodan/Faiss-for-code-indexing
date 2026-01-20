from __future__ import annotations

from pathlib import Path
from typing import Protocol

import gradio as gr
from sentence_transformers import SentenceTransformer

from refine import run_refine
from src.knowledge_store import JSONKnowledgeStore
from src.search import CodeSearchEngine
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


DOCS_DIR = Path("source_code_link").expanduser().resolve()
KNOW_DIR = Path("knowledge_base").expanduser().resolve()
VECTOR_DIR = Path("vector_db_storage").expanduser().resolve()


def initialize_llm_and_embedding_model() -> tuple[LLMClient, SentenceTransformer]:
    print("Step 1: Initializing models...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = SimpleLLM()
    return llm, embedding_model


def setup_vector_database(embedding_model: SentenceTransformer) -> FaissManager:
    print("Step 3: Setting up Vector DB Manager...")
    dimension = embedding_model.get_sentence_embedding_dimension()
    return FaissManager(dimension=dimension)


def index_documents_if_needed(
    docs_dir: Path,
    know_dir: Path,
    vector_dir: Path,
    vector_db: FaissManager,
    embedding_model: SentenceTransformer,
    updated_count: int,
) -> None:
    print("Step 4: Checking index status...")
    store = JSONKnowledgeStore(str(know_dir), str(docs_dir))
    loaded = vector_db.load_local(str(vector_dir))
    if loaded:
        if updated_count > 0:
            print(
                "Warning: Knowledge was updated during refine, but an index already exists. "
                "Consider rebuilding the vector index to stay in sync."
            )
        print(f"Loading existing index from {vector_dir}...")
        return

    print(f"No index found in {vector_dir}. Building from Knowledge Store...")
    vector_db.index_from_store(store, embedding_model)

    print(f"Saving new index to {vector_dir}...")
    vector_db.save_local(str(vector_dir))


def initialize_chatbot(
    system_prompt: str,
    llm: LLMClient,
    vector_db: FaissManager,
    embedding_model: SentenceTransformer,
) -> gr.ChatInterface:
    print("Step 5: Initializing Chatbot Engine...")
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
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(exist_ok=True, parents=True)
    KNOW_DIR.mkdir(exist_ok=True, parents=True)
    VECTOR_DIR.mkdir(exist_ok=True, parents=True)

    llm, embedding_model = initialize_llm_and_embedding_model()

    print(f"Step 2: Refining knowledge from {DOCS_DIR} into {KNOW_DIR}...")
    updated_count = run_refine(str(DOCS_DIR), str(KNOW_DIR))
    print(f"Refinement complete. {updated_count} symbols updated/created.")

    vector_db = setup_vector_database(embedding_model)

    index_documents_if_needed(
        DOCS_DIR,
        KNOW_DIR,
        VECTOR_DIR,
        vector_db,
        embedding_model,
        updated_count,
    )

    system_prompt = "You are a helpful coding assistant named AI-ZhiCheng."
    demo = initialize_chatbot(system_prompt, llm, vector_db, embedding_model)

    print("Step 6 & 7: Launching Gradio...")
    demo.launch(share=False)


if __name__ == "__main__":
    main()
