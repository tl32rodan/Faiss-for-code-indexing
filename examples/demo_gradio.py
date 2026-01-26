from __future__ import annotations

import sys
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from src.vector_store import FaissManager
    from src.search import CodeSearchEngine
    from src.embeddings import BaseEmbeddingModel
    from refine import run_refine
except ImportError:
    pass

# --- Configuration ---
LLM_CONFIG = {
    "base_url": "http://f15dtpai1:11517/v1",
    "model_name": "gpt-oss-120b",
    "api_key": "EMPTY"
}

EMBEDDING_CONFIG = {
    "host": "f15dtpai1:11436",
    "model": "nomic_embed_text:latest",
    "batch_size": 32,
    "max_workers": 8
}

# --- Remote Embedding Model ---
class RemoteEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.host = EMBEDDING_CONFIG["host"]
        self.model_name = EMBEDDING_CONFIG["model"]
        self.batch_size = EMBEDDING_CONFIG["batch_size"]
        self.max_workers = EMBEDDING_CONFIG["max_workers"]
        self.dimension = self._fetch_dimension()

    def _fetch_dimension(self) -> int:
        print("Connecting to embedding server to fetch dimension...")
        url = f"http://{self.host}/api/embed"
        data = {"model": self.model_name, "input": "ping"}
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            if not embeddings or not embeddings[0]:
                raise ValueError("Empty embedding.")
            return len(embeddings[0])
        except Exception as e:
            print(f"Error fetching dimension: {e}")
            raise RuntimeError("Could not determine dimension.")

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        if not texts: return []
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch_embeddings in executor.map(self._embed_batch, batches):
                results.extend(batch_embeddings)
        return results

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        url = f"http://{self.host}/api/embed"
        data = {"model": self.model_name, "input": batch_texts}
        try:
            with requests.Session() as session:
                response = session.post(url, json=data, timeout=60)
                response.raise_for_status()
                embeddings = response.json().get("embeddings", [])
                if len(embeddings) != len(batch_texts):
                    raise ValueError(f"Batch mismatch.")
                return embeddings
        except Exception as e:
            print(f"Batch error: {e}")
            raise

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

# --- Chatbot Logic ---
class MyBot:
    def __init__(self, vector_db: FaissManager, embedding_model: RemoteEmbeddingModel):
        self.search_engine = CodeSearchEngine(vector_db=vector_db, embedding_model=embedding_model)
        self.llm_client = ChatOpenAI(
            openai_api_key=LLM_CONFIG["api_key"],
            openai_api_base=LLM_CONFIG["base_url"],
            model_name=LLM_CONFIG["model_name"],
            temperature=0.3
        )
        self.messages: List[BaseMessage] = []
        self.messages.append(HumanMessage(content="You are a helpful coding assistant.")) 

    def chat(self, user_text: str) -> str:
        retrieved_content_str = self.search_engine.query(user_text, top_k=5)
        full_prompt = (
            f"**Prompt**:{user_text}\n"
            f"**Retrieved Documents**:\n{retrieved_content_str}\n"
            f"**Prompt**:{user_text}"
        )
        self.messages.append(HumanMessage(content=full_prompt))
        print(f"Invoking LLM with {len(self.messages)} messages...")
        response = self.llm_client.invoke(self.messages)
        self.messages.append(response)
        return response.content

# --- Main ---
def setup_knowledge_base():
    docs_dir = Path("source_code_link").resolve()
    know_dir = Path("knowledge_base").resolve()
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        if not any(docs_dir.iterdir()):
            (docs_dir / "demo_placeholder.py").write_text("def hello(): pass", encoding="utf-8")
    run_refine(str(docs_dir), str(know_dir))
    embedding_model = RemoteEmbeddingModel()
    vector_db = FaissManager(dimension=embedding_model.get_sentence_embedding_dimension())
    from src.knowledge_store import JSONKnowledgeStore
    store = JSONKnowledgeStore(str(know_dir), str(docs_dir))
    vector_db.index_from_store(store, embedding_model)
    return vector_db, embedding_model

def main():
    global_vector_db, global_embedding_model = setup_knowledge_base()

    with gr.Blocks(title="Codebase RAG Demo") as demo:
        gr.Markdown("# Codebase RAG Agent")

        bot_state = gr.State()
        history_state = gr.State([])

        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(placeholder="Ask a question...", label="User Input")
        clear = gr.Button("Clear Context")

        def init_user_session():
            return MyBot(vector_db=global_vector_db, embedding_model=global_embedding_model), []

        demo.load(init_user_session, None, [bot_state, history_state])

        def respond(user_message, bot_instance, current_history):
            if bot_instance is None:
                bot_instance, _ = init_user_session()
                current_history = []

            try:
                bot_response = bot_instance.chat(user_message)
            except Exception as e:
                bot_response = f"Error: {str(e)}"

            current_history.append((user_message, str(bot_response)))

            return current_history, current_history, "", bot_instance

        def clear_context():
            new_bot, new_history = init_user_session()
            return new_history, new_history, new_bot

        msg.submit(
            respond, 
            [msg, bot_state, history_state], 
            [history_state, chatbot, msg, bot_state]
        )
        clear.click(
            clear_context, 
            None, 
            [history_state, chatbot, bot_state]
        )

    print("Launching Gradio Server...")
    demo.launch(share=False)

if __name__ == "__main__":
    main()