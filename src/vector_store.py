from typing import List, Any, Dict
import numpy as np
from .models import CodeChunk

class FaissManager:
    def __init__(self):
        """
        Initialize a FAISS index (using IndexFlatIP or IndexIVFFlat for inner product/cosine similarity).
        Initialize a docstore (Dictionary or generic Key-Value store) to map chunk_id -> CodeChunk object.
        """
        pass

    def add_chunks(self, chunks: List[CodeChunk], embedding_model: Any):
        """
        Call chunk.get_embedding_content() for each chunk.
        Generate vectors using the embedding model.
        index.add(vectors) to FAISS.
        Store CodeChunk objects in docstore keyed by ID.
        """
        pass

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[CodeChunk]:
        """
        index.search(query_vector, k).
        Retrieve full CodeChunk objects from docstore using returned indices.
        """
        pass

    def save_local(self, path: str):
        """Serialize both the FAISS index and the docstore to disk."""
        pass

    def load_local(self, path: str):
        """Deserialize from disk."""
        pass
