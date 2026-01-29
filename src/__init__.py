"""FAISS storage library for vector documents."""

from src.core import VectorDocument
from src.engine import FaissEngine, IndexRegistry

__all__ = ["FaissEngine", "IndexRegistry", "VectorDocument"]
