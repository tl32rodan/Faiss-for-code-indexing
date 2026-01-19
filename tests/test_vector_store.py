import unittest
from src.vector_store import FaissManager

class TestVectorStore(unittest.TestCase):
    def test_faiss_manager_stub(self):
        manager = FaissManager()
        # Ensure methods exist and are callable
        self.assertIsNone(manager.add_chunks([], None))
        self.assertIsNone(manager.search(None))
        self.assertIsNone(manager.save_local("path"))
        self.assertIsNone(manager.load_local("path"))

if __name__ == '__main__':
    unittest.main()
