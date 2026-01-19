import unittest
from src.search import CodeSearchEngine

class TestSearch(unittest.TestCase):
    def test_search_engine_stub(self):
        engine = CodeSearchEngine()
        # Ensure methods exist and are callable
        self.assertIsNone(engine.query("question"))

if __name__ == '__main__':
    unittest.main()
