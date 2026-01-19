import unittest
from src.ingest import FileLoader, CodeSplitter

class TestIngest(unittest.TestCase):
    def test_file_loader_stub(self):
        loader = FileLoader()
        # Ensure methods exist and are callable
        self.assertIsNone(loader.scan_directory("."))
        self.assertIsNone(loader.determine_tier("file.py"))

    def test_code_splitter_stub(self):
        splitter = CodeSplitter()
        # Ensure methods exist and are callable
        self.assertIsNone(splitter.chunk_file("file.py", "content"))

if __name__ == '__main__':
    unittest.main()
