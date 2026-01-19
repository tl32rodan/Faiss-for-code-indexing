import unittest
from src.models import CodeChunk, QualityTier

class TestModels(unittest.TestCase):
    def test_code_chunk_embedding_content(self):
        chunk = CodeChunk(
            id="123",
            filepath="/path/to/file.py",
            content="print('hello')",
            meta_intent="Greeting",
            quality_tier=QualityTier.GOLD,
            start_line=1
        )
        expected = "# File: /path/to/file.py\n# Tier: GOLD\n# Intent: Greeting\nprint('hello')"
        self.assertEqual(chunk.get_embedding_content(), expected)

    def test_code_chunk_default_intent(self):
        chunk = CodeChunk(
            id="456",
            filepath="/path/to/file.py",
            content="x = 1",
            quality_tier=QualityTier.SILVER,
            start_line=10
        )
        expected = "# File: /path/to/file.py\n# Tier: SILVER\n# Intent: \nx = 1"
        self.assertEqual(chunk.get_embedding_content(), expected)

if __name__ == '__main__':
    unittest.main()
