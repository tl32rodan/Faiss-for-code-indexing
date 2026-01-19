import unittest

from src.models import CodeChunk, _compute_chunk_id


class TestCodeChunk(unittest.TestCase):
    def test_code_chunk_id_generation(self) -> None:
        chunk = CodeChunk(filepath="/tmp/example.py", content="print('hi')")
        self.assertEqual(chunk.id, _compute_chunk_id(chunk.filepath, chunk.content))

    def test_get_embedding_content(self) -> None:
        chunk = CodeChunk(
            filepath="/tmp/example.py",
            content="print('hi')",
            quality_tier="GOLD",
            meta_intent="demo",
        )
        expected = (
            "# File: /tmp/example.py\n"
            "# Tier: GOLD\n"
            "# Intent: demo\n"
            "print('hi')"
        )
        self.assertEqual(chunk.get_embedding_content(), expected)


if __name__ == "__main__":
    unittest.main()
