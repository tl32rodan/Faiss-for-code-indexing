import unittest

from src.models import SymbolChunk, compute_code_hash


class TestSymbolChunk(unittest.TestCase):
    def test_code_hash_generation(self) -> None:
        content = "print('hi')"
        self.assertEqual(compute_code_hash(content), compute_code_hash(content))

    def test_get_embedding_content(self) -> None:
        chunk = SymbolChunk(
            symbol_id="example:print_hi",
            filepath="/tmp/example.py",
            symbol_name="print_hi",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="print('hi')",
            code_hash=compute_code_hash("print('hi')"),
            intent="demo",
        )
        expected = (
            "# Symbol: example:print_hi\n"
            "# File: /tmp/example.py\n"
            "# Kind: function\n"
            "# Intent: demo\n"
            "print('hi')"
        )
        self.assertEqual(chunk.get_embedding_content(), expected)

    def test_to_from_dict_roundtrip(self) -> None:
        chunk = SymbolChunk(
            symbol_id="example:print_hi",
            filepath="/tmp/example.py",
            symbol_name="print_hi",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            content="print('hi')",
            code_hash=compute_code_hash("print('hi')"),
            intent="demo",
            status="OK",
        )
        payload = chunk.to_dict()
        recreated = SymbolChunk.from_dict(payload)
        self.assertEqual(recreated, chunk)


if __name__ == "__main__":
    unittest.main()
