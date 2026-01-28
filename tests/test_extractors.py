import os
import tempfile
import unittest
from pathlib import Path

from src.extractors import PythonAstExtractor, RegexExtractor
from src.ingest import FileLoader


class TestExtractors(unittest.TestCase):
    def test_parse_file_with_symlinked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as target_dir:
            root = Path(temp_dir)
            source_root = root / "doc"
            target_root = Path(target_dir)
            source_root.mkdir()
            target_file = target_root / "linked.py"
            target_file.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            link_path = source_root / "link"
            try:
                os.symlink(target_root, link_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Symlink not supported: {exc}")

            loader = FileLoader(valid_extensions=(".py",), follow_symlinks=True)
            files = loader.scan_directory(str(source_root))
            linked_path = next(path for path in files if path.endswith("linked.py"))

            extractor = PythonAstExtractor(str(source_root))
            symbols = extractor.parse_file(linked_path)

            self.assertEqual(len(symbols), 1)
            self.assertTrue(symbols[0].symbol_id.endswith("linked:alpha"))

    def test_regex_extractor_finds_perl_subs(self) -> None:
        content = "sub alpha {\\n  return 1;\\n}\\nsub beta {\\n  return 2;\\n}\\n"
        extractor = RegexExtractor(r"^sub\s+(\w+)")
        symbols = extractor.extract_symbols("script.pl", content)
        names = [symbol.symbol_name for symbol in symbols]

        self.assertEqual(names, ["alpha", "beta"])
        self.assertEqual(symbols[0].start_line, 1)


if __name__ == "__main__":
    unittest.main()
