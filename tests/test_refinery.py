import tempfile
import unittest
from pathlib import Path

from src.extractors import PythonAstExtractor
from src.knowledge_store import JSONKnowledgeStore
from src.refinery import KnowledgeRefinery


class TestRefinery(unittest.TestCase):
    def test_refinery_marks_new_and_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            knowledge_root = root / "knowledge_base"
            source_root.mkdir()
            source_file = source_root / "module.py"
            source_file.write_text(
                "def alpha():\n    return 1\n", encoding="utf-8"
            )

            extractor = PythonAstExtractor(str(source_root))
            store = JSONKnowledgeStore(str(knowledge_root), str(source_root))
            refinery = KnowledgeRefinery(extractor, store)

            refinery.refine_file(str(source_file))
            stored = store.load_symbols(str(source_file))
            symbol = stored["module:alpha"]
            self.assertEqual(symbol.status, "NEW")
            self.assertEqual(symbol.intent, "TODO: LLM gen")
            self.assertEqual(symbol.meta_hash, symbol.code_hash)

            source_file.write_text(
                "def alpha():\n    return 2\n", encoding="utf-8"
            )
            refinery.refine_file(str(source_file))
            updated = store.load_symbols(str(source_file))
            symbol = updated["module:alpha"]
            self.assertEqual(symbol.status, "STALE")
            self.assertNotEqual(symbol.meta_hash, symbol.code_hash)


if __name__ == "__main__":
    unittest.main()
