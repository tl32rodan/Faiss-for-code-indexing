import tempfile
import unittest
from pathlib import Path

from src.knowledge_store import JSONKnowledgeStore
from src.models import SymbolChunk, compute_code_hash


class TestKnowledgeStore(unittest.TestCase):
    def test_symlinked_source_path_uses_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "doc"
            knowledge_root = root / "knowledge_base"
            external_root = root / "external"
            source_root.mkdir()
            external_root.mkdir()

            source_file = external_root / "notes.txt"
            source_file.write_text("hello", encoding="utf-8")

            linked_dir = source_root / "linked"
            linked_dir.symlink_to(external_root, target_is_directory=True)
            linked_file = linked_dir / "notes.txt"

            store = JSONKnowledgeStore(str(knowledge_root), str(source_root))
            chunk = SymbolChunk(
                symbol_id="linked:notes",
                filepath=str(linked_file),
                symbol_name="notes",
                symbol_kind="file",
                start_line=1,
                end_line=1,
                content="hello",
                code_hash=compute_code_hash("hello"),
            )
            store.save_symbols(str(linked_file), [chunk])

            knowledge_path = knowledge_root / "linked" / "notes.json"
            self.assertTrue(knowledge_path.exists())
            loaded = store.load_symbols(str(linked_file))
            self.assertIn("linked:notes", loaded)


if __name__ == "__main__":
    unittest.main()
