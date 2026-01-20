import tempfile
import unittest
from pathlib import Path

from refine import run_refine


class TestRefineScript(unittest.TestCase):
    def test_run_refine_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / "missing"
            knowledge_root = Path(temp_dir) / "knowledge_base"
            updated = run_refine(str(missing_root), str(knowledge_root))
            self.assertEqual(updated, 0)
            self.assertFalse(knowledge_root.exists())

    def test_run_refine_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "src"
            knowledge_root = Path(temp_dir) / "knowledge_base"
            source_root.mkdir()
            updated = run_refine(str(source_root), str(knowledge_root))
            self.assertEqual(updated, 0)
            self.assertFalse(knowledge_root.exists())

    def test_run_refine_writes_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "src"
            knowledge_root = Path(temp_dir) / "knowledge_base"
            source_root.mkdir()
            (source_root / "demo.py").write_text(
                "def alpha():\n    return 1\n", encoding="utf-8"
            )
            updated = run_refine(str(source_root), str(knowledge_root))
            self.assertEqual(updated, 1)
            self.assertTrue((knowledge_root / "demo.json").exists())

    def test_run_refine_non_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "src"
            knowledge_root = Path(temp_dir) / "knowledge_base"
            source_root.mkdir()
            (source_root / "notes.md").write_text("Hello\n", encoding="utf-8")
            (source_root / "script.tcl").write_text("puts \"hi\"\n", encoding="utf-8")
            updated = run_refine(str(source_root), str(knowledge_root))
            self.assertEqual(updated, 2)
            self.assertTrue((knowledge_root / "notes.json").exists())
            self.assertTrue((knowledge_root / "script.json").exists())


if __name__ == "__main__":
    unittest.main()
