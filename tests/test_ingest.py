import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.ingest import FileLoader


class TestIngest(unittest.TestCase):
    def test_scan_directory_ignores_hidden_and_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visible = root / "visible.py"
            hidden_dir = root / ".hidden"
            hidden_dir.mkdir()
            hidden_file = hidden_dir / "secret.py"
            binary_file = root / "data.bin"

            visible.write_text("print('ok')\n", encoding="utf-8")
            hidden_file.write_text("print('no')\n", encoding="utf-8")
            binary_file.write_bytes(b"\x00\x01\x02")

            loader = FileLoader(valid_extensions=(".py", ".bin"))
            results = loader.scan_directory(str(root))

            self.assertIn(str(visible), results)
            self.assertNotIn(str(hidden_file), results)
            self.assertNotIn(str(binary_file), results)

    def test_scan_directory_follows_symlink_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as target_dir:
            root = Path(temp_dir)
            target_root = Path(target_dir)
            target_file = target_root / "linked.py"
            target_file.write_text("print('link')\n", encoding="utf-8")
            link_path = root / "link"
            try:
                os.symlink(target_root, link_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Symlink not supported: {exc}")

            disabled_loader = FileLoader(valid_extensions=(".py",), follow_symlinks=False)
            disabled_results = disabled_loader.scan_directory(str(root))
            self.assertNotIn(str(target_file), disabled_results)

            enabled_loader = FileLoader(valid_extensions=(".py",), follow_symlinks=True)
            enabled_results = enabled_loader.scan_directory(str(root))
            resolved_results = {Path(path).resolve() for path in enabled_results}
            self.assertIn(target_file.resolve(), resolved_results)

    def test_scan_directory_skips_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "missing.py"
            file_path.write_text("print('gone')\n", encoding="utf-8")
            with mock.patch("pathlib.Path.open", side_effect=FileNotFoundError):
                loader = FileLoader(valid_extensions=(".py",))
                results = loader.scan_directory(str(root))
        self.assertEqual(results, [])

    def test_determine_tier(self) -> None:
        loader = FileLoader()
        self.assertEqual(loader.determine_tier("/repo/src/main.py"), "GOLD")
        self.assertEqual(loader.determine_tier("/repo/tests/test_main.py"), "SILVER")
        self.assertEqual(loader.determine_tier("/repo/other/file.py"), "JUNK")

if __name__ == "__main__":
    unittest.main()
