from pathlib import Path

from src.ingest import CodeSplitter, FileLoader


def test_scan_directory_ignores_hidden_and_binary(tmp_path: Path) -> None:
    visible = tmp_path / "visible.py"
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    hidden_file = hidden_dir / "secret.py"
    binary_file = tmp_path / "data.bin"

    visible.write_text("print('ok')\n", encoding="utf-8")
    hidden_file.write_text("print('no')\n", encoding="utf-8")
    binary_file.write_bytes(b"\x00\x01\x02")

    loader = FileLoader(valid_extensions=(".py", ".bin"))
    results = loader.scan_directory(str(tmp_path))

    assert str(visible) in results
    assert str(hidden_file) not in results
    assert str(binary_file) not in results


def test_determine_tier() -> None:
    loader = FileLoader()
    assert loader.determine_tier("/repo/src/main.py") == "GOLD"
    assert loader.determine_tier("/repo/tests/test_main.py") == "SILVER"
    assert loader.determine_tier("/repo/other/file.py") == "JUNK"


def test_code_splitter_chunks_with_overlap() -> None:
    splitter = CodeSplitter(window_size=4, overlap=2, tier_resolver=lambda _: "GOLD")
    raw_text = "one two three four five six"
    chunks = splitter.chunk_file("/repo/src/main.py", raw_text)

    assert len(chunks) == 2
    assert chunks[0].content == "one two three four"
    assert chunks[1].content == "three four five six"
    assert chunks[0].start_line == 1
