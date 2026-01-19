from src.models import CodeChunk, _compute_chunk_id


def test_code_chunk_id_generation() -> None:
    chunk = CodeChunk(filepath="/tmp/example.py", content="print('hi')")
    assert chunk.id == _compute_chunk_id(chunk.filepath, chunk.content)


def test_get_embedding_content() -> None:
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
    assert chunk.get_embedding_content() == expected
