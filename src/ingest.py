from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.models import CodeChunk


def _is_binary_file(path: Path, sample_size: int = 1024) -> bool:
    with path.open("rb") as handle:
        sample = handle.read(sample_size)
    return b"\0" in sample


@dataclass(slots=True)
class FileLoader:
    valid_extensions: tuple[str, ...] = (".py", ".md", ".txt", ".js", ".ts")

    def scan_directory(self, root_path: str) -> list[str]:
        root = Path(root_path)
        files: list[str] = []
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.suffix and path.suffix.lower() not in self.valid_extensions:
                continue
            if _is_binary_file(path):
                continue
            files.append(str(path))
        return files

    def determine_tier(self, filepath: str) -> str:
        path = filepath.lower()
        if "/src/" in path or "/lib/" in path:
            return "GOLD"
        if "/tests/" in path or "/examples/" in path or "/docs/" in path:
            return "SILVER"
        return "JUNK"


@dataclass(slots=True)
class CodeSplitter:
    window_size: int = 500
    overlap: int = 100
    tier_resolver: Callable[[str], str] | None = None

    def _tokenize_lines(self, raw_text: str) -> list[tuple[str, int]]:
        tokens_with_lines: list[tuple[str, int]] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            for token in line.split():
                tokens_with_lines.append((token, line_number))
        return tokens_with_lines

    def _build_chunk(
        self,
        filepath: str,
        tokens_with_lines: list[tuple[str, int]],
        start_index: int,
        end_index: int,
        quality_tier: str,
    ) -> CodeChunk:
        chunk_tokens = tokens_with_lines[start_index:end_index]
        content = " ".join(token for token, _ in chunk_tokens)
        start_line = chunk_tokens[0][1] if chunk_tokens else 1
        return CodeChunk(
            filepath=filepath,
            content=content,
            quality_tier=quality_tier,
            start_line=start_line,
        )

    def chunk_file(self, filepath: str, raw_text: str) -> list[CodeChunk]:
        resolver = self.tier_resolver or FileLoader().determine_tier
        quality_tier = resolver(filepath)
        tokens_with_lines = self._tokenize_lines(raw_text)
        chunks: list[CodeChunk] = []
        if not tokens_with_lines:
            chunks.append(
                CodeChunk(
                    filepath=filepath,
                    content="",
                    quality_tier=quality_tier,
                    start_line=1,
                )
            )
            return chunks

        step = max(self.window_size - self.overlap, 1)
        for start in range(0, len(tokens_with_lines), step):
            end = min(start + self.window_size, len(tokens_with_lines))
            chunks.append(
                self._build_chunk(
                    filepath=filepath,
                    tokens_with_lines=tokens_with_lines,
                    start_index=start,
                    end_index=end,
                    quality_tier=quality_tier,
                )
            )
            if end == len(tokens_with_lines):
                break
        return chunks
