from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


def _is_binary_file(path: Path, sample_size: int = 1024) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(sample_size)
    except (FileNotFoundError, OSError):
        return True
    return b"\0" in sample
@dataclass
class FileLoader:
    valid_extensions: tuple[str, ...] = (".py", ".md", ".txt", ".js", ".ts")
    follow_symlinks: bool = False

    def scan_directory(self, root_path: str) -> List[str]:
        files: List[str] = []
        seen_dirs: Set[str] = set()
        for current_root, dirs, filenames in os.walk(
            root_path, followlinks=self.follow_symlinks
        ):
            real_root = os.path.realpath(current_root)
            if real_root in seen_dirs:
                dirs[:] = []
                continue
            seen_dirs.add(real_root)
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = Path(current_root) / filename
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
