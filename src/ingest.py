from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import yaml

from src.core.knowledge_unit import KnowledgeUnit
from src.extractors import get_extractor_for_path
from src.models import SymbolChunk


def _is_binary_file(path: Path, sample_size: int = 1024) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(sample_size)
    except (FileNotFoundError, OSError):
        return True
    return b"\0" in sample


@dataclass
class FileLoader:
    valid_extensions: tuple[str, ...] = (".py", ".md", ".txt", ".js", ".ts", ".pl")
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


class SidecarManager:
    def __init__(self, knowledge_root: str, source_root: str) -> None:
        self.knowledge_root = Path(knowledge_root)
        self.source_root = Path(source_root)

    def sidecar_path_for_source(self, source_path: str) -> Path:
        rel_path = self._relative_source_path(source_path)
        return self.knowledge_root / f"{rel_path.as_posix()}.meta.yaml"

    def load_sidecar(self, source_path: str) -> Optional[Dict[str, object]]:
        sidecar_path = self.sidecar_path_for_source(source_path)
        if not sidecar_path.exists():
            return None
        payload = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload
        raise ValueError("Sidecar content must be a mapping.")

    def save_sidecar(self, source_path: str, payload: Dict[str, object]) -> None:
        sidecar_path = self.sidecar_path_for_source(source_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def ensure_sidecar(
        self, source_path: str, extracted: List[SymbolChunk]
    ) -> Dict[str, object]:
        existing = self.load_sidecar(source_path)
        if existing is not None:
            return existing
        sidecar = self._build_skeleton(source_path, extracted)
        self.save_sidecar(source_path, sidecar)
        return sidecar

    def _build_skeleton(
        self, source_path: str, extracted: List[SymbolChunk]
    ) -> Dict[str, object]:
        file_uid = self._relative_source_path(source_path).as_posix()
        chunks = [
            {
                "id_suffix": f"::{symbol.symbol_name}",
                "intent": "",
                "relations": [],
            }
            for symbol in extracted
        ]
        return {
            "file_uid": file_uid,
            "global_intent": "",
            "chunks": chunks,
        }

    def _relative_source_path(self, source_path: str) -> Path:
        source_path_obj = Path(source_path)
        try:
            return source_path_obj.relative_to(self.source_root)
        except ValueError:
            resolved_root = self.source_root.resolve()
            resolved_source = source_path_obj.resolve()
            try:
                return resolved_source.relative_to(resolved_root)
            except ValueError:
                rel_path = source_path_obj
                if rel_path.is_absolute():
                    parts = [part for part in rel_path.parts if part != rel_path.anchor]
                    rel_path = Path(*parts)
                return rel_path


class Ingestor:
    def __init__(
        self,
        source_root: str,
        knowledge_root: str,
        loader: FileLoader,
        sidecar_manager: SidecarManager,
    ) -> None:
        self.source_root = source_root
        self.knowledge_root = knowledge_root
        self.loader = loader
        self.sidecar_manager = sidecar_manager

    def ingest(self) -> List[KnowledgeUnit]:
        units: List[KnowledgeUnit] = []
        for path in self.loader.scan_directory(self.source_root):
            units.extend(self.ingest_file(path))
        return units

    def ingest_file(self, source_path: str) -> List[KnowledgeUnit]:
        extractor = get_extractor_for_path(self.source_root, source_path)
        content = Path(source_path).read_text(encoding="utf-8")
        symbols = extractor.extract_symbols(source_path, content)
        sidecar = self.sidecar_manager.ensure_sidecar(source_path, symbols)
        return self._units_from_sidecar(symbols, sidecar)

    def _units_from_sidecar(
        self, extracted: Iterable[SymbolChunk], sidecar: Dict[str, object]
    ) -> List[KnowledgeUnit]:
        by_name = {symbol.symbol_name: symbol for symbol in extracted}
        global_intent = str(sidecar.get("global_intent", ""))
        chunks_payload = sidecar.get("chunks", [])
        units: List[KnowledgeUnit] = []
        for entry in chunks_payload:
            if not isinstance(entry, dict):
                continue
            suffix = str(entry.get("id_suffix", ""))
            symbol_name = suffix.split("::")[-1] if suffix else ""
            symbol = by_name.get(symbol_name)
            if symbol is None:
                continue
            intent = str(entry.get("intent", ""))
            relations_value = entry.get("relations", [])
            relations = relations_value if isinstance(relations_value, list) else []
            unit = KnowledgeUnit(
                uid=symbol.symbol_id,
                content=symbol.content,
                source_type="code",
                metadata={
                    "language": Path(symbol.filepath).suffix.lstrip("."),
                    "filepath": symbol.filepath,
                    "symbol_kind": symbol.symbol_kind,
                    "symbol_name": symbol.symbol_name,
                    "global_intent": global_intent,
                    "intent": intent,
                },
                related_ids=[str(rel) for rel in relations],
            )
            units.append(unit)
        return units
