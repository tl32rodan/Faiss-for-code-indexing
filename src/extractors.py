from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.models import SymbolChunk, compute_code_hash


class BaseExtractor(ABC):
    @abstractmethod
    def extract_symbols(self, filepath: str, content: str) -> List[SymbolChunk]:
        raise NotImplementedError

    def parse_file(self, filepath: str) -> List[SymbolChunk]:
        content = Path(filepath).read_text(encoding="utf-8")
        return self.extract_symbols(filepath, content)


class PythonAstExtractor(BaseExtractor):
    def __init__(self, source_root: str) -> None:
        self.source_root = Path(source_root)

    def extract_symbols(self, filepath: str, content: str) -> List[SymbolChunk]:
        tree = ast.parse(content)
        module_path = self._module_path(Path(filepath))
        visitor = _PythonSymbolVisitor(content, filepath, module_path)
        visitor.visit(tree)
        return visitor.symbols

    def _module_path(self, filepath: Path) -> str:
        resolved_root = self.source_root.resolve()
        resolved_file = filepath.resolve()
        rel_path = resolved_file.relative_to(resolved_root)
        parts = list(rel_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return rel_path.stem
        return ".".join(parts)


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self, content: str, filepath: str, module_path: str) -> None:
        self.symbols: List[SymbolChunk] = []
        self._content = content
        self._lines = content.splitlines()
        self._filepath = filepath
        self._module_path = module_path
        self._class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_symbol(node, node.name, "class")
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)
        self.generic_visit(node)

    def _record_function(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if self._class_stack:
                name = ".".join([*self._class_stack, node.name])
                kind = "method"
            else:
                name = node.name
                kind = "function"
            self._record_symbol(node, name, kind)

    def _record_symbol(self, node: ast.AST, name: str, kind: str) -> None:
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        segment = ast.get_source_segment(self._content, node)
        if segment is None:
            segment = self._fallback_segment(start_line, end_line)
        symbol_id = f"{self._module_path}:{name}"
        code_hash = compute_code_hash(segment)
        self.symbols.append(
            SymbolChunk(
                symbol_id=symbol_id,
                filepath=self._filepath,
                symbol_name=name,
                symbol_kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=segment,
                code_hash=code_hash,
            )
        )

    def _fallback_segment(self, start_line: int, end_line: int) -> str:
        return "\n".join(self._lines[start_line - 1 : end_line])
