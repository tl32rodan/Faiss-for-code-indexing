from typing import List
from .models import CodeChunk, QualityTier

class FileLoader:
    def scan_directory(self, root_path: str) -> List[str]:
        """
        Recursively find valid code files.
        Implement a filter to ignore binary files or hidden folders.
        """
        pass

    def determine_tier(self, filepath: str) -> QualityTier:
        """
        Rule-based logic to assign 'GOLD'/'SILVER'/'JUNK'.
        """
        pass

class CodeSplitter:
    def chunk_file(self, filepath: str, raw_text: str) -> List[CodeChunk]:
        """
        Implement a sliding window approach (e.g., 500 tokens window, 100 tokens overlap).

        Crucial: Ensure every chunk inherits the filepath and tier of its parent file.
        (Note: Tier determination logic might need to be invoked here or passed in).
        """
        pass
