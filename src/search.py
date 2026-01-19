from typing import Dict, Optional

class CodeSearchEngine:
    def query(self, user_question: str, filters: Optional[Dict] = None) -> str:
        """
        Convert user question to vector.
        Call FaissManager.search.
        (Optional) Apply quality_tier filtering logic on results.
        Format the output for the LLM (Prompt construction).
        """
        pass
