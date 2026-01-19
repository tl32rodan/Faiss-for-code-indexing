from enum import Enum
from pydantic import BaseModel, Field

class QualityTier(str, Enum):
    GOLD = 'GOLD'
    SILVER = 'SILVER'
    JUNK = 'JUNK'

class CodeChunk(BaseModel):
    """
    Core data class for code chunks.
    """
    id: str = Field(..., description="Unique identifier (Hash of filepath + content)")
    filepath: str = Field(..., description="Absolute path to the file")
    content: str = Field(..., description="The raw code snippet")
    meta_intent: str = Field(default="", description="Optional. Human-annotated intent or notes")
    quality_tier: QualityTier = Field(..., description="'GOLD', 'SILVER', or 'JUNK'")
    start_line: int = Field(..., description="Starting line number in the source file")

    def get_embedding_content(self) -> str:
        """
        Concatenate meta info with code for the embedding model.
        Format: "# File: {filepath}\n# Tier: {quality_tier}\n# Intent: {meta_intent}\n{content}"
        """
        return f"# File: {self.filepath}\n# Tier: {self.quality_tier.value}\n# Intent: {self.meta_intent}\n{self.content}"
