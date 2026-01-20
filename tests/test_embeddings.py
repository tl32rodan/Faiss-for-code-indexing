import unittest
from typing import List

from src.embeddings import BaseEmbeddingModel


class DummyEmbedding(BaseEmbeddingModel):
    def encode(self, text: List[str], **kwargs: object) -> List[List[float]]:
        return [[float(len(item))] for item in text]

    def get_sentence_embedding_dimension(self) -> int:
        return 1


class TestEmbeddings(unittest.TestCase):
    def test_base_embedding_contract(self) -> None:
        model = DummyEmbedding()
        self.assertEqual(model.encode(["hi"]), [[2.0]])
        self.assertEqual(model.get_sentence_embedding_dimension(), 1)


if __name__ == "__main__":
    unittest.main()
