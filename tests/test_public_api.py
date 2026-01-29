import unittest

import src
from src import core, engine
from src.core import VectorDocument
from src.engine import FaissEngine, IndexRegistry


class TestPublicApi(unittest.TestCase):
    def test_package_exports(self) -> None:
        self.assertIs(src.VectorDocument, VectorDocument)
        self.assertIs(src.FaissEngine, FaissEngine)
        self.assertIs(src.IndexRegistry, IndexRegistry)

    def test_module_exports(self) -> None:
        self.assertIn("VectorDocument", core.__all__)
        self.assertIn("FaissEngine", engine.__all__)
        self.assertIn("IndexRegistry", engine.__all__)


if __name__ == "__main__":
    unittest.main()
