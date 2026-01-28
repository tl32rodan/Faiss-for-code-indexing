import unittest

from src.agent.router import KeywordRouter


class TestRouter(unittest.TestCase):
    def test_keyword_router_routes_to_issue_index(self) -> None:
        router = KeywordRouter()
        indices = router.route("Is there an issue with the login bug?")

        self.assertEqual(indices, ["issues"])

    def test_keyword_router_defaults_to_source_code(self) -> None:
        router = KeywordRouter()
        indices = router.route("How does authentication work?")

        self.assertEqual(indices, ["source_code"])


if __name__ == "__main__":
    unittest.main()
