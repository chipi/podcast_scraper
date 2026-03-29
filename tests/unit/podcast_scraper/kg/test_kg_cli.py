"""Unit tests for kg CLI parsing and main() dispatch."""

import unittest

from podcast_scraper import cli


class TestKgSubcommand(unittest.TestCase):
    """Test kg subcommand parsing and execution."""

    def test_parse_args_kg_validate(self) -> None:
        args = cli.parse_args(["kg", "validate", "tests/fixtures/kg"])
        self.assertEqual(args.command, "kg")
        self.assertEqual(args.kg_subcommand, "validate")
        self.assertEqual(args.paths, ["tests/fixtures/kg"])

    def test_parse_args_kg_inspect(self) -> None:
        args = cli.parse_args(
            ["kg", "inspect", "--episode-path", "tests/fixtures/kg/minimal.kg.json"]
        )
        self.assertEqual(args.kg_subcommand, "inspect")
        self.assertTrue(args.episode_path.endswith("minimal.kg.json"))

    def test_main_kg_validate_fixture(self) -> None:
        code = cli.main(["kg", "validate", "tests/fixtures/kg", "--strict"])
        self.assertEqual(code, 0)

    def test_main_kg_inspect_fixture(self) -> None:
        code = cli.main(
            [
                "kg",
                "inspect",
                "--episode-path",
                "tests/fixtures/kg/minimal.kg.json",
                "--format",
                "json",
            ]
        )
        self.assertEqual(code, 0)
