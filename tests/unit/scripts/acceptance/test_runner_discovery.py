"""Unit tests for acceptance runner stem / path discovery (no subprocess runs)."""

import importlib.util
import unittest
from pathlib import Path


def _load_run_acceptance_module():
    # tests/unit/scripts/acceptance/ → repo root is parents[4]
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "scripts" / "acceptance" / "run_acceptance_tests.py"
    spec = importlib.util.spec_from_file_location("_run_acceptance_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAcceptanceRunnerDiscovery(unittest.TestCase):
    """``--from-fast-stems`` helpers (CI fixture matrix)."""

    def test_load_fast_config_stems_includes_ml_dev(self):
        mod = _load_run_acceptance_module()
        stems = mod.load_fast_config_stems()
        self.assertIn("acceptance_planet_money_ml_dev", stems)

    def test_load_fast_config_stems_includes_multi_feed(self):
        mod = _load_run_acceptance_module()
        stems = mod.load_fast_config_stems()
        self.assertIn("acceptance_multi_feed_planet_money_journal_openai", stems)

    def test_resolve_yaml_paths_multi_feed_finds_under_acceptance_dir(self):
        mod = _load_run_acceptance_module()
        paths = mod.resolve_yaml_paths_from_stems(
            {"acceptance_multi_feed_planet_money_journal_openai"}
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            paths[0].name,
            "acceptance_multi_feed_planet_money_journal_openai.yaml",
        )
        self.assertTrue(paths[0].is_file())
        self.assertEqual(paths[0].parent.name, "acceptance")
