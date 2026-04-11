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

    def test_load_fast_config_stems_includes_single_feed_ml_fixture(self):
        """Fast stems include single-feed ML; multi_ml is omitted (see FAST_CONFIGS.txt)."""
        mod = _load_run_acceptance_module()
        stems = mod.load_fast_config_stems()
        self.assertIn("sample_acceptance_e2e_fixture_single", stems)
        self.assertNotIn("sample_acceptance_e2e_fixture_multi_ml", stems)

    def test_load_fast_config_stems_includes_multi_feed_fixture(self):
        mod = _load_run_acceptance_module()
        stems = mod.load_fast_config_stems()
        self.assertIn("sample_acceptance_e2e_fixture_multi_openai", stems)

    def test_resolve_yaml_paths_finds_tracked_sample_under_acceptance_dir(self):
        mod = _load_run_acceptance_module()
        paths = mod.resolve_yaml_paths_from_stems({"sample_acceptance_e2e_fixture_single"})
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            paths[0].name,
            "sample_acceptance_e2e_fixture_single.yaml",
        )
        self.assertTrue(paths[0].is_file())
        self.assertEqual(paths[0].parent.name, "acceptance")
