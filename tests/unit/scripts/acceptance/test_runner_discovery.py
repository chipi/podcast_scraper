"""Unit tests for acceptance runner fast matrix / path discovery (no subprocess runs)."""

import importlib.util
import tempfile
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
    """``MAIN_ACCEPTANCE_CONFIG.yaml`` + ``--from-fast-stems`` helpers."""

    def test_load_fast_matrix_ids_includes_dev_and_cloud_rows(self):
        mod = _load_run_acceptance_module()
        ids = mod.load_fast_matrix_ids()
        self.assertEqual(
            ids,
            {
                "fast_dev_single",
                "fast_cloud_balanced_single",
                "fast_cloud_balanced_multi",
                "fast_cloud_quality_single",
            },
        )

    def test_load_fast_config_stems_matches_matrix_ids(self):
        mod = _load_run_acceptance_module()
        stems = mod.load_fast_config_stems()
        self.assertEqual(stems, mod.load_fast_matrix_ids())

    def test_materialize_fast_matrix_writes_materialized_id_yaml(self):
        mod = _load_run_acceptance_module()
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session_test"
            session_dir.mkdir()
            paths = mod.materialize_fast_matrix_configs(session_dir)
            self.assertEqual(len(paths), 4)
            for p in paths:
                self.assertEqual(p.parent.name, "materialized")
                self.assertTrue(p.name.endswith(".yaml"))
                text = p.read_text(encoding="utf-8")
                self.assertIn("profile:", text)

    def test_resolve_yaml_paths_finds_example_config_when_present(self):
        mod = _load_run_acceptance_module()
        paths = mod.resolve_yaml_paths_from_stems({"config.example"})
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "config.example.yaml")
        self.assertTrue(paths[0].is_file())
