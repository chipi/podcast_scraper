"""Unit tests for scripts/registry/promote_baseline.py."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts/registry to path so we can import promote_baseline
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_REGISTRY = _PROJECT_ROOT / "scripts" / "registry"
if str(_SCRIPTS_REGISTRY) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_REGISTRY))

from promote_baseline import promote_baseline  # noqa: E402


@pytest.mark.unit
class TestPromoteBaseline:
    def test_promotes_dev_baseline_into_temp_registry_file(self):
        baseline_dir = (
            _PROJECT_ROOT / "data" / "eval" / "baselines" / "baseline_ml_dev_authority_smoke_v1"
        )
        assert baseline_dir.exists()

        # Copy the real registry to a temp file so we don't modify the repo file.
        from podcast_scraper.providers.ml import model_registry as mr

        src_registry_path = Path(mr.__file__).resolve()
        with tempfile.TemporaryDirectory() as td:
            tmp_registry_path = Path(td) / "model_registry.py"
            shutil.copy2(src_registry_path, tmp_registry_path)

            mode = promote_baseline(
                baseline_dir=baseline_dir,
                mode_id="ml_small_authority_test",
                registry_path=tmp_registry_path,
                baseline_id="baseline_ml_dev_authority_smoke_v1",
            )

            assert mode.mode_id == "ml_small_authority_test"
            assert mode.map_model == "bart-small"
            assert mode.reduce_model == "long-fast"

            updated = tmp_registry_path.read_text(encoding="utf-8")
            assert "BEGIN MODE REGISTRY" in updated
            assert "END MODE REGISTRY" in updated
            assert "'ml_small_authority_test': ModeConfiguration(" in updated
            assert "map_model='bart-small'" in updated
            assert "reduce_model='long-fast'" in updated
            assert "promoted_from='baseline_ml_dev_authority_smoke_v1'" in updated

    def test_rejects_duplicate_mode_id(self):
        baseline_dir = (
            _PROJECT_ROOT / "data" / "eval" / "baselines" / "baseline_ml_dev_authority_smoke_v1"
        )
        from podcast_scraper.providers.ml import model_registry as mr

        src_registry_path = Path(mr.__file__).resolve()
        with tempfile.TemporaryDirectory() as td:
            tmp_registry_path = Path(td) / "model_registry.py"
            shutil.copy2(src_registry_path, tmp_registry_path)

            promote_baseline(
                baseline_dir=baseline_dir,
                mode_id="dup_mode",
                registry_path=tmp_registry_path,
                baseline_id="baseline_ml_dev_authority_smoke_v1",
            )

            with pytest.raises(ValueError, match="already exists"):
                promote_baseline(
                    baseline_dir=baseline_dir,
                    mode_id="dup_mode",
                    registry_path=tmp_registry_path,
                    baseline_id="baseline_ml_dev_authority_smoke_v1",
                )
