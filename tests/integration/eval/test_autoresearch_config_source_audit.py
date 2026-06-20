"""#1033 CI regression — every autoresearch GI/KG eval config MUST use ``provider`` source.

History: pre-#1033, the autoresearch eval configs (``data/eval/configs/``) used
``kg_extraction_source: summary_bullets`` and ``gi_insight_source: summary_bullets``,
which routes structured extraction through already-distilled summary bullets
(whose generating prompt explicitly strips speaker names — see
``ollama/qwen3.5_35b/summarization/system_v1.j2``). Production profiles
(``config/profiles/prod_dgx_*.yaml``) always used ``provider``. The eval and
prod code paths diverged, so #1016 Round 3 / #1022 Cell F GI+KG rankings
were measured under a non-production pipeline.

This test guards both directions:

1. Every ``data/eval/configs/{gi,kg}_autoresearch_*.yaml`` config has
   ``provider`` as its source — no drift back to ``summary_bullets``.
2. Production profiles (``config/profiles/prod_dgx_*.yaml`` +
   ``eval_default.yaml`` + ``cloud_*.yaml`` + ``airgapped*.yaml``) keep
   ``provider`` as the source.

The test recognises both field forms — the short ``kg_extraction_src`` /
``gi_insight_src`` (used in eval configs) and the canonical long form
``kg_extraction_source`` / ``gi_insight_source`` (used in profiles, since
the Config Pydantic alias is the long form).

Refs: #1033 (the audit issue), #112 (the experimental finding that surfaced
the gap), #1016 § 6b (original mis-attribution of the 0% entity coverage).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_CONFIGS_DIR = REPO_ROOT / "data" / "eval" / "configs"
PROFILES_DIR = REPO_ROOT / "config" / "profiles"

_KG_FIELDS = ("kg_extraction_source", "kg_extraction_src")
_GI_FIELDS = ("gi_insight_source", "gi_insight_src")

# Acceptable values per the Config Pydantic Literal:
#   ``Literal["stub", "summary_bullets", "provider"]``
# The audit rule: nothing in the autoresearch eval-config space or production
# profiles may use ``summary_bullets``. ``stub`` is fine for non-LLM paths.
_FORBIDDEN_VALUES = {"summary_bullets"}


def _load_yaml(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except yaml.YAMLError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _read_nested_field(doc: dict | None, *field_names: str) -> tuple[str, str] | None:
    """Walk a YAML doc dict looking for any of ``field_names`` anywhere in the tree.

    Returns ``(found_field_name, value)`` on first hit, else ``None``.
    """
    if not isinstance(doc, dict):
        return None
    for key, value in doc.items():
        if key in field_names and isinstance(value, str):
            return key, value
        if isinstance(value, dict):
            nested = _read_nested_field(value, *field_names)
            if nested is not None:
                return nested
    return None


_AUTORESEARCH_GI_KG_RE = re.compile(r"^(?:gi|kg)_autoresearch_prompt_.+\.yaml$")


def _autoresearch_eval_configs() -> list[Path]:
    """All ``gi_/kg_autoresearch_*.yaml`` under ``data/eval/configs/``."""
    if not EVAL_CONFIGS_DIR.is_dir():
        return []
    return sorted(p for p in EVAL_CONFIGS_DIR.iterdir() if _AUTORESEARCH_GI_KG_RE.match(p.name))


def _production_profiles() -> list[Path]:
    """Production-shipped profiles. Excludes ``local_*`` since those follow a
    different lifecycle and may legitimately use Ollama-only modes."""
    if not PROFILES_DIR.is_dir():
        return []
    eligible = []
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        name = path.name
        if name.startswith("local_"):
            continue
        eligible.append(path)
    return eligible


@pytest.mark.integration
class TestAutoresearchEvalConfigSourceAudit:
    """#1033 — autoresearch eval configs must use ``provider`` source."""

    def test_at_least_one_eval_config_exists(self) -> None:
        configs = _autoresearch_eval_configs()
        assert configs, (
            "Expected at least one gi_/kg_autoresearch_*.yaml config under "
            f"{EVAL_CONFIGS_DIR}. If the eval configs moved, update this test."
        )

    @pytest.mark.parametrize(
        "config_path",
        _autoresearch_eval_configs(),
        ids=lambda p: p.name,
    )
    def test_eval_config_uses_provider_source_for_its_stage(self, config_path: Path) -> None:
        """Each autoresearch eval config sets the relevant source to ``provider``."""
        doc = _load_yaml(config_path)
        assert isinstance(
            doc, dict
        ), f"{config_path.name}: failed to parse YAML or top-level isn't a mapping."

        if config_path.name.startswith("kg_"):
            relevant_fields = _KG_FIELDS
            stage_name = "KG"
        elif config_path.name.startswith("gi_"):
            relevant_fields = _GI_FIELDS
            stage_name = "GI"
        else:
            pytest.fail(f"{config_path.name}: unexpected prefix; expected kg_ or gi_")

        found = _read_nested_field(doc, *relevant_fields)
        if found is None:
            # KG without an explicit source falls back to summary_bullets per
            # the Config default — that's the bug #1033 closes. Fail loudly.
            # GI default is 'stub' (non-LLM); allow absence.
            if stage_name == "KG":
                pytest.fail(
                    f"{config_path.name}: KG eval config doesn't set "
                    f"kg_extraction_src/kg_extraction_source. Pydantic default is "
                    f"'summary_bullets' which #1033 forbids for eval configs."
                )
            return

        field_name, value = found
        assert value not in _FORBIDDEN_VALUES, (
            f"{config_path.name}: {field_name}: {value!r} is forbidden per #1033. "
            f"Eval configs must use 'provider' for {stage_name} extraction, not "
            f"'summary_bullets' (which routes through the name-stripping summary "
            f"prompt). See docs/wip/EVAL_112_ENTITY_FOCUSED_KG_2026-06-19.md."
        )

    @pytest.mark.parametrize(
        "profile_path",
        _production_profiles(),
        ids=lambda p: p.name,
    )
    def test_profile_does_not_revert_to_summary_bullets(self, profile_path: Path) -> None:
        """Production-tier profiles must not regress to ``summary_bullets``."""
        doc = _load_yaml(profile_path)
        if not isinstance(doc, dict):
            return
        for fields in (_KG_FIELDS, _GI_FIELDS):
            found = _read_nested_field(doc, *fields)
            if found is None:
                continue
            field_name, value = found
            assert value not in _FORBIDDEN_VALUES, (
                f"{profile_path.name}: {field_name}: {value!r} is forbidden per #1033. "
                f"Production profiles must use 'provider' for KG/GI extraction."
            )
