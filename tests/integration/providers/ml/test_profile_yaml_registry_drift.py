"""Drift detection: profile YAML routing must match the declared registry preset.

When a YAML under ``config/profiles/`` declares a top-level ``profile:`` field
that names a known ``_PROFILE_PRESETS`` entry, the YAML's routing fields must
agree with what ``resolve_profile_to_settings(name, ...)`` returns. Otherwise
the eval reports document a decision the runtime never adopts.

Today most profile YAMLs do NOT declare ``profile:`` (they predate #907's
runtime wiring). Those YAMLs are skipped — adding ``profile:`` to them is the
opt-in for drift-checking.

Adding a new profile preset to the registry without a matching YAML
declaration is fine and won't trip this test. The test only fires once a YAML
has explicitly opted in by declaring ``profile:``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from podcast_scraper.providers.ml.model_registry import (
    _PROFILE_PRESETS,
    resolve_profile_to_settings,
)

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

_PROFILE_DIR = Path(__file__).resolve().parents[4] / "config" / "profiles"

_ROUTING_FIELDS = (
    "transcription_provider",
    "summary_provider",
    "summary_model",
    "kg_extraction_source",
    "kg_max_topics",
    "kg_max_entities",
    "speaker_detector_provider",
    "ner_model",
    "gi_insight_source",
    "gi_max_insights",
    "gi_require_grounding",
    "gil_evidence_quote_mode",
    "gil_evidence_nli_mode",
)


def _opted_in_yamls() -> List[Tuple[Path, Dict[str, Any]]]:
    """Yield (path, parsed_dict) for every YAML that declares a ``profile:`` field."""
    opted_in: List[Tuple[Path, Dict[str, Any]]] = []
    if not _PROFILE_DIR.is_dir():
        return opted_in
    for path in sorted(_PROFILE_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and "profile" in data:
            opted_in.append((path, data))
    return opted_in


_OPTED_IN = _opted_in_yamls()


@pytest.mark.parametrize(
    "path,data",
    _OPTED_IN,
    ids=[p.name for p, _ in _OPTED_IN] or ["no-opted-in-yamls"],
)
def test_profile_yaml_matches_registry_preset(path: Path, data: Dict[str, Any]) -> None:
    """Every routing field present in the YAML must equal the registry's value."""
    profile_name = data["profile"]
    assert profile_name in _PROFILE_PRESETS, (
        f"{path.name} declares profile={profile_name!r}, which is not in the "
        f"registry. Known presets: {sorted(_PROFILE_PRESETS)}"
    )
    expected = resolve_profile_to_settings(profile_name, dgx_tailnet_host="drift-check.example")
    for field in _ROUTING_FIELDS:
        if field not in data:
            continue  # YAML omits this field — registry default applies, no drift.
        # nosec B608 — not SQL; just a multi-line f-string assert message.
        msg = (
            f"{path.name}: field {field!r} = {data[field]!r} disagrees with "  # nosec B608
            f"registry preset {profile_name!r} "
            f"(expected {expected.get(field)!r}). "
            f"Update either the YAML or the registry preset "
            f"(with new research_ref) to bring them back in sync."
        )
        assert data[field] == expected.get(field), msg


def test_no_opted_in_yamls_is_self_documenting() -> None:
    """Smoke test that surfaces an informative message if no YAML has opted in yet.

    Without this, the parametrized test above generates a single ``no-opted-in-yamls``
    case that silently passes — making the drift check look broken at a glance.
    """
    if not _OPTED_IN:
        pytest.skip(
            "No profile YAML declares `profile: <name>` yet — drift check is a no-op. "
            "Opt in by adding `profile: <preset_name>` to a YAML under config/profiles/."
        )
    # When at least one YAML opts in, this test passes trivially.
    assert _OPTED_IN
