"""Every config knob an enricher READS must be DECLARED in its manifest.

The composed schema behind ``PUT /api/enrichment/config`` sets ``additionalProperties: false`` per
enricher, built from ``manifest.config_schema``. An enricher that reads ``config.get("x")`` without
declaring ``x`` produces a silent trap: the value works when set in the corpus YAML by hand, and
the API **400s** on any attempt to set the same value — so the documented way to configure it is
the one that fails.

Found by auditing after #1930/#1928 added knobs. Three enrichers were already affected before that
work: ``topic_theme_clusters`` (``min_pair_episode_count``, ``merge_threshold``,
``super_theme_target``) and ``guest_coappearance`` (``community_min_pair``) — all read, none
declared. This test exists so the next knob cannot repeat it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from podcast_scraper.enrichment.enrichers import register_deterministic_enrichers
from podcast_scraper.enrichment.registry import EnricherRegistry

pytestmark = pytest.mark.unit

#: Knobs read for test injection rather than operator configuration. ``now`` is documented in
#: ``temporal_velocity._now_utc`` as "for testability"; declaring it would advertise a clock
#: override on the operator config surface, which is worse than leaving it undeclared.
_TEST_ONLY_KNOBS = {"now"}

_ENRICHERS_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "podcast_scraper"
    / "enrichment"
    / "enrichers"
)


def _registry() -> EnricherRegistry:
    reg = EnricherRegistry()
    register_deterministic_enrichers(reg)
    return reg


def _knobs_read(enricher_id: str) -> set[str]:
    src = _ENRICHERS_DIR / f"{enricher_id}.py"
    if not src.is_file():
        return set()
    return set(re.findall(r'config\.get\(\s*"([a-z_]+)"', src.read_text(encoding="utf-8")))


def _knobs_declared(manifest: object) -> set[str]:
    cs = getattr(manifest, "config_schema", None) or {}
    props = cs.get("properties") if isinstance(cs, dict) else None
    return set(props.keys()) if isinstance(props, dict) else set()


@pytest.mark.parametrize("enricher_id", sorted(_registry().all_ids()))
def test_every_knob_read_is_declared(enricher_id: str) -> None:
    """THE regression: reading an undeclared knob makes the API reject its own config."""
    manifest = _registry().get(enricher_id).manifest
    missing = _knobs_read(enricher_id) - _knobs_declared(manifest) - _TEST_ONLY_KNOBS
    assert not missing, (
        f"{enricher_id} reads config knob(s) {sorted(missing)} that its manifest does not "
        "declare. PUT /api/enrichment/config validates against a composed schema with "
        "additionalProperties:false, so an operator setting these gets a 400 while the same "
        "value works if hand-written into the corpus YAML. Add them to manifest.config_schema."
    )


@pytest.mark.parametrize("enricher_id", sorted(_registry().all_ids()))
def test_declared_knobs_are_actually_read(enricher_id: str) -> None:
    """The mirror: a declared knob nobody reads is a documented no-op.

    Softer than the above — a knob may legitimately be consumed by a helper module rather than
    inline — so this only asserts the declaration is not obviously dead by checking the id
    appears somewhere in the enricher's own source.
    """
    manifest = _registry().get(enricher_id).manifest
    src = _ENRICHERS_DIR / f"{enricher_id}.py"
    if not src.is_file():
        pytest.skip(f"{enricher_id} has no single-module source")
    text = src.read_text(encoding="utf-8")
    dead = [k for k in _knobs_declared(manifest) if k not in text]
    assert not dead, (
        f"{enricher_id} declares knob(s) {sorted(dead)} that never appear in its source — "
        "an operator can set them and nothing happens."
    )


def test_the_audit_covers_every_registered_enricher() -> None:
    """Guard the guard: if registration changes shape, this file must not silently pass."""
    ids = sorted(_registry().all_ids())
    assert len(ids) >= 7, f"expected the deterministic enricher set, got {ids}"
    for eid in ids:
        assert (
            _ENRICHERS_DIR / f"{eid}.py"
        ).is_file(), (
            f"{eid} has no module at the audited path — the knob audit would skip it silently"
        )
