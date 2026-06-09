"""Unit tests for cross-episode entity canonicalization (kg/entity_clusters.py, #852)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.kg.entity_clusters import (
    _are_xep_variants,
    build_entity_canonical_map,
    build_entity_id_map,
    collect_entity_candidates,
    EntityCandidate,
    id_map_from_clusters_payload,
)

pytestmark = pytest.mark.unit


def _cand(cid, kind, name, eps, shows):
    return EntityCandidate(id=cid, kind=kind, name=name, episodes=set(eps), shows=set(shows))


# --- variant rule: MERGE real drift ---------------------------------------------


@pytest.mark.parametrize(
    "a,b,kind",
    [
        ("Cargil", "Cargill", "org"),
        ("Data Bricks", "Databricks", "org"),
        ("Chat GPT", "ChatGPT", "org"),
        ("Byrne Hobart", "Burne Hobart", "person"),
        ("Tracy Alloway", "Tracey Alloway", "person"),
        ("Donald Mackenzie", "Donald McKenzie", "person"),
        ("David Shor", "David Shore", "person"),
    ],
)
def test_xep_variants_merge_real_drift(a, b, kind):
    assert _are_xep_variants(a, b, kind) is True


# --- variant rule: REJECT the landmines -----------------------------------------


@pytest.mark.parametrize(
    "a,b,kind",
    [
        ("UPS", "USPS", "org"),  # acronyms
        ("Claude", "Claude 3", "org"),  # version token
        ("GPT-4", "GPT-4o", "org"),  # version-ish
        ("Bloomberg Audio Studios", "Bloomberg Media Studios", "org"),  # distinct content word
        ("Sam Altman", "Tim Cook", "person"),  # different people
        ("John Smith", "Jane Smith", "person"),  # different first names
    ],
)
def test_xep_variants_reject_landmines(a, b, kind):
    assert _are_xep_variants(a, b, kind) is False


# --- #904 predicate redesign: nickname + token-count tolerance ------------------


@pytest.mark.parametrize(
    "a,b",
    [
        # Nickname class (same surname, nickname/full-name first)
        ("Mike Selig", "Michael Selig"),
        ("Nicholas Snyder", "Nick Snyder"),
        ("Elizabeth Reid", "Liz Reid"),
        ("Emmanuel Roman", "Manny Roman"),
        ("Rich Clarida", "Richard Clarida"),
        ("Rob Goldstein", "Robert Goldstein"),
        # Initial-vs-full first name (J. → Jerome)
        ("J. Powell", "Jerome Powell"),
        # Title prefix on one side only (Dr / Ayatollah / President)
        ("Dr. Elena Fischer", "Elena Fischer"),
        ("Ayatollah Ali Khamenei", "Ali Khamenei"),
        ("President Trump", "Donald Trump"),
        # Family-only reference (last-token match)
        ("Mark Carney", "Carney"),
        ("Donald Trump", "Trump"),
    ],
)
def test_xep_variants_904_nickname_and_token_count_merge(a, b):
    """#904 — predicate redesign covers nickname class + token-count mismatches."""
    assert _are_xep_variants(a, b, "person") is True


@pytest.mark.parametrize(
    "a,b,kind",
    [
        # Two distinct people sharing first name — predicate must NOT merge.
        # `Marco` (alone) vs `Marco Bianchi` is the v2 two-Marcos test:
        # bare `Marco` (p03 wreck diver) vs the surname-disambiguated
        # `Marco Bianchi` (p05 tax-loss researcher). The predicate
        # deliberately does NOT first-name-merge because it can't tell ASR
        # aliases (`Liam` ↔ `Liam Verbeek`, same person) from organic
        # same-first-name pairs. Differentiating needs external signal.
        ("Marco", "Marco Bianchi", "person"),
        ("Jacob Goldstein", "Rob Goldstein", "person"),
        # Family-only reference must NOT cross-merge people sharing a last name
        ("Mark Carney", "John Carney", "person"),
        # Org-side token-count tolerance is intentionally disabled — guards
        # against `Adobe` ↔ `Adobe Creative Cloud` (sub-product, not alias).
        ("Adobe", "Adobe Creative Cloud", "org"),
    ],
)
def test_xep_variants_904_predicate_does_not_overmerge(a, b, kind):
    assert _are_xep_variants(a, b, kind) is False


@pytest.mark.parametrize(
    "a,b",
    [
        # First-name-only alias — currently does NOT merge by design (see the
        # NOTE in `_token_count_tolerant_match`). When a future LLM-tier
        # escalation or same-show evidence-based merge ships, this test
        # should flip to expecting True.
        ("Liam Verbeek", "Liam"),
    ],
)
def test_xep_variants_904_first_name_only_alias_deferred(a, b):
    """First-name-only-alias merge deferred — same predicate shape as the
    two-Marcos distinct-people case; can't disambiguate without external
    signal. Tracked for follow-up (#906 / #921)."""
    assert _are_xep_variants(a, b, "person") is False


# --- canonical map: frequency + same-show ---------------------------------------


def test_canonical_is_highest_frequency():
    cands = {
        "org:cargill": _cand("org:cargill", "org", "Cargill", ["e1", "e2"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e3"], ["showA"]),
    }
    payload, id_map = build_entity_canonical_map(cands)
    # Lower-frequency variant maps to the higher-frequency canonical.
    assert id_map == {"org:cargil": "org:cargill"}
    assert payload["merged_variants"] == 1
    assert payload["clusters"][0]["canonical_id"] == "org:cargill"


def test_same_show_required_blocks_cross_show_merge():
    cands = {
        "org:cargill": _cand("org:cargill", "org", "Cargill", ["e1", "e2"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e3"], ["showB"]),  # other show
    }
    _, id_map = build_entity_canonical_map(cands, same_show_required=True)
    assert id_map == {}


def test_landmines_not_merged_in_map():
    cands = {
        "org:claude": _cand("org:claude", "org", "Claude", ["e1", "e2"], ["showA"]),
        "org:claude-3": _cand("org:claude-3", "org", "Claude 3", ["e1"], ["showA"]),
        "org:ups": _cand("org:ups", "org", "UPS", ["e1"], ["showA"]),
        "org:usps": _cand("org:usps", "org", "USPS", ["e1"], ["showA"]),
    }
    _, id_map = build_entity_canonical_map(cands)
    assert id_map == {}


def test_kind_aware_no_cross_kind_merge():
    cands = {
        "person:cargill": _cand("person:cargill", "person", "Cargill", ["e1"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e1"], ["showA"]),
    }
    _, id_map = build_entity_canonical_map(cands)
    assert id_map == {}


# --- collect + end-to-end on a synthetic corpus ---------------------------------


def _write_kg(path: Path, episode_id: str, show: str, entities):
    nodes = [{"id": f"episode:{episode_id}", "type": "Episode", "properties": {"podcast_id": show}}]
    for eid, name in entities:
        nodes.append({"id": eid, "type": "Entity", "properties": {"name": name}})
    path.write_text(json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": []}))


def test_collect_and_build_id_map_from_corpus(tmp_path):
    # Same show, two episodes: Cargill (ep1, ep2) + Cargil (ep3) → collapse.
    _write_kg(tmp_path / "e1.kg.json", "e1", "showA", [("org:cargill", "Cargill")])
    _write_kg(tmp_path / "e2.kg.json", "e2", "showA", [("org:cargill", "Cargill")])
    _write_kg(tmp_path / "e3.kg.json", "e3", "showA", [("org:cargil", "Cargil")])

    cands = collect_entity_candidates(tmp_path)
    assert cands["org:cargill"].freq == 2
    assert cands["org:cargil"].freq == 1
    assert cands["org:cargill"].shows == {"showA"}

    id_map = build_entity_id_map(tmp_path)
    assert id_map == {"org:cargil": "org:cargill"}


def test_id_map_from_payload_roundtrip():
    cands = {
        "org:cargill": _cand("org:cargill", "org", "Cargill", ["e1", "e2"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e3"], ["showA"]),
    }
    payload, id_map = build_entity_canonical_map(cands)
    assert id_map_from_clusters_payload(payload) == id_map
