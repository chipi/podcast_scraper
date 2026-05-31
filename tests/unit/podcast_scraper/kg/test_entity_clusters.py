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
