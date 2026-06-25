"""#1058 chunk 1 — unit tests for ``kg.ner_postpass``.

Covers the deterministic spaCy NER post-pass that adds KG
``Organization`` nodes under LLM-free profiles (airgapped /
airgapped_thin). Uses a fake spaCy ``Doc`` / ``Span`` shape so the
test runs without loading a real model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.kg.ner_postpass import (
    _existing_org_slugs,
    _looks_like_org,
    add_org_entities_from_ner,
    apply_org_postpass_to_kg_artifact,
    extract_org_entities,
    slug_for_org,
)

pytestmark = pytest.mark.unit


def _fake_nlp(by_text: Dict[str, List[Tuple[str, str]]]):
    """Build a callable that mimics spaCy: input text → Doc with
    ``ents`` of the supplied ``(text, label)`` tuples for that input."""

    def _call(text: str) -> Any:
        ents = [SimpleNamespace(text=t, label_=lbl) for t, lbl in by_text.get(text, [])]
        return SimpleNamespace(ents=ents)

    return _call


def _gi(insights: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Minimal GI artifact with the given ``(insight_id, text)`` pairs."""
    return {
        "schema_version": "3.0",
        "nodes": [
            {
                "id": iid,
                "type": "Insight",
                "properties": {"text": text},
            }
            for iid, text in insights
        ],
        "edges": [],
    }


def _kg(nodes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {"schema_version": "2.0", "nodes": list(nodes or []), "edges": []}


class TestSlugForOrg:
    def test_simple_lowercase(self) -> None:
        assert slug_for_org("Acme") == "acme"

    def test_multi_word_kebab(self) -> None:
        assert slug_for_org("Open AI") == "open-ai"

    def test_strips_punctuation(self) -> None:
        assert slug_for_org("AT&T, Inc.") == "at-t-inc"

    def test_collapses_runs_of_separators(self) -> None:
        assert slug_for_org("  Foo // Bar  ") == "foo-bar"

    def test_empty_input(self) -> None:
        assert slug_for_org("") == ""
        assert slug_for_org("   ") == ""


class TestLooksLikeOrg:
    def test_accepts_normal_org(self) -> None:
        assert _looks_like_org("Acme Corporation") is True

    def test_rejects_short_single_token(self) -> None:
        # 2-char single token (e.g. spaCy mistagging "AI") — drop.
        assert _looks_like_org("AI") is False

    def test_accepts_short_multi_token(self) -> None:
        # Multi-token is acceptable even when individual tokens short.
        assert _looks_like_org("AI Lab") is True

    def test_rejects_numeric_only(self) -> None:
        # spaCy occasionally tags years/numbers as ORG.
        assert _looks_like_org("2024") is False

    def test_rejects_empty(self) -> None:
        assert _looks_like_org("") is False
        assert _looks_like_org("   ") is False


class TestExtractOrgEntities:
    def test_returns_empty_when_nlp_is_none(self) -> None:
        assert extract_org_entities("Some text mentioning Acme", None) == []

    def test_returns_empty_for_empty_text(self) -> None:
        assert extract_org_entities("", _fake_nlp({})) == []

    def test_extracts_single_org(self) -> None:
        nlp = _fake_nlp({"Acme launched a product.": [("Acme", "ORG")]})
        result = extract_org_entities("Acme launched a product.", nlp)
        assert len(result) == 1 and result[0][0] == "Acme"

    def test_skips_non_org_labels(self) -> None:
        nlp = _fake_nlp(
            {
                "Alice met Bob at Acme.": [
                    ("Alice", "PERSON"),
                    ("Bob", "PERSON"),
                    ("Acme", "ORG"),
                ]
            }
        )
        result = extract_org_entities("Alice met Bob at Acme.", nlp)
        assert [n for n, _ in result] == ["Acme"]

    def test_dedupes_by_slug(self) -> None:
        nlp = _fake_nlp({"Acme Inc and acme inc": [("Acme Inc", "ORG"), ("acme inc", "ORG")]})
        result = extract_org_entities("Acme Inc and acme inc", nlp)
        assert len(result) == 1

    def test_handles_nlp_throwing_gracefully(self) -> None:
        def _crashy(_text: str) -> Any:
            raise RuntimeError("spaCy crashed")

        assert extract_org_entities("text", _crashy) == []

    def test_drops_short_single_token_spans(self) -> None:
        nlp = _fake_nlp({"We use AI for everything.": [("AI", "ORG")]})
        assert extract_org_entities("We use AI for everything.", nlp) == []


class TestExistingOrgSlugs:
    def test_collects_present_slugs(self) -> None:
        kg = _kg(
            [
                {
                    "id": "org:acme",
                    "type": "Organization",
                    "properties": {"name": "Acme"},
                },
                {
                    "id": "org:open-ai",
                    "type": "Organization",
                    "properties": {"name": "Open AI"},
                },
                {
                    "id": "person:alice",
                    "type": "Person",
                    "properties": {"name": "Alice"},
                },
            ]
        )
        assert _existing_org_slugs(kg) == {"acme", "open-ai"}

    def test_handles_empty_kg(self) -> None:
        assert _existing_org_slugs(_kg([])) == set()


class TestAddOrgEntitiesFromNer:
    def test_adds_organization_node_from_insight_text(self) -> None:
        kg = _kg([])
        gi = _gi([("insight:1", "Acme launched a product.")])
        nlp = _fake_nlp({"Acme launched a product.": [("Acme", "ORG")]})

        added = add_org_entities_from_ner(kg, gi, nlp)

        assert added == 1
        org_nodes = [n for n in kg["nodes"] if n["type"] == "Organization"]
        assert len(org_nodes) == 1
        n = org_nodes[0]
        assert n["id"] == "org:acme"
        assert n["properties"]["name"] == "Acme"
        assert n["properties"]["role"] == "mentioned"

    def test_idempotent_when_org_already_present(self) -> None:
        kg = _kg(
            [
                {
                    "id": "org:acme",
                    "type": "Organization",
                    "properties": {"name": "Acme"},
                }
            ]
        )
        gi = _gi([("insight:1", "Acme launched a product.")])
        nlp = _fake_nlp({"Acme launched a product.": [("Acme", "ORG")]})

        added = add_org_entities_from_ner(kg, gi, nlp)

        assert added == 0
        assert sum(1 for n in kg["nodes"] if n["type"] == "Organization") == 1

    def test_dedupes_across_insights(self) -> None:
        kg = _kg([])
        gi = _gi(
            [
                ("insight:1", "Acme launched a product."),
                ("insight:2", "Acme posted strong earnings."),
            ]
        )
        nlp = _fake_nlp(
            {
                "Acme launched a product.": [("Acme", "ORG")],
                "Acme posted strong earnings.": [("Acme", "ORG")],
            }
        )
        added = add_org_entities_from_ner(kg, gi, nlp)
        assert added == 1

    def test_returns_zero_when_nlp_is_none(self) -> None:
        kg = _kg([])
        gi = _gi([("insight:1", "Acme.")])
        assert add_org_entities_from_ner(kg, gi, None) == 0
        assert not kg["nodes"]

    def test_returns_zero_when_no_insights(self) -> None:
        kg = _kg([])
        gi = _gi([])
        nlp = _fake_nlp({"anything": [("Acme", "ORG")]})
        assert add_org_entities_from_ner(kg, gi, nlp) == 0

    def test_node_satisfies_schema_required_fields(self) -> None:
        """Produced Organization node carries the strict v2.0 required
        keys (id / type / properties.name) per kg.schema.json."""
        kg = _kg([])
        gi = _gi([("insight:1", "Acme launched.")])
        nlp = _fake_nlp({"Acme launched.": [("Acme", "ORG")]})
        add_org_entities_from_ner(kg, gi, nlp)
        node = next(n for n in kg["nodes"] if n["type"] == "Organization")
        assert set(node) == {"id", "type", "properties"}
        assert "name" in node["properties"]
        assert node["properties"]["role"] == "mentioned"


class TestApplyOrgPostpassToKgArtifact:
    def test_public_wrapper_returns_added_count(self) -> None:
        kg = _kg([])
        gi = _gi([("insight:1", "Acme launched.")])
        nlp = _fake_nlp({"Acme launched.": [("Acme", "ORG")]})
        assert apply_org_postpass_to_kg_artifact(kg, gi, nlp) == 1
