"""Unit tests for derived relational edges (Insight→Entity, Podcast→HAS_EPISODE)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.relational_edges import (
    add_episode_show_edges,
    add_insight_entity_edges,
    apply_typed_mentions_and_rewrite_gi,
    apply_typed_mentions_to_gi_artifact,
    kg_entity_index,
    kg_entity_names,
)

pytestmark = pytest.mark.unit


def _artifact():
    return {
        "schema_version": "3.0",
        "nodes": [
            {"id": "episode:e1", "type": "Episode", "properties": {}},
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {"text": "Elon Musk plans to list SpaceX."},
            },
            {
                "id": "insight:2",
                "type": "Insight",
                "properties": {"text": "Markets recovered this quarter."},
            },
        ],
        "edges": [],
    }


def test_insight_entity_matches_whole_word_name():
    """RFC-097 v3.0: emits typed MENTIONS_PERSON / MENTIONS_ORG based on kind."""
    art = _artifact()
    added = add_insight_entity_edges(
        art,
        {
            "person:elon-musk": ("Elon Musk", "person"),
            "org:spacex": ("SpaceX", "organization"),
        },
    )
    assert added == 2  # both names appear in insight:1
    typed = {(e["from"], e["to"], e["type"]) for e in art["edges"]}
    assert ("insight:1", "person:elon-musk", "MENTIONS_PERSON") in typed
    assert ("insight:1", "org:spacex", "MENTIONS_ORG") in typed
    # insight:2 mentions neither
    assert all(t[0] == "insight:1" for t in typed)
    # Schema bumped to v3.0 on first typed-edge addition.
    assert art["schema_version"] == "3.0"


def test_insight_entity_no_substring_false_positive():
    art = _artifact()
    # "Musk" alone should not match inside another word; and an absent entity adds nothing
    added = add_insight_entity_edges(art, {"person:cathie-wood": ("Cathie Wood", "person")})
    assert added == 0
    # No typed edges added -> schema_version unchanged.
    assert art["schema_version"] == "3.0"


def test_insight_entity_idempotent():
    """Repeated calls don't double-add (dedup by (from, to, type))."""
    art = _artifact()
    index = {"person:elon-musk": ("Elon Musk", "person")}
    assert add_insight_entity_edges(art, index) == 1
    assert add_insight_entity_edges(art, index) == 0


def test_insight_entity_skips_when_legacy_mentions_already_present():
    """Permissive: legacy generic MENTIONS suppresses re-emission of typed edge."""
    art = _artifact()
    # Pre-existing legacy edge from an old artifact
    art["edges"].append({"type": "MENTIONS", "from": "insight:1", "to": "person:elon-musk"})
    added = add_insight_entity_edges(art, {"person:elon-musk": ("Elon Musk", "person")})
    assert added == 0  # legacy edge dedup-blocks the typed one
    # schema_version not bumped because no new edge landed
    assert art["schema_version"] == "3.0"


def test_episode_show_adds_podcast_node_and_edge():
    art = _artifact()
    added = add_episode_show_edges(art, "Odd Lots")
    assert added == 1
    assert {"type": "HAS_EPISODE", "from": "podcast:odd-lots", "to": "episode:e1"} in art["edges"]
    pod = next(n for n in art["nodes"] if n["id"] == "podcast:odd-lots" and n["type"] == "Podcast")
    # RFC-097 chunk-4 retroactive sweep: schema requires "title" (not "name").
    assert pod["properties"] == {"title": "Odd Lots"}
    # idempotent
    assert add_episode_show_edges(art, "Odd Lots") == 0


def test_insight_entity_edges_add_missing_target_nodes():
    """RFC-097 chunk-4 retroactive: target Person/Organization nodes are added
    when they don't already exist in the artifact (otherwise the edge would
    dangle and the viewer would have to cross-join with kg.json)."""
    art = _artifact()
    assert not any(n["type"] in ("Person", "Organization") for n in art["nodes"])
    added = add_insight_entity_edges(
        art,
        {
            "person:elon-musk": ("Elon Musk", "person"),
            "org:spacex": ("SpaceX", "organization"),
        },
    )
    assert added == 2
    persons = [n for n in art["nodes"] if n["type"] == "Person"]
    orgs = [n for n in art["nodes"] if n["type"] == "Organization"]
    assert len(persons) == 1 and persons[0]["id"] == "person:elon-musk"
    assert persons[0]["properties"] == {"name": "Elon Musk"}
    assert len(orgs) == 1 and orgs[0]["id"] == "org:spacex"
    assert orgs[0]["properties"] == {"name": "SpaceX"}


def test_insight_entity_edges_do_not_duplicate_existing_target_nodes():
    """If the target Person already exists in the artifact (e.g. from
    SPOKEN_BY emission), the helper does NOT add a duplicate."""
    art = _artifact()
    art["nodes"].append(
        {"id": "person:elon-musk", "type": "Person", "properties": {"name": "Elon Musk"}}
    )
    pre_count = sum(1 for n in art["nodes"] if n["id"] == "person:elon-musk")
    added = add_insight_entity_edges(art, {"person:elon-musk": ("Elon Musk", "person")})
    assert added == 1  # edge added
    post_count = sum(1 for n in art["nodes"] if n["id"] == "person:elon-musk")
    assert post_count == pre_count  # node count unchanged


def test_episode_show_empty_title_noop():
    art = _artifact()
    assert add_episode_show_edges(art, "") == 0


def test_kg_entity_names_extracts_id_to_name():
    """Backward-compat: kg_entity_names returns plain {id: name} (no kind)."""
    kg = {
        "nodes": [
            {
                "id": "person:gillian-tett",
                "type": "Entity",
                "properties": {"name": "Gillian Tett", "kind": "person"},
            },
            {
                "id": "org:financial-times",
                "type": "Entity",
                "properties": {"name": "Financial Times"},
            },
            {"id": "topic:debt", "type": "Topic", "properties": {}},
        ]
    }
    assert kg_entity_names(kg) == {
        "person:gillian-tett": "Gillian Tett",
        "org:financial-times": "Financial Times",
    }


def test_kg_entity_index_carries_kind_for_typed_mentions():
    """RFC-097: kg_entity_index returns (name, kind) so typed MENTIONS_* edges can be emitted."""
    kg = {
        "nodes": [
            # v2.0 typed Person node (RFC-097)
            {
                "id": "person:gillian-tett",
                "type": "Person",
                "properties": {"name": "Gillian Tett"},
            },
            # v2.0 typed Organization node
            {
                "id": "org:financial-times",
                "type": "Organization",
                "properties": {"name": "Financial Times"},
            },
            # Legacy v1.2 Entity with `kind`
            {
                "id": "person:martin-wolf",
                "type": "Entity",
                "properties": {"name": "Martin Wolf", "kind": "person"},
            },
            # Legacy v1.0/1.1 Entity with `entity_kind`
            {
                "id": "org:lse",
                "type": "Entity",
                "properties": {"name": "LSE", "entity_kind": "organization"},
            },
            {"id": "topic:debt", "type": "Topic", "properties": {}},
        ]
    }
    assert kg_entity_index(kg) == {
        "person:gillian-tett": ("Gillian Tett", "person"),
        "org:financial-times": ("Financial Times", "organization"),
        "person:martin-wolf": ("Martin Wolf", "person"),
        "org:lse": ("LSE", "organization"),
    }


class TestApplyTypedMentionsToGiArtifact:
    """RFC-097 v3.0 chunk-4 post-pass — typed MENTIONS edges via KG entity index.

    These tests cover the cross-layer helper used by the production
    orchestrator (``workflow/metadata_generation.py``) and any future eval
    surface that builds both GI + KG for an episode.
    """

    def _gi_artifact(self) -> dict:
        return {
            "schema_version": "3.0",
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {}},
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "text": "Elon Musk announced new SpaceX contracts.",
                        "grounded": True,
                    },
                },
                {
                    "id": "insight:2",
                    "type": "Insight",
                    "properties": {
                        "text": "The quarter closed with mild gains.",
                        "grounded": True,
                    },
                },
            ],
            "edges": [],
        }

    def _kg_artifact(self, *, with_entities: bool = True) -> dict:
        nodes = [
            {"id": "episode:e1", "type": "Episode", "properties": {}},
            {"id": "topic:markets", "type": "Topic", "properties": {"label": "markets"}},
        ]
        if with_entities:
            nodes.extend(
                [
                    {
                        "id": "person:elon-musk",
                        "type": "Person",
                        "properties": {"name": "Elon Musk"},
                    },
                    {
                        "id": "org:spacex",
                        "type": "Organization",
                        "properties": {"name": "SpaceX"},
                    },
                ]
            )
        return {"schema_version": "2.0", "nodes": nodes, "edges": []}

    def test_emits_typed_edges_from_kg_index(self):
        """Helper end-to-end: KG → entity_index → typed MENTIONS edges."""
        gi = self._gi_artifact()
        kg = self._kg_artifact()
        added = apply_typed_mentions_to_gi_artifact(gi, kg)
        assert added == 2  # one PERSON + one ORG matched in insight:1
        edge_types = sorted(e["type"] for e in gi["edges"])
        assert edge_types == ["MENTIONS_ORG", "MENTIONS_PERSON"]
        # schema_version was already "3.0", so no further bump
        assert gi["schema_version"] == "3.0"

    def test_no_kg_entities_is_zero_op(self):
        """KG without Person/Org nodes leaves GI untouched."""
        gi = self._gi_artifact()
        kg = self._kg_artifact(with_entities=False)
        added = apply_typed_mentions_to_gi_artifact(gi, kg)
        assert added == 0
        assert gi["edges"] == []

    def test_idempotent_repeat_invocations_no_double_add(self):
        """Running the post-pass twice does not duplicate edges."""
        gi = self._gi_artifact()
        kg = self._kg_artifact()
        first = apply_typed_mentions_to_gi_artifact(gi, kg)
        second = apply_typed_mentions_to_gi_artifact(gi, kg)
        assert first == 2
        assert second == 0
        edge_keys = {(e["from"], e["to"], e["type"]) for e in gi["edges"]}
        assert len(edge_keys) == 2  # no duplicates

    def test_bumps_v2_schema_to_v3_when_edges_added(self):
        """Helper bumps GI schema_version 2.0 → 3.0 on first typed-edge add."""
        gi = self._gi_artifact()
        gi["schema_version"] = "2.0"
        kg = self._kg_artifact()
        added = apply_typed_mentions_to_gi_artifact(gi, kg)
        assert added == 2
        assert gi["schema_version"] == "3.0"

    def test_legacy_entity_kind_node_in_kg_still_resolved(self):
        """KG v1.x ``Entity`` with legacy ``entity_kind`` resolves to MENTIONS_ORG.

        v1.2 used ``properties.kind`` = ``"person"`` / ``"org"`` (note the abbreviation).
        Earlier shape used ``properties.entity_kind`` = ``"person"`` / ``"organization"``.
        Both should map correctly via :func:`normalized_entity_kind_from_node`.
        """
        gi = self._gi_artifact()
        kg = self._kg_artifact(with_entities=False)
        kg["nodes"].append(
            {
                "id": "org:spacex-legacy",
                "type": "Entity",
                "properties": {"name": "SpaceX", "entity_kind": "organization"},
            }
        )
        added = apply_typed_mentions_to_gi_artifact(gi, kg)
        assert added == 1
        assert gi["edges"][0]["type"] == "MENTIONS_ORG"
        assert gi["edges"][0]["to"] == "org:spacex-legacy"

    def test_empty_gi_artifact_safe(self):
        """An empty GI (no Insight nodes) is a clean no-op, not an error."""
        gi = {"schema_version": "3.0", "nodes": [], "edges": []}
        kg = self._kg_artifact()
        added = apply_typed_mentions_to_gi_artifact(gi, kg)
        assert added == 0
        assert gi["edges"] == []


class TestApplyTypedMentionsAndRewriteGi:
    """Tests for the orchestrator-facing combined helper:
    mutate-in-memory + conditional disk re-write.
    """

    def _gi_artifact(self) -> dict:
        return {
            "schema_version": "3.0",
            "model_version": "test",
            "prompt_version": "v1",
            "episode_id": "e1",
            "nodes": [
                {
                    "id": "episode:e1",
                    "type": "Episode",
                    "properties": {
                        "episode_id": "e1",
                        "podcast_id": "podcast:test",
                        "title": "Test Episode",
                    },
                },
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "text": "Elon Musk announced new SpaceX contracts.",
                        "grounded": True,
                        "episode_id": "e1",
                        "insight_type": "claim",
                    },
                },
            ],
            "edges": [
                {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:1"},
            ],
        }

    def _kg_with_entities(self) -> dict:
        return {
            "schema_version": "2.0",
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {}},
                {
                    "id": "person:elon-musk",
                    "type": "Person",
                    "properties": {"name": "Elon Musk"},
                },
                {
                    "id": "org:spacex",
                    "type": "Organization",
                    "properties": {"name": "SpaceX"},
                },
            ],
            "edges": [],
        }

    def test_rewrites_disk_when_edges_added(self, tmp_path):
        """Helper writes the mutated GI artifact to disk when edges are added."""
        import json

        gi = self._gi_artifact()
        kg = self._kg_with_entities()
        gi_path = tmp_path / "gi.json"
        gi_path.write_text(json.dumps(gi))  # initial on-disk state

        added = apply_typed_mentions_and_rewrite_gi(gi, kg, str(gi_path))

        assert added == 2
        on_disk = json.loads(gi_path.read_text())
        edge_types = sorted(e["type"] for e in on_disk["edges"])
        assert "MENTIONS_PERSON" in edge_types
        assert "MENTIONS_ORG" in edge_types

    def test_no_rewrite_when_zero_op(self, tmp_path):
        """Helper does NOT touch disk when no edges are added (mtime stable).

        Critical for idempotency — re-running the orchestrator on a corpus
        already at v3.0 mustn't churn artifact mtimes (which would invalidate
        downstream caches like FAISS index).
        """
        import json
        import time

        gi = self._gi_artifact()
        # KG without Person/Org entities → no typed edges possible
        kg = {"schema_version": "2.0", "nodes": [], "edges": []}
        gi_path = tmp_path / "gi.json"
        gi_path.write_text(json.dumps(gi))
        original_mtime = gi_path.stat().st_mtime_ns

        # Sleep a tiny bit to make mtime change detectable
        time.sleep(0.01)

        added = apply_typed_mentions_and_rewrite_gi(gi, kg, str(gi_path))

        assert added == 0
        assert gi_path.stat().st_mtime_ns == original_mtime  # disk untouched

    def test_idempotent_second_call_does_not_rewrite(self, tmp_path):
        """Calling the helper twice with the same payloads doesn't churn disk."""
        import json
        import time

        gi = self._gi_artifact()
        kg = self._kg_with_entities()
        gi_path = tmp_path / "gi.json"
        gi_path.write_text(json.dumps(gi))

        first = apply_typed_mentions_and_rewrite_gi(gi, kg, str(gi_path))
        assert first == 2
        mtime_after_first = gi_path.stat().st_mtime_ns

        time.sleep(0.01)
        second = apply_typed_mentions_and_rewrite_gi(gi, kg, str(gi_path))
        assert second == 0  # nothing to add on second run
        assert gi_path.stat().st_mtime_ns == mtime_after_first  # disk untouched


# ─────────────────────────────────────────────────────────────────────
# #1076 chunk 4-A — spaCy NER pass coverage. The leaf function accepts
# an optional ``nlp`` kwarg that catches name fragments the literal
# regex misses (e.g. KG entry "Maya Hutchinson" matched by spaCy span
# "Maya"). Stub nlp implementations keep the test hermetic — no real
# spaCy model load.
# ─────────────────────────────────────────────────────────────────────


class _StubEnt:
    """Mimics a single spaCy Span — label_ + text are all the production
    code reads."""

    def __init__(self, text: str, label: str = "PERSON") -> None:
        self.text = text
        self.label_ = label


class _StubDoc:
    def __init__(self, ents):
        self.ents = ents


class _StubNlp:
    """Callable that returns a deterministic ``_StubDoc`` per input text.

    Configured at construction with a mapping ``text -> list of PERSON
    spans``; unknown texts return an empty doc. Production spaCy is
    ``nlp(text) -> Doc``; this matches that contract.
    """

    def __init__(self, ent_map):
        self._ent_map = ent_map

    def __call__(self, text):
        spans = self._ent_map.get(text, [])
        return _StubDoc([_StubEnt(s) for s in spans])


class TestAddInsightEntityEdgesWithNer:
    """#1076 chunk 4-A — NER pass behavior on the leaf function."""

    def _gi_with_paraphrased_insight(self):
        """GI artifact whose Insight text mentions 'Maya' (a first-name
        fragment) but NOT the full 'Maya Hutchinson' that the KG carries."""
        return {
            "schema_version": "3.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "ep:1",
            "nodes": [
                {
                    "id": "ep:1",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "podcast:p",
                        "title": "T",
                        "publish_date": "2024-01-01T00:00:00Z",
                    },
                },
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "text": "Maya argued that trail building needs more federal support.",
                        "episode_id": "ep:1",
                        "grounded": True,
                        "insight_type": "claim",
                        "position_hint": 0.4,
                    },
                },
            ],
            "edges": [],
        }

    def _kg_index_with_maya(self):
        """entity_index matching the leaf fn's input shape: id -> (name, kind).

        KG carries the FULL name 'Maya Hutchinson'; the Insight text says
        only 'Maya' so the regex pass misses but the NER pass should
        catch via token-subset match.
        """
        return {"person:maya-hutchinson": ("Maya Hutchinson", "person")}

    def test_regex_alone_misses_first_name_fragment(self):
        """Without nlp, the substring regex doesn't match 'Maya' against
        'Maya Hutchinson' — the documented limitation that motivates Path A."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        gi = self._gi_with_paraphrased_insight()
        added = add_insight_entity_edges(gi, self._kg_index_with_maya())
        assert added == 0
        assert all(e.get("type") != "MENTIONS_PERSON" for e in gi.get("edges", []))

    def test_ner_catches_first_name_fragment(self):
        """With an nlp that detects 'Maya' as PERSON, the NER pass emits
        the MENTIONS_PERSON edge the regex missed."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        nlp = _StubNlp({"Maya argued that trail building needs more federal support.": ["Maya"]})
        gi = self._gi_with_paraphrased_insight()
        added = add_insight_entity_edges(gi, self._kg_index_with_maya(), nlp=nlp)
        assert added == 1
        edge = next(e for e in gi["edges"] if e.get("type") == "MENTIONS_PERSON")
        assert edge["from"] == "insight:1"
        assert edge["to"] == "person:maya-hutchinson"

    def test_ner_does_not_double_emit_when_regex_also_matched(self):
        """If both passes would emit the same edge, dedup keeps it to one."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        gi = self._gi_with_paraphrased_insight()
        gi["nodes"][1]["properties"]["text"] = "Maya Hutchinson argued strongly."
        nlp = _StubNlp({"Maya Hutchinson argued strongly.": ["Maya", "Maya Hutchinson"]})
        added = add_insight_entity_edges(gi, self._kg_index_with_maya(), nlp=nlp)
        assert added == 1
        mp_edges = [e for e in gi["edges"] if e.get("type") == "MENTIONS_PERSON"]
        assert len(mp_edges) == 1

    def test_ner_rejects_span_that_is_not_a_subset_of_an_indexed_name(self):
        """Span 'Maya Smith' must NOT match KG 'Maya Hutchinson' — the
        subset constraint catches the 'shared first name, different
        person' false positive."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        gi = self._gi_with_paraphrased_insight()
        gi["nodes"][1]["properties"]["text"] = "Maya Smith argued strongly."
        nlp = _StubNlp({"Maya Smith argued strongly.": ["Maya Smith"]})
        added = add_insight_entity_edges(gi, self._kg_index_with_maya(), nlp=nlp)
        assert added == 0

    def test_ner_skips_short_spans(self):
        """Spans <3 chars (rare wrong PERSON detections like 'AI') get
        filtered before the subset check fires."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        gi = self._gi_with_paraphrased_insight()
        gi["nodes"][1]["properties"]["text"] = "AI argued strongly."
        nlp = _StubNlp({"AI argued strongly.": ["AI"]})
        added = add_insight_entity_edges(
            gi, {"person:ai-research": ("AI Research", "person")}, nlp=nlp
        )
        assert added == 0

    def test_ner_handles_nlp_throwing_gracefully(self):
        """A malformed spaCy doc on one Insight doesn't kill the pass."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        class _CrashyNlp:
            def __call__(self, text):
                raise RuntimeError("simulated spaCy crash")

        gi = self._gi_with_paraphrased_insight()
        added = add_insight_entity_edges(gi, self._kg_index_with_maya(), nlp=_CrashyNlp())
        assert added == 0

    def test_ner_rejects_ambiguous_shared_surname_match(self):
        """When the KG has multiple people sharing a surname (e.g. 'Donald
        Trump' AND 'Eric Trump'), a bare span 'Trump' must NOT fire any
        edge — we can't tell which one the text refers to, so emitting
        to all of them would scatter wrong attributions across the corpus
        and emitting to one would be arbitrary. The disambiguation rule
        added in #1076 chunk 4-A's prod-v2 measurement run rejects the
        match entirely.
        """
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        gi = self._gi_with_paraphrased_insight()
        gi["nodes"][1]["properties"]["text"] = "Trump argued that policy must shift."
        entity_index = {
            "person:donald-trump": ("Donald Trump", "person"),
            "person:eric-trump": ("Eric Trump", "person"),
        }
        nlp = _StubNlp({"Trump argued that policy must shift.": ["Trump"]})
        added = add_insight_entity_edges(gi, entity_index, nlp=nlp)
        assert added == 0

    def test_ner_keeps_multi_token_match_when_no_ambiguity(self):
        """A two-token span 'Donald Trump' is unambiguous even when KG
        also has 'Eric Trump' — only one entry matches the full span."""
        from podcast_scraper.gi.relational_edges import add_insight_entity_edges

        gi = self._gi_with_paraphrased_insight()
        gi["nodes"][1]["properties"]["text"] = "Donald Trump argued for the policy."
        entity_index = {
            "person:donald-trump": ("Donald Trump", "person"),
            "person:eric-trump": ("Eric Trump", "person"),
        }
        nlp = _StubNlp({"Donald Trump argued for the policy.": ["Donald Trump"]})
        added = add_insight_entity_edges(gi, entity_index, nlp=nlp)
        assert added == 1
        edge = next(e for e in gi["edges"] if e.get("type") == "MENTIONS_PERSON")
        assert edge["to"] == "person:donald-trump"
