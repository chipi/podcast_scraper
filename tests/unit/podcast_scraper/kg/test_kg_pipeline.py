"""Unit tests for KG pipeline (stub artifact builder)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from podcast_scraper.kg.pipeline import build_artifact
from podcast_scraper.kg.schema import validate_artifact


class TestKgPipeline(unittest.TestCase):
    """Tests for build_artifact."""

    def test_build_artifact_minimal_passes_strict_schema(self) -> None:
        """v2.0 (RFC-097): minimal graph emits Episode + Podcast + HAS_EPISODE."""
        art = build_artifact(
            "episode:test-1",
            "Hello transcript.",
            podcast_id="podcast:abc",
            episode_title="My Episode",
            publish_date="2024-01-15T12:00:00Z",
            transcript_ref="transcripts/ep.txt",
        )
        self.assertEqual(art["schema_version"], "2.0")
        self.assertEqual(art["episode_id"], "episode:test-1")
        types = {n["type"] for n in art["nodes"]}
        self.assertEqual(types, {"Episode", "Podcast"})
        ep_node = next(n for n in art["nodes"] if n["type"] == "Episode")
        self.assertEqual(ep_node["id"], "episode:episode:test-1")
        # HAS_EPISODE edge from Podcast → Episode (RFC-097 chunk 3)
        has_episode_edges = [e for e in art["edges"] if e["type"] == "HAS_EPISODE"]
        self.assertEqual(len(has_episode_edges), 1)
        self.assertEqual(has_episode_edges[0]["from"], "podcast:abc")
        self.assertEqual(has_episode_edges[0]["to"], "episode:episode:test-1")
        validate_artifact(art, strict=True)

    def test_build_artifact_feed_id_lands_on_episode_properties(self) -> None:
        """#658: feed_id parameter writes to Episode properties + still validates."""
        art = build_artifact(
            "ep:k",
            "x",
            podcast_id="podcast:abc",
            episode_title="My Episode",
            feed_id="rss_example_com_abc123",
        )
        ep_nodes = [n for n in art["nodes"] if n["type"] == "Episode"]
        self.assertEqual(len(ep_nodes), 1)
        self.assertEqual(ep_nodes[0]["properties"]["feed_id"], "rss_example_com_abc123")
        validate_artifact(art, strict=True)

    def test_build_artifact_omits_feed_id_when_not_supplied(self) -> None:
        """#658: feed_id is optional; omitting it leaves the property absent."""
        art = build_artifact(
            "ep:k",
            "x",
            podcast_id="podcast:abc",
            episode_title="My Episode",
        )
        ep_nodes = [n for n in art["nodes"] if n["type"] == "Episode"]
        self.assertNotIn("feed_id", ep_nodes[0]["properties"])

    def test_build_artifact_topic_and_hosts(self) -> None:
        """v2.0: topic + host Person nodes produce MENTIONS edges to Episode."""
        art = build_artifact(
            "ep:x",
            "x",
            podcast_id="podcast:p1",
            episode_title="T",
            topic_label="Inflation outlook",
            detected_hosts=["Alice"],
            detected_guests=["Bob"],
        )
        validate_artifact(art, strict=True)
        types = {n["type"] for n in art["nodes"]}
        self.assertIn("Episode", types)
        self.assertIn("Topic", types)
        self.assertIn("Person", types)  # RFC-097: hosts/guests emit as Person
        self.assertTrue(any(e["type"] == "MENTIONS" for e in art["edges"]))

    def test_stub_source_skips_summary_topics(self) -> None:
        """stub + cfg: topic hints do not create Topic nodes; v2.0 Person for host."""
        cfg = SimpleNamespace(kg_extraction_source="stub")
        art = build_artifact(
            "ep:stub",
            "transcript",
            podcast_id="podcast:p1",
            episode_title="T",
            cfg=cfg,
            topic_label="Should not appear",
            detected_hosts=["Pat"],
        )
        validate_artifact(art, strict=True)
        types = {n["type"] for n in art["nodes"]}
        self.assertIn("Episode", types)
        self.assertIn("Person", types)  # RFC-097: host emits as Person
        self.assertNotIn("Topic", types)
        self.assertEqual(art["extraction"]["model_version"], "stub")

    def test_provider_path_without_ner_prepass_passes_no_hints(self) -> None:
        """#1035 — when flag is off, no ner_entity_hints / kg_prompt_version in params."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "AI"}],
            "entities": [],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
            kg_extraction_use_ner_prepass=False,
        )
        build_artifact(
            "ep:no-ner",
            "Maya from Singletrack Sessions discusses braking.",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        prov.extract_kg_graph.assert_called_once()
        call_params = prov.extract_kg_graph.call_args.kwargs.get("params") or {}
        self.assertNotIn("ner_entity_hints", call_params)
        self.assertNotIn("kg_prompt_version", call_params)

    def test_provider_path_with_ner_prepass_passes_hints(self) -> None:
        """#1035 — when flag on + spaCy cached on provider, hints flow via params."""
        from types import SimpleNamespace as _SN

        # Fake spaCy: returns a Doc with two PERSON entities
        fake_ents = [
            _SN(text="Maya", label_="PERSON"),
            _SN(text="Singletrack Sessions", label_="ORG"),
            _SN(text="Tuesday", label_="DATE"),  # dropped — not PERSON/ORG
        ]
        fake_doc = _SN(ents=fake_ents)

        def fake_nlp(text: str):
            return fake_doc

        prov = MagicMock()
        prov.summary_model = "test-model"
        prov._spacy_nlp = fake_nlp  # cached per Issue #387
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "AI"}],
            "entities": [
                {"name": "Maya", "entity_kind": "person"},
                {"name": "Singletrack Sessions", "entity_kind": "organization"},
            ],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
            kg_extraction_use_ner_prepass=True,
        )
        build_artifact(
            "ep:with-ner",
            "Maya from Singletrack Sessions discusses braking on Tuesday.",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        prov.extract_kg_graph.assert_called_once()
        call_params = prov.extract_kg_graph.call_args.kwargs.get("params") or {}
        self.assertEqual(call_params.get("kg_prompt_version"), "v5")
        hints = call_params.get("ner_entity_hints") or []
        assert hints, "expected NER hints to flow through params"
        texts = sorted(h["text"] for h in hints)
        self.assertEqual(texts, ["Maya", "Singletrack Sessions"])
        labels = sorted(h["label"] for h in hints)
        self.assertEqual(labels, ["ORG", "PERSON"])

    def test_provider_path_with_ner_prepass_no_nlp_falls_back_to_v4(self) -> None:
        """#1035 — flag on but no spaCy resolvable → no hints, no v5 (silent v4)."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov._spacy_nlp = None  # not cached
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "AI"}],
            "entities": [],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
            kg_extraction_use_ner_prepass=True,
            ner_model=None,  # speaker_detection.get_ner_model returns None
            speaker_detector_provider="spacy",
        )
        build_artifact(
            "ep:no-nlp",
            "transcript",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        prov.extract_kg_graph.assert_called_once()
        call_params = prov.extract_kg_graph.call_args.kwargs.get("params") or {}
        # No v5 promotion, no hints
        self.assertNotIn("ner_entity_hints", call_params)
        self.assertNotIn("kg_prompt_version", call_params)

    def test_provider_path_uses_llm_partial(self) -> None:
        """provider source calls extract_kg_graph and merges pipeline entities."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "AI policy"}],
            "entities": [{"name": "ACME Corp", "entity_kind": "organization"}],
        }
        metrics = SimpleNamespace(kg_provider_extractions=0)
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=True,
        )
        art = build_artifact(
            "ep:llm",
            "We discuss AI policy at ACME Corp.",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
            pipeline_metrics=metrics,
            detected_hosts=["Pat Host"],
        )
        validate_artifact(art, strict=True)
        prov.extract_kg_graph.assert_called_once()
        self.assertEqual(metrics.kg_provider_extractions, 1)
        types = {n["type"] for n in art["nodes"]}
        self.assertIn("Topic", types)
        # RFC-097 v2.0: typed Person / Organization replace Entity discriminator.
        self.assertTrue(types & {"Person", "Organization"})
        self.assertTrue(art["extraction"]["model_version"].startswith("provider:"))

    def test_provider_entities_dedup_by_kind_and_name(self) -> None:
        """Same display name may appear as person and organization; both are kept."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [],
            "entities": [
                {"name": "Mercury", "entity_kind": "person"},
                {"name": "Mercury", "entity_kind": "organization"},
                {"name": "Mercury", "entity_kind": "person"},
            ],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
        )
        art = build_artifact(
            "ep:dedup-kind",
            "transcript",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        validate_artifact(art, strict=True)
        # RFC-097 v2.0: typed Person + Organization nodes (was Entity(kind=...)).
        entities = [n for n in art["nodes"] if n["type"] in ("Person", "Organization")]
        kinds = sorted((n["type"], n["properties"]["name"]) for n in entities)
        self.assertEqual(
            kinds,
            [("Organization", "Mercury"), ("Person", "Mercury")],
        )
        self.assertEqual(len(entities), 2)

    def test_pipeline_host_skipped_only_if_same_kind_and_name_as_llm(self) -> None:
        """Host merged only when person+name matches; org with same name does not block."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [],
            "entities": [{"name": "ACME", "entity_kind": "organization"}],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=True,
        )
        art = build_artifact(
            "ep:host-org",
            "x",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
            detected_hosts=["ACME"],
        )
        validate_artifact(art, strict=True)
        # RFC-097 v2.0: typed Person + Organization (host emits Person; LLM emits Organization).
        entities = [n for n in art["nodes"] if n["type"] in ("Person", "Organization")]
        kinds = {(n["type"], n["properties"]["name"]) for n in entities}
        self.assertIn(("Organization", "ACME"), kinds)
        self.assertIn(("Person", "ACME"), kinds)
        self.assertEqual(len(entities), 2)

    def test_provider_path_propagates_llm_descriptions(self) -> None:
        """LLM partial topic/entity descriptions land on nodes (#487)."""
        prov = MagicMock()
        prov.summary_model = "m"
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "X", "description": "Why X matters."}],
            "entities": [
                {"name": "Pat", "entity_kind": "person", "description": "Guest expert."},
            ],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
        )
        art = build_artifact(
            "ep:desc",
            "t",
            podcast_id="p",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        validate_artifact(art, strict=True)
        topics = [n for n in art["nodes"] if n["type"] == "Topic"]
        self.assertEqual(topics[0]["properties"].get("description"), "Why X matters.")
        # RFC-097 v2.0: description landed on the typed Person node.
        ents = [n for n in art["nodes"] if n["type"] in ("Person", "Organization")]
        self.assertEqual(ents[0]["properties"].get("description"), "Guest expert.")

    # ─── RFC-097 chunk 3 — explicit v2.0 emission contracts ───

    def test_v2_emission_drops_legacy_kind_property(self) -> None:
        """v2.0: Person / Organization nodes drop the legacy `kind` discriminator
        (node type is the discriminator now)."""
        prov = MagicMock()
        prov.summary_model = "m"
        prov.extract_kg_graph.return_value = {
            "topics": [],
            "entities": [
                {"name": "Pat", "entity_kind": "person"},
                {"name": "ACME", "entity_kind": "organization"},
            ],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
        )
        art = build_artifact(
            "ep:v2-shape",
            "t",
            podcast_id="podcast:p",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        validate_artifact(art, strict=True)
        ents = [n for n in art["nodes"] if n["type"] in ("Person", "Organization")]
        for n in ents:
            self.assertNotIn(
                "kind",
                n["properties"],
                f"Legacy `kind` discriminator must not be emitted on v2 {n['type']} node",
            )

    def test_v2_emission_podcast_node_and_has_episode_edge(self) -> None:
        """v2.0: Podcast node + HAS_EPISODE edge land alongside Episode."""
        art = build_artifact(
            "ep:show",
            "t",
            podcast_id="podcast:the-journal",
            episode_title="Ep",
        )
        validate_artifact(art, strict=True)
        podcasts = [n for n in art["nodes"] if n["type"] == "Podcast"]
        self.assertEqual(len(podcasts), 1)
        self.assertEqual(podcasts[0]["id"], "podcast:the-journal")
        self.assertEqual(podcasts[0]["properties"]["title"], "The Journal")
        has_ep = [e for e in art["edges"] if e["type"] == "HAS_EPISODE"]
        self.assertEqual(len(has_ep), 1)
        self.assertEqual(has_ep[0]["from"], "podcast:the-journal")
        self.assertEqual(has_ep[0]["to"], "episode:ep:show")

    def test_v2_emission_skips_podcast_unknown_placeholder(self) -> None:
        """v2.0: skip Podcast node + HAS_EPISODE for the `podcast:unknown` placeholder."""
        art = build_artifact(
            "ep:nopod",
            "t",
            podcast_id="",  # build_artifact substitutes "podcast:unknown"
            episode_title="Ep",
        )
        validate_artifact(art, strict=True)
        podcasts = [n for n in art["nodes"] if n["type"] == "Podcast"]
        has_ep = [e for e in art["edges"] if e["type"] == "HAS_EPISODE"]
        self.assertEqual(podcasts, [])
        self.assertEqual(has_ep, [])

    def test_v2_emission_normalizes_unprefixed_podcast_id(self) -> None:
        """v2.0: bare `podcast_id` (no 'podcast:' prefix) is normalized for the node."""
        art = build_artifact(
            "ep:bare",
            "t",
            podcast_id="my-show",
            episode_title="Ep",
        )
        validate_artifact(art, strict=True)
        podcasts = [n for n in art["nodes"] if n["type"] == "Podcast"]
        self.assertEqual(len(podcasts), 1)
        self.assertEqual(podcasts[0]["id"], "podcast:my-show")
