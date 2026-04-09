"""Unit tests for append / resume (GitHub #444)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
import yaml

from podcast_scraper import Config, models
from podcast_scraper.workflow.append_resume import episode_complete_for_append_resume

pytestmark = [pytest.mark.unit]


def _episode(guid: str, idx: int = 1, title: str = "Ep", title_safe: str = "ep") -> models.Episode:
    item = ET.Element("item")
    ET.SubElement(item, "title").text = title
    ET.SubElement(item, "guid").text = guid
    return models.Episode(
        idx=idx,
        title=title,
        title_safe=title_safe,
        item=item,
        transcript_urls=[],
    )


@pytest.mark.unit
class TestEpisodeCompleteForAppendResume(unittest.TestCase):
    """Tests for episode_complete_for_append_resume."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.transcripts = os.path.join(self.tmp, "transcripts")
        self.metadata = os.path.join(self.tmp, "metadata")
        os.makedirs(self.transcripts, exist_ok=True)
        os.makedirs(self.metadata, exist_ok=True)
        self.feed_url = "https://example.com/podcast.xml"
        self.run_suffix = "append_deadbeef"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_transcript(self) -> str:
        rel = f"transcripts/0001 - ep_{self.run_suffix}.txt"
        path = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("hello")
        return rel

    def _write_metadata(self, episode_id: str, trel: str, **extra: object) -> None:
        doc: dict = {
            "episode": {"episode_id": episode_id},
            "content": {"transcript_file_path": trel},
        }
        doc.update(extra)
        meta_name = f"0001 - ep_{self.run_suffix}.metadata.json"
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            json.dump(doc, handle)

    def test_complete_when_metadata_and_transcript_match(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-one")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, trel)
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertTrue(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_incomplete_when_episode_id_mismatch(self) -> None:
        trel = self._write_transcript()
        self._write_metadata("wrong-id", trel)
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        ep = _episode("g-one")
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_incomplete_when_summary_required_but_missing(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-two")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, trel)
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_complete_when_summary_present(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-three")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(
            eid,
            trel,
            summary={"bullets": ["a"], "generated_at": "2026-01-01T00:00:00"},
        )
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertTrue(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_generate_metadata_off(self) -> None:
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        ep = _episode("g-four")
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_gi_enabled_but_artifact_missing(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-gi-miss")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(
            eid,
            trel,
            grounded_insights={"artifact_path": "metadata/missing.gi.json"},
        )
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_gi=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_true_when_gi_enabled_and_artifact_present(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-gi-ok")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        gi_rel = "metadata/x.gi.json"
        gi_path = os.path.join(self.tmp, gi_rel)
        os.makedirs(os.path.dirname(gi_path), exist_ok=True)
        with open(gi_path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        self._write_metadata(
            eid,
            trel,
            grounded_insights={"artifact_path": gi_rel},
        )
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_gi=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertTrue(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_kg_enabled_but_artifact_missing(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-kg-miss")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(
            eid,
            trel,
            knowledge_graph={"artifact_path": "metadata/missing.kg.json"},
        )
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_kg=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_true_when_kg_enabled_and_artifact_present(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-kg-ok")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        kg_rel = "metadata/x.kg.json"
        kg_path = os.path.join(self.tmp, kg_rel)
        os.makedirs(os.path.dirname(kg_path), exist_ok=True)
        with open(kg_path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        self._write_metadata(
            eid,
            trel,
            knowledge_graph={"artifact_path": kg_rel},
        )
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_kg=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertTrue(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_yaml_metadata_format_loaded(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-yaml")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        doc = {"episode": {"episode_id": eid}, "content": {"transcript_file_path": trel}}
        meta_name = f"0001 - ep_{self.run_suffix}.metadata.yaml"
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            yaml.safe_dump(doc, handle)

        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            metadata_format="yaml",
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertTrue(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_metadata_json_invalid(self) -> None:
        self._write_transcript()
        meta_name = f"0001 - ep_{self.run_suffix}.metadata.json"
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            handle.write("{ not json")

        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        ep = _episode("g-badjson")
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    @patch(
        "podcast_scraper.workflow.append_resume.find_episode_metadata_relative_path",
        return_value="",
    )
    def test_false_when_metadata_relative_path_unresolved(self, _mock: object) -> None:
        """``find_episode_metadata_relative_path`` empty → skip (line 77–78)."""
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        ep = _episode("g-nometa")
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_complete_when_summary_is_raw_text_only(self) -> None:
        """``_summary_looks_present`` accepts non-empty ``raw_text`` (lines 42–43)."""
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-rawsum")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(
            eid,
            trel,
            summary={"raw_text": " summary body ", "generated_at": "2026-01-01T00:00:00"},
        )
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertTrue(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_content_missing_or_not_dict(self) -> None:
        self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-nocontent")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        meta_name = f"0001 - ep_{self.run_suffix}.metadata.json"
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            json.dump({"episode": {"episode_id": eid}}, handle)

        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_transcript_rel_not_string(self) -> None:
        self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-treltype")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        meta_name = f"0001 - ep_{self.run_suffix}.metadata.json"
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            json.dump(
                {"episode": {"episode_id": eid}, "content": {"transcript_file_path": 123}},
                handle,
            )

        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_transcript_file_missing_on_disk(self) -> None:
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-notfile")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, "transcripts/does-not-exist.txt")
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_gi_block_not_dict(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-gitype")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, trel, grounded_insights="not-a-dict")
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_gi=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_gi_path_blank(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-giblank")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, trel, grounded_insights={"artifact_path": "  "})
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_gi=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_kg_block_not_dict(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-kgtype")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, trel, knowledge_graph=["x"])
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_kg=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_kg_path_blank(self) -> None:
        trel = self._write_transcript()
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep = _episode("g-kgblank")
        eid, _ = get_episode_id_from_episode(ep, self.feed_url)
        self._write_metadata(eid, trel, knowledge_graph={"artifact_path": ""})
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            generate_summaries=False,
            generate_kg=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )

    def test_false_when_yaml_metadata_not_mapping(self) -> None:
        meta_name = f"0001 - ep_{self.run_suffix}.metadata.yaml"
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            handle.write("- not\n- a\n- mapping\n")

        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            generate_metadata=True,
            metadata_format="yaml",
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        ep = _episode("g-yamllist")
        self.assertFalse(
            episode_complete_for_append_resume(cfg, ep, self.feed_url, self.tmp, self.run_suffix)
        )
