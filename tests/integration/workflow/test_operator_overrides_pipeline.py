"""Operator overrides reach the pipeline, and no language means NOTHING is downloaded (#2283).

End to end through ``cli.main`` with HTTP mocked, so what is asserted is what the pipeline did:
which URLs it fetched and which files it wrote.

The feed here carries a PUBLISHER TRANSCRIPT on purpose. That path used to bypass the language
gate entirely (the gate ran only inside transcription), so it is the case most able to leak a
download for a feed with no language.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

from podcast_scraper import cli, downloader
from tests.conftest import create_rss_response, create_transcript_response  # noqa: E402

pytestmark = [pytest.mark.integration]

RSS_URL = "https://example.com/untagged.xml"
TRANSCRIPT_URL = "https://example.com/ep1.txt"
GUID = "ep-guid-1"


def _rss(language: Optional[str]) -> str:
    lang = f"\n    <language>{language}</language>" if language is not None else ""
    return f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Override Feed</title>{lang}
    <item>
      <title>Episode 1</title>
      <guid isPermaLink="false">{GUID}</guid>
      <podcast:transcript url="{TRANSCRIPT_URL}" type="text/plain" />
    </item>
  </channel>
</rss>""".strip()


class TestOperatorOverridesInThePipeline(unittest.TestCase):
    def _run(self, rss_xml: str, tmpdir: str, extra: Optional[List[str]] = None) -> List[str]:
        """Run the CLI over a mocked feed; returns every URL the pipeline fetched."""
        fetched: List[str] = []
        responses: Dict[str, object] = {
            downloader.normalize_url(RSS_URL): create_rss_response(rss_xml, RSS_URL),
            downloader.normalize_url(TRANSCRIPT_URL): create_transcript_response(
                "Episode 1 transcript", TRANSCRIPT_URL
            ),
        }

        def _http(url, user_agent, timeout, stream=False):
            norm = downloader.normalize_url(url)
            fetched.append(norm)
            if norm not in responses:
                raise AssertionError(f"Unexpected HTTP request: {norm}")
            return responses[norm]

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=_http),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=_http),
        ):
            exit_code = cli.main(
                [RSS_URL, "--output-dir", tmpdir, "--no-auto-speakers", *(extra or [])]
            )
        self.assertEqual(exit_code, 0)
        return fetched

    def _transcripts(self, tmpdir: str) -> List[str]:
        return glob.glob(os.path.join(tmpdir, "run_*", "transcripts", "*.txt"))

    def _write_overrides(self, tmpdir: str, doc: dict) -> None:
        Path(tmpdir, "overrides.json").write_text(json.dumps(doc), encoding="utf-8")

    def test_a_feed_with_no_language_downloads_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fetched = self._run(_rss(None), tmpdir)
            self.assertNotIn(downloader.normalize_url(TRANSCRIPT_URL), fetched)
            self.assertEqual(self._transcripts(tmpdir), [])

    def test_a_tag_that_is_not_an_iso_code_downloads_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fetched = self._run(_rss("und"), tmpdir)
            self.assertNotIn(downloader.normalize_url(TRANSCRIPT_URL), fetched)
            self.assertEqual(self._transcripts(tmpdir), [])

    def test_a_feed_language_override_lets_it_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_overrides(tmpdir, {"feeds": {RSS_URL: {"fields": {"language": "en"}}}})
            fetched = self._run(_rss(None), tmpdir)
            self.assertIn(downloader.normalize_url(TRANSCRIPT_URL), fetched)
            self.assertEqual(len(self._transcripts(tmpdir)), 1)

    def test_an_episode_language_override_lets_only_that_episode_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_overrides(
                tmpdir,
                {"feeds": {RSS_URL: {"episodes": {GUID: {"language": "en"}}}}},
            )
            fetched = self._run(_rss(None), tmpdir)
            self.assertIn(downloader.normalize_url(TRANSCRIPT_URL), fetched)

    def test_a_disabled_language_downloads_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fetched = self._run(_rss("ja"), tmpdir)
            self.assertNotIn(downloader.normalize_url(TRANSCRIPT_URL), fetched)

    def test_an_episode_title_override_reaches_the_written_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_overrides(
                tmpdir,
                {"feeds": {RSS_URL: {"episodes": {GUID: {"title": "Corrected Title"}}}}},
            )
            self._run(_rss("en"), tmpdir, ["--generate-metadata"])
            metas = glob.glob(os.path.join(tmpdir, "run_*", "metadata", "*.metadata.json"))
            self.assertEqual(len(metas), 1, metas)
            doc = json.loads(Path(metas[0]).read_text(encoding="utf-8"))
            self.assertEqual(doc["episode"]["title"], "Corrected Title")
            # The file name is NOT renamed: an override must not orphan existing artifacts.
            self.assertIn("Episode 1", os.path.basename(self._transcripts(tmpdir)[0]))

    def test_a_broken_overrides_file_stops_the_run_rather_than_being_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "overrides.json").write_text("{not json", encoding="utf-8")
            fetched: List[str] = []
            with self.assertRaises(Exception):
                self._run_raw(tmpdir, fetched)
            self.assertNotIn(downloader.normalize_url(TRANSCRIPT_URL), fetched)

    def _run_raw(self, tmpdir: str, fetched: List[str]) -> int:
        responses = {
            downloader.normalize_url(RSS_URL): create_rss_response(_rss("en"), RSS_URL),
        }

        def _http(url, user_agent, timeout, stream=False):
            norm = downloader.normalize_url(url)
            fetched.append(norm)
            return responses[norm]

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=_http),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=_http),
        ):
            code = cli.main([RSS_URL, "--output-dir", tmpdir, "--no-auto-speakers"])
        if code != 0:
            raise RuntimeError(f"run exited {code}")
        return code
