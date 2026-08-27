"""THE NIGHTLY CONTRACT: a second batch run over an unchanged corpus ingests NOTHING.

2026-08-27: batch-mode skip_existing was blind to prior runs (flag-gated corpus lookup;
both multi-feed loops — cli.py's own, the prod nightly path, and ``service.run_multi_feed``
— rebase child ``output_dir`` to ``<corpus>/feeds/<slug>`` with the flag off) and the nightly
re-ingested its whole window every fire. TWO hand-modelled fixes in a row validated their own
assumptions instead of production's paths, so this test asks the real pipeline instead: it
drives the ACTUAL batch entry (``cli.main`` with multiple ``--rss``) twice with only HTTP
mocked, and asserts run 2 fetches zero transcripts. No hand-built fixtures, no monkeypatched
internals — a wrong belief about directory shapes cannot make this pass.
"""

from __future__ import annotations

import glob

# Load the top-level tests/conftest.py under a unique module name — a bare `import conftest`
# collides with whichever conftest pytest imported first in a multi-directory run.
import importlib.util  # noqa: E402
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import cli
from podcast_scraper.rss import downloader

_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
parent_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent_conftest)

create_rss_response = parent_conftest.create_rss_response
create_transcript_response = parent_conftest.create_transcript_response


def _rss_with_guid(title: str, guid: str, transcript_url: str) -> str:
    """One-item feed WITH a <guid> — corpus presence is guid-keyed (run_index._episode_guid),
    matching prod (705/705 episodes carry guids). KNOWN LIMITATION, deliberately not asserted
    here: a guid-LESS episode can never be skipped under corpus layout (fresh run dir every
    run + guid-keyed index) — that hole predates batch mode and needs an index change, not a
    gate change.
    """
    return f"""<?xml version='1.0'?>
<rss xmlns:podcast=\"https://podcastindex.org/namespace/1.0\">
  <channel>
    <title>{title}</title>
    <item>
      <title>Episode 1</title>
      <guid>{guid}</guid>
      <podcast:transcript url=\"{transcript_url}\" type=\"text/plain\" />
    </item>
  </channel>
</rss>""".strip()


@pytest.mark.integration
class TestBatchRerunIngestsNothing(unittest.TestCase):
    RSS_A = "https://alpha.example/feed.xml"
    RSS_B = "https://beta.example/feed.xml"
    TR_A = "https://alpha.example/ep1.txt"
    TR_B = "https://beta.example/ep1.txt"

    def _run_batch(self, corpus: str, fetch_counts: dict) -> int:
        """One real batch run (two feeds) with counted, mocked HTTP."""
        responses = {
            downloader.normalize_url(self.RSS_A): lambda: create_rss_response(
                _rss_with_guid("Feed A", "guid-alpha-ep1", self.TR_A), self.RSS_A
            ),
            downloader.normalize_url(self.TR_A): lambda: create_transcript_response(
                "Alpha transcript", self.TR_A
            ),
            downloader.normalize_url(self.RSS_B): lambda: create_rss_response(
                _rss_with_guid("Feed B", "guid-beta-ep1", self.TR_B), self.RSS_B
            ),
            downloader.normalize_url(self.TR_B): lambda: create_transcript_response(
                "Beta transcript", self.TR_B
            ),
        }

        def _side_effect(url, user_agent, timeout, stream=False):
            normalized = downloader.normalize_url(url)
            factory = responses.get(normalized)
            if factory is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            fetch_counts[normalized] = fetch_counts.get(normalized, 0) + 1
            return factory()

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=_side_effect),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=_side_effect),
        ):
            return cli.main(
                [
                    self.RSS_A,
                    "--rss",
                    self.RSS_B,
                    "--output-dir",
                    corpus,
                    "--no-auto-speakers",
                    "--skip-existing",
                ]
            )

    def _transcript_files(self, corpus: str) -> list[str]:
        return sorted(
            glob.glob(os.path.join(corpus, "feeds", "*", "run_*", "transcripts", "*.txt"))
        )

    def test_second_batch_run_over_unchanged_feeds_ingests_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as corpus:
            counts_run1: dict = {}
            self.assertEqual(self._run_batch(corpus, counts_run1), 0)
            tr_a = downloader.normalize_url(self.TR_A)
            tr_b = downloader.normalize_url(self.TR_B)
            self.assertEqual(
                (counts_run1.get(tr_a), counts_run1.get(tr_b)),
                (1, 1),
                f"run 1 must ingest both episodes; fetches: {counts_run1!r}",
            )
            files_after_run1 = self._transcript_files(corpus)
            self.assertEqual(len(files_after_run1), 2, files_after_run1)

            # Prod runs each batch as its own subprocess (server/jobs.py spawns
            # `python -m podcast_scraper.cli`), so the process-scoped corpus index cache
            # never spans two runs there. Both runs share this test process — reset the
            # cache to reproduce the real process boundary, nothing else.
            from podcast_scraper.workflow import run_index

            run_index.reset_corpus_metadata_index_cache_for_tests()

            counts_run2: dict = {}
            self.assertEqual(self._run_batch(corpus, counts_run2), 0)
            transcript_fetches_run2 = counts_run2.get(tr_a, 0) + counts_run2.get(tr_b, 0)
            self.assertEqual(
                transcript_fetches_run2,
                0,
                "SECOND run re-fetched transcripts — batch skip_existing is blind to prior "
                f"runs again (the nightly re-ingest bug). fetches: {counts_run2!r}",
            )
            self.assertEqual(
                self._transcript_files(corpus),
                files_after_run1,
                "second run wrote new transcript files for episodes already in the corpus",
            )
