# mypy: disable-error-code="call-arg"
"""W2 (#1874): in ONE batch, a pinned feed routes differently from its neighbour.

This is the contract the DGX onboarding rests on — new feeds transcribe on the DGX while the
proven feeds stay on Deepgram, in the same nightly run — and nothing tested it end to end.
The unit tests covered ``merge_feed_entry_into_config`` in isolation; the argv tests covered
``build_pipeline_argv``; the seam between them (does the batch loop actually give each feed
its own resolved config?) was untested, and that seam is where the 2026-08-28 bugs lived.

It also pins the two properties that were silently broken and fixed the same day:
  * the run context must not LEAK from the pinned feed onto its neighbour (a plain feed
    reported the pinned feed's ASR fallback), and
  * a per-request override must beat a pin when the operator asks for it.

Deliberately asserts the RESOLVED CONFIG each feed runs with, captured at the boundary the
pipeline is actually invoked through — not internals of the merge helper, which is what let
a 15-field precedence inversion pass three green spot-assertions.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration]

PINNED = "https://pinned.example/feed.xml"
PLAIN = "https://plain.example/feed.xml"


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
    monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")
    (tmp_path / "feeds.spec.yaml").write_text(
        textwrap.dedent(f"""
            feeds:
              - {PLAIN}
              - url: {PINNED}
                profile: cloud_with_dgx_primary
            """).strip(),
        encoding="utf-8",
    )
    return tmp_path


def _configs_the_batch_would_run(corpus: Path, base_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve each feed exactly as the multi-feed loop does, and return {url: config}."""
    from podcast_scraper import config as config_mod
    from podcast_scraper.rss.feeds_spec import load_feeds_spec_file, merge_feed_entry_into_config

    base = config_mod.Config(
        rss_url="https://placeholder.example/f", profile="cloud_balanced", **base_kwargs
    )
    doc = load_feeds_spec_file(str(corpus / "feeds.spec.yaml"))
    return {entry.url: merge_feed_entry_into_config(base, entry) for entry in doc.feeds}


class TestOneBatchTwoRoutes:
    def test_pinned_and_plain_feeds_get_different_asr_in_the_same_batch(self, corpus) -> None:
        resolved = _configs_the_batch_would_run(corpus, {})

        assert resolved[PLAIN].transcription_provider == "deepgram"
        assert (
            resolved[PINNED].transcription_provider == "tailnet_dgx_whisper"
        ), "the pinned feed did not route to the DGX — the whole point of per-feed pinning"

    def test_the_pin_does_not_bleed_onto_its_neighbour(self, corpus) -> None:
        """Order matters: the pinned feed is resolved first, then the plain one."""
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        base = config_mod.Config(rss_url="https://placeholder.example/f", profile="cloud_balanced")
        merge_feed_entry_into_config(
            base, RssFeedEntry(url=PINNED, profile="cloud_with_dgx_primary")
        )
        after = merge_feed_entry_into_config(base, RssFeedEntry(url=PLAIN))

        assert after.transcription_provider == "deepgram"
        assert after.profile == "cloud_balanced", (
            "resolving a pinned feed mutated the shared base config — every later feed in the "
            "batch would inherit the pin"
        )

    def test_each_feed_keeps_the_corpus_settings_the_pin_is_silent_about(self, corpus) -> None:
        resolved = _configs_the_batch_would_run(corpus, {"cost_soft_cap_usd_per_run": 25.0})
        for url, cfg in resolved.items():
            assert cfg.cost_soft_cap_usd_per_run == 25.0, (
                f"{url} lost the corpus cost cap — a pin overlays routing, it does not reset "
                "the deployment"
            )

    def test_a_request_override_beats_the_pin_for_every_feed(self, corpus) -> None:
        resolved = _configs_the_batch_would_run(corpus, {"profile_overrides_feed_pins": True})
        assert resolved[PINNED].transcription_provider == "deepgram", (
            "the pin ignored an explicit per-request override (#1872 cascade: corpus < pin < "
            "request)"
        )
        assert resolved[PLAIN].transcription_provider == "deepgram"


class TestRunContextTracksTheFeedBeingProcessed:
    """Events from feed B must not carry feed A's routing (the 2026-08-28 leak)."""

    def test_context_is_replaced_per_feed_not_merged(self, corpus) -> None:
        """Drive the REAL per-run setup for feed A then feed B; B must show only B's routing.

        Calls ``stamp_run_identity`` — the function run_pipeline itself calls — rather than
        re-implementing "clear then set" in the test. An earlier draft did the latter and
        happily passed with the production fix reverted: it proved the test knew the rule, not
        that the code followed it. Asserting EQUALITY, not the absence of one known key, because
        the leak carried whichever keys B's config left unset.
        """
        from podcast_scraper.obs.events import (
            get_run_context,
            run_context_from_config,
        )
        from podcast_scraper.workflow.orchestration import stamp_run_identity

        resolved = _configs_the_batch_would_run(corpus, {})

        stamp_run_identity(resolved[PINNED])
        assert get_run_context()["asr_provider"] == "tailnet_dgx_whisper"

        stamp_run_identity(resolved[PLAIN])
        expected_plain = run_context_from_config(resolved[PLAIN])

        assert get_run_context() == expected_plain, (
            "the plain feed's run context carried residue from the pinned feed — every event, "
            "span and error it raises would report routing it never used"
        )
        assert get_run_context()["profile"] == "cloud_balanced"

    def test_correlation_feed_and_profile_track_the_current_feed(self, corpus) -> None:
        """Logs and Sentry read these; they must follow the feed, not the first one seen."""
        from podcast_scraper.utils import correlation
        from podcast_scraper.workflow.orchestration import stamp_run_identity

        resolved = _configs_the_batch_would_run(corpus, {})

        stamp_run_identity(resolved[PINNED])
        assert correlation.get_feed_id() == PINNED
        assert correlation.get_profile() == "cloud_with_dgx_primary"

        stamp_run_identity(resolved[PLAIN])
        assert correlation.get_feed_id() == PLAIN
        assert correlation.get_profile() == "cloud_balanced"
