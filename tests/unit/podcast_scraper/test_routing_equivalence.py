# mypy: disable-error-code="call-arg"
"""W3 (#1874): the paths that must agree, asserted against each other.

Every bug this session lived in a DISAGREEMENT between two things that were each individually
tested and individually green:

  * per-feed resolution vs top-level resolution (a 15-field precedence inversion)
  * a profile's nested ``transcription:`` block vs its flat fallback ladder (a stale openai)
  * batch argv vs single-feed argv for the same feed (a pin honoured one way, ignored the other)

Spot assertions cannot see a disagreement — both sides pass their own checks. Only comparing
the two paths does. These are cheap because neither side needs a fixture: they are pure
resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

PINNED = "https://pinned.example/feed.xml"
PLAIN = "https://plain.example/feed.xml"
PIN_PROFILE = "cloud_with_dgx_primary"


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch) -> Path:
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "dummy-for-validation")
    monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")
    (tmp_path / "viewer_operator.yaml").write_text(
        "profile: cloud_balanced\nmax_episodes: 4\n", encoding="utf-8"
    )
    (tmp_path / "feeds.spec.yaml").write_text(
        f"feeds:\n  - {PLAIN}\n  - url: {PINNED}\n    profile: {PIN_PROFILE}\n",
        encoding="utf-8",
    )
    return tmp_path


def _profile_in(argv: list[str]) -> str | None:
    return argv[argv.index("--profile") + 1] if "--profile" in argv else None


class TestBatchAndSingleFeedAgree:
    """The same feed, launched two ways, must run on the same profile.

    They are built by different code: the batch resolves pins inside
    ``merge_feed_entry_into_config``; the single-feed path resolves them in
    ``build_pipeline_argv``. Two implementations of one rule is how they drift.
    """

    def test_pinned_feed_gets_its_pin_either_way(self, corpus: Path) -> None:
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import (
            load_feeds_spec_file,
            merge_feed_entry_into_config,
        )
        from podcast_scraper.server.jobs import build_pipeline_argv

        single_argv = build_pipeline_argv(corpus, corpus / "viewer_operator.yaml", feed_url=PINNED)
        via_single = _profile_in(single_argv)

        base = config_mod.Config(rss_url="https://placeholder.example/f", profile="cloud_balanced")
        entry = next(
            e
            for e in load_feeds_spec_file(str(corpus / "feeds.spec.yaml")).feeds
            if e.url == PINNED
        )
        via_batch = merge_feed_entry_into_config(base, entry).profile

        assert via_single == via_batch == PIN_PROFILE, (
            f"the same feed runs on different profiles depending on how it was launched: "
            f"single-feed={via_single!r} batch={via_batch!r}"
        )

    def test_unpinned_feed_gets_the_corpus_profile_either_way(self, corpus: Path) -> None:
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import (
            load_feeds_spec_file,
            merge_feed_entry_into_config,
        )
        from podcast_scraper.server.jobs import build_pipeline_argv

        via_single = _profile_in(
            build_pipeline_argv(corpus, corpus / "viewer_operator.yaml", feed_url=PLAIN)
        )
        base = config_mod.Config(rss_url="https://placeholder.example/f", profile="cloud_balanced")
        entry = next(
            e for e in load_feeds_spec_file(str(corpus / "feeds.spec.yaml")).feeds if e.url == PLAIN
        )
        via_batch = merge_feed_entry_into_config(base, entry).profile

        assert via_single == via_batch == "cloud_balanced"

    def test_a_request_override_wins_either_way(self, corpus: Path) -> None:
        """Single-feed applies it directly; batch needs the explicit flag. Same outcome."""
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import (
            load_feeds_spec_file,
            merge_feed_entry_into_config,
        )
        from podcast_scraper.server.jobs import build_pipeline_argv

        via_single = _profile_in(
            build_pipeline_argv(
                corpus,
                corpus / "viewer_operator.yaml",
                feed_url=PINNED,
                profile_override="cloud_thin",
            )
        )
        base = config_mod.Config(
            rss_url="https://placeholder.example/f",
            profile="cloud_thin",
            profile_overrides_feed_pins=True,
        )
        entry = next(
            e
            for e in load_feeds_spec_file(str(corpus / "feeds.spec.yaml")).feeds
            if e.url == PINNED
        )
        via_batch = merge_feed_entry_into_config(base, entry).profile

        assert via_single == via_batch == "cloud_thin", (
            f"an override is honoured one way and not the other: single={via_single!r} "
            f"batch={via_batch!r}"
        )


class TestProfileInternalRepresentationsAgree:
    """A profile that states its routing twice must state it the same way both times."""

    @staticmethod
    def _profiles() -> list[str]:
        here = Path(__file__).resolve()
        for parent in here.parents:
            d = parent / "config" / "profiles"
            if d.is_dir():
                return [p.stem for p in sorted(d.glob("*.yaml")) if not p.stem.endswith(".example")]
        raise AssertionError("config/profiles not found")

    def test_a_ladder_that_names_the_primary_must_name_it_first(self, monkeypatch) -> None:
        """A ladder listing the primary must list it FIRST — never behind a fallback.

        Deliberately narrower than "the ladder starts with the primary". The convention is not
        universal and asserting it would be wrong: ``eval_default`` declares primary
        ``tailnet_dgx_whisper`` with ladder ``['whisper']`` — a pure list of what to try AFTER
        the primary, which is a legitimate second convention. What is never legitimate is a
        ladder that mentions the primary somewhere behind a fallback, i.e. that says the run
        degrades to something else before trying its own primary.

        (The stronger cross-check — the single ``transcription_fallback_provider`` field must
        appear in the ladder — lives in W1's profile suite.)
        """
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "DEEPGRAM_API_KEY",
            "DEEPSEEK_API_KEY",
            "GROQ_API_KEY",
            "GROK_API_KEY",
            "MISTRAL_API_KEY",
            "LITELLM_API_KEY",
        ):
            monkeypatch.setenv(key, "dummy-for-validation")
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")

        from podcast_scraper import config as config_mod

        mismatched: dict[str, tuple[str, list[str]]] = {}
        for name in self._profiles():
            cfg = config_mod.Config(rss_url="https://example.test/f", profile=name)
            primary = (getattr(cfg, "transcription_provider", None) or "").strip()
            ladder = [
                str(p).strip()
                for p in (getattr(cfg, "transcription_fallback_providers", None) or [])
                if str(p).strip()
            ]
            if primary and primary in ladder and ladder[0] != primary:
                mismatched[name] = (primary, ladder)

        assert not mismatched, (
            "profiles whose ladder reaches their own primary only AFTER a fallback — the "
            f"documented degradation order is not the one that runs: {mismatched}"
        )
