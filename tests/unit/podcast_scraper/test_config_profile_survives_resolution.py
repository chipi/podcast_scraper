"""The resolved profile name must survive Config resolution (#1648).

For the entire life of the corpus, every enrichment run was a 3 ms no-op reporting success.
The cause was one missing field. The chain:

    config.py  ``profile_name = data.pop("profile", None)``   name known here
    config.py  ``profile_dict.pop("profile", None)``          discarded (extra="forbid")
    orchestration.py  ``getattr(cfg, "profile", None)``       -> None
    profile_sets.py   ``enricher_set_for_profile(None)``      -> EMPTY set
    executor                                                  -> ran nothing, reported ok

``orchestration.py`` needs the name twice: to choose the enricher set, and to put
``--profile`` on the argv of the enrichment CHILD PROCESS — the only way the name crosses
that boundary. Both silently received None.

These tests pin the field, not the plumbing, so any future refactor that drops the name again
fails here rather than in production two months later.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config as config_mod

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _no_ambient_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tests/conftest.py`` sets ``PODCAST_SCRAPER_PROFILE=test_default`` for the whole run.

    That ambient value is exactly the fallback under test, so it has to be cleared per test —
    otherwise "no profile supplied" silently becomes "test_default supplied" and the None case
    can never be observed.
    """
    monkeypatch.delenv("PODCAST_SCRAPER_PROFILE", raising=False)


class TestProfileNameIsRetained:
    def test_explicit_profile_is_stored_on_the_config(self) -> None:
        cfg = config_mod.Config.model_validate(
            {"rss_url": "https://example.com/feed.xml", "profile": "test_default"}
        )
        assert cfg.profile == "test_default"

    def test_no_profile_leaves_the_field_none(self) -> None:
        cfg = config_mod.Config.model_validate({"rss_url": "https://example.com/feed.xml"})
        assert cfg.profile is None

    def test_env_var_profile_is_also_retained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """conftest wires the test profile this way, and prod can set it per deployment."""
        monkeypatch.setenv("PODCAST_SCRAPER_PROFILE", "test_default")
        cfg = config_mod.Config.model_validate({"rss_url": "https://example.com/feed.xml"})
        assert cfg.profile == "test_default"

    def test_explicit_profile_beats_the_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_SCRAPER_PROFILE", "test_default")
        cfg = config_mod.Config.model_validate(
            {"rss_url": "https://example.com/feed.xml", "profile": "no_such_profile_xyz"}
        )
        assert cfg.profile == "no_such_profile_xyz"

    def test_an_unknown_profile_name_is_still_retained(self) -> None:
        """Resolution logs and ignores an unknown profile — but the NAME must not vanish.

        Losing it here would reproduce the original defect for exactly the case an operator
        most needs to debug: they named a profile and nothing happened.
        """
        cfg = config_mod.Config.model_validate(
            {"rss_url": "https://example.com/feed.xml", "profile": "no_such_profile_xyz"}
        )
        assert cfg.profile == "no_such_profile_xyz"


class TestProfileReachesEnricherResolution:
    """The consumer contract: a stored profile yields a NON-empty enricher set."""

    def test_cloud_balanced_resolves_a_non_empty_enricher_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact production path: cloud_balanced -> Config.profile -> enricher set.

        cloud_balanced transcribes with Deepgram and validates that its key is present, so a
        dummy is supplied — the assertion is about profile plumbing, not credentials.
        """
        from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile

        for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
            monkeypatch.setenv(key, f"test-{key.lower()}-dummy")

        cfg = config_mod.Config.model_validate(
            {"rss_url": "https://example.com/feed.xml", "profile": "cloud_balanced"}
        )
        assert cfg.profile == "cloud_balanced"
        enricher_set = enricher_set_for_profile(getattr(cfg, "profile", None))
        assert (
            enricher_set.enabled_enrichers
        ), "cloud_balanced resolved to zero enrichers — this is the #1648 no-op returning"

    def test_the_none_case_still_yields_the_empty_set(self) -> None:
        """The empty set for None is CORRECT; the bug was never getting a name to pass."""
        from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile

        assert not enricher_set_for_profile(None).enabled_enrichers
