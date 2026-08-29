# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — alias="rss"; populate-by-name accepts either at runtime.
"""Unit tests for structured feeds spec (#626)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from podcast_scraper import config as cfg_mod
from podcast_scraper.rss.feeds_spec import (
    FeedsSpecDocument,
    load_feeds_spec_file,
    merge_feed_entry_into_config,
    RSS_FEED_ENTRY_OVERRIDE_KEYS,
    RssFeedEntry,
)


def test_load_feeds_spec_yaml_and_json_equivalent(tmp_path: Path) -> None:
    y = tmp_path / "f.yaml"
    j = tmp_path / "f.json"
    data = {
        "feeds": [
            "https://a.example/feed.xml",
            {"url": "https://b.example/feed.xml", "timeout": 99},
        ]
    }
    y.write_text(yaml.safe_dump(data), encoding="utf-8")
    j.write_text(json.dumps(data), encoding="utf-8")
    dy = load_feeds_spec_file(y)
    dj = load_feeds_spec_file(j)
    assert dy.model_dump() == dj.model_dump()
    assert dy.feeds[1].timeout == 99


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        yaml.safe_dump({"feeds": ["https://a.example/x"], "extra_root": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level"):
        load_feeds_spec_file(p)


def test_feed_entry_unknown_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        RssFeedEntry.model_validate({"url": "https://a.example/x", "not_allowed": 1})


def test_merge_feed_entry_overrides_global_timeout() -> None:
    base = cfg_mod.Config(
        rss="https://ignored.example/x",
        timeout=30,
        user_agent="global-ua",
    )
    ent = RssFeedEntry(url="https://feed.example/rss", timeout=77, user_agent=None)
    merged = merge_feed_entry_into_config(base, ent)
    assert merged.rss_url == "https://feed.example/rss"
    assert merged.timeout == 77
    assert merged.user_agent == "global-ua"
    assert merged.rss_urls is None


def test_merge_feed_entry_overrides_known_hosts_per_feed() -> None:
    # Step B: a network feed can name its recurring hosts that never self-introduce.
    base = cfg_mod.Config(rss="https://ignored.example/x", known_hosts=["Global Host"])
    ent = RssFeedEntry(
        url="https://feed.example/rss", known_hosts=["Erika Barris", "Nick Fountain"]
    )
    merged = merge_feed_entry_into_config(base, ent)
    assert merged.known_hosts == ["Erika Barris", "Nick Fountain"]  # feed overrides global
    # A feed that omits known_hosts inherits the global list.
    inherit = merge_feed_entry_into_config(base, RssFeedEntry(url="https://feed.example/rss"))
    assert inherit.known_hosts == ["Global Host"]


def test_merge_feed_entry_show_centric_per_feed() -> None:
    # A news-desk feed marks itself show-centric so an unnamed "Host" is expected, not a failure.
    base = cfg_mod.Config(rss="https://ignored.example/x")
    assert base.show_centric is False
    merged = merge_feed_entry_into_config(
        base, RssFeedEntry(url="https://feed.example/rss", show_centric=True)
    )
    assert merged.show_centric is True
    inherit = merge_feed_entry_into_config(base, RssFeedEntry(url="https://feed.example/rss"))
    assert inherit.show_centric is False


def test_merge_feed_entry_diarization_min_segment_ms_per_feed() -> None:
    # A news-desk feed with no real cameos can squelch phantom micro-speakers harder (#1170).
    base = cfg_mod.Config(rss="https://ignored.example/x")
    merged = merge_feed_entry_into_config(
        base, RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=1500)
    )
    assert merged.diarization_min_segment_ms == 1500
    # omitted -> inherits the base (global) value
    inherit = merge_feed_entry_into_config(base, RssFeedEntry(url="https://feed.example/rss"))
    assert inherit.diarization_min_segment_ms == base.diarization_min_segment_ms
    # in the override-key allowlist so the merge propagates it
    assert "diarization_min_segment_ms" in RSS_FEED_ENTRY_OVERRIDE_KEYS


def test_feed_entry_diarization_min_segment_ms_range_validated() -> None:
    # Field validator: 0 <= ms <= 60000.
    RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=0)
    RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=60000)
    with pytest.raises(ValidationError):
        RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=60001)
    with pytest.raises(ValidationError):
        RssFeedEntry(url="https://feed.example/rss", diarization_min_segment_ms=-1)


def test_feeds_spec_document_accepts_comment_keys() -> None:
    doc = FeedsSpecDocument.model_validate(
        {"_comment": "x", "_comment_resilience": "y", "feeds": ["https://a.example/z"]}
    )
    assert len(doc.feeds) == 1


@pytest.mark.unit
def test_repo_example_feeds_specs_load() -> None:
    """Tracked ``config/examples/feeds.spec.example.*`` stay valid feeds-spec documents."""
    root = Path(__file__).resolve().parents[4]
    examples_dir = root / "config" / "examples"
    for name in ("feeds.spec.example.yaml", "feeds.spec.example.json"):
        p = examples_dir / name
        if not p.is_file():
            pytest.skip(f"{p} not present")
        doc = load_feeds_spec_file(p)
        assert doc.feeds, f"{p.name}: expected at least one feed"


class TestPerFeedProfileOverride:
    """A feed may name its own deployment profile (2026-08-28, DGX onboarding).

    A batch run applies ONE profile to every feed, so onboarding feeds onto the DGX while the
    proven feeds stay on Deepgram needs per-feed routing. The override must RESOLVE the
    profile: ``model_copy(update=...)`` skips validators, so assigning the name alone would
    relabel the config and route nothing — the failure mode these tests exist to forbid.
    """

    @pytest.fixture()
    def base(self, monkeypatch):
        """cloud_balanced batch config. The key is monkeypatched, NOT os.environ.setdefault —
        that leaks into the process and breaks the tests asserting a MISSING deepgram key."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
        from podcast_scraper import config as config_mod

        return config_mod.Config(
            rss_url="https://placeholder.example/f", profile="cloud_balanced", max_episodes=10
        )

    def test_named_profile_actually_changes_routing(self, base):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        assert base.transcription_provider == "deepgram"  # batch default
        sub = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://new.example/f", profile="cloud_with_dgx_primary")
        )
        assert sub.profile == "cloud_with_dgx_primary"
        assert sub.transcription_provider == "tailnet_dgx_whisper", (
            "the per-feed profile relabelled the config without re-routing — the exact "
            "model_copy-skips-validators no-op this feature must not have"
        )

    def test_feed_without_profile_is_untouched(self, base):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        sub = merge_feed_entry_into_config(base, RssFeedEntry(url="https://old.example/f"))
        assert sub.profile == "cloud_balanced"
        assert sub.transcription_provider == "deepgram"

    def test_entry_own_overrides_win_over_the_named_profile(self, base):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        sub = merge_feed_entry_into_config(
            base,
            RssFeedEntry(
                url="https://new.example/f", profile="cloud_with_dgx_primary", max_episodes=3
            ),
        )
        assert sub.transcription_provider == "tailnet_dgx_whisper"
        assert sub.max_episodes == 3

    def test_unknown_profile_warns_and_keeps_the_batch_route(self, base, caplog):
        from podcast_scraper.rss.feeds_spec import (
            merge_feed_entry_into_config,
            RssFeedEntry,
        )

        with caplog.at_level("WARNING"):
            sub = merge_feed_entry_into_config(
                base, RssFeedEntry(url="https://x.example/f", profile="no-such-profile")
            )
        assert sub.transcription_provider == "deepgram"
        assert "matched no registry preset" in caplog.text


class TestPerFeedProfileEquivalence:
    """A pinned feed must resolve to what a top-level ``profile:`` would produce.

    THE TEST THAT WAS MISSING (2026-08-28 review). The original per-feed tests asserted three
    spot fields and passed while FIFTEEN others were wrong — the profile's layers were applied
    over the operator's explicit values instead of under them, so a pinned prod feed silently
    swapped litellm_api_base for the profile YAML's default, and the audio-preprocessing preset
    was reported but never applied. Spot assertions cannot see that class of bug; equivalence
    against the reference path can, and does so for every field at once.
    """

    REFERENCE_EXPLICIT = {
        "litellm_api_base": "http://100.64.0.7:4001/v1",
        "cost_soft_cap_usd_per_run": 25.0,
        "max_episodes": 10,
    }

    @pytest.fixture()
    def base(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
        from podcast_scraper import config as config_mod

        return config_mod.Config(
            rss_url="https://placeholder.example/f",
            profile="cloud_balanced",
            **self.REFERENCE_EXPLICIT,
        )

    @pytest.mark.parametrize(
        "base_profile,pin_profile",
        [
            ("cloud_balanced", "cloud_with_dgx_primary"),
            # The pairs an adversarial sweep proved broken while this test was green,
            # because it only ever exercised the pair above — which happens to be the one
            # shape (flat base, materialized pin) the old classifier handled.
            ("cloud_qwen", "cloud_balanced"),
            ("prod_dgx_full", "cloud_balanced"),
            ("cloud_balanced", "cloud_thin"),
            ("cloud_thin", "cloud_with_dgx_primary"),
        ],
    )
    def test_every_field_the_pin_speaks_to_matches_top_level_resolution(
        self, base_profile: str, pin_profile: str, monkeypatch
    ):
        """Equivalence where equivalence is the contract, across REAL profile pairs.

        Two lessons are baked into this test's shape. First, it is not a whole-config diff:
        an earlier version demanded that and failed on storage/retry/cost settings the CORPUS
        profile owns and the pin never mentions — a pin overlays routing, it does not reset
        the deployment. Second, it is PARAMETRIZED: the single-pair version passed while 231
        ordered profile pairs were silently un-pinned, because the one pair it used was the
        only shape the classifier got right.

        The comparison covers the nested ``transcription:`` sugar too — ~20 profiles express
        routing that way, the layer resolver does not flatten it, and that mismatch was the
        core of the bug.
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
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
        ):
            monkeypatch.setenv(key, "dummy-for-validation")
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")

        from podcast_scraper import config as config_mod
        from podcast_scraper.config import resolve_profile_layers
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        url = "https://pinned.example/f.xml"
        base = config_mod.Config(rss_url="https://placeholder.example/f", profile=base_profile)
        merged = merge_feed_entry_into_config(base, RssFeedEntry(url=url, profile=pin_profile))
        alone = config_mod.Config(rss_url=url, profile=pin_profile)

        layers, _ = resolve_profile_layers(pin_profile)
        spoken = [f for f in layers if f in type(alone).model_fields]
        nested = layers.get("transcription")
        if isinstance(nested, dict):
            if "primary" in nested:
                spoken.append("transcription_provider")
            if "fallback" in nested:
                spoken.append("transcription_fallback_provider")
        assert spoken, f"{pin_profile} supplies no Config fields; the case is vacuous"

        differences = {
            f: (getattr(merged, f, None), getattr(alone, f, None))
            for f in set(spoken)
            if getattr(merged, f, None) != getattr(alone, f, None)
        }
        assert not differences, (
            f"pinning {pin_profile!r} onto a {base_profile!r} corpus diverges from top-level "
            f"resolution on fields the pin itself declares: {differences}"
        )

    def test_corpus_settings_the_pin_is_silent_about_are_preserved(self, base):
        """The overlay half of the contract — a pin must not reset the deployment."""
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        assert base.audio_storage_backend == "remote"  # from the corpus profile
        sub = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="cloud_with_dgx_primary")
        )
        assert (
            sub.audio_storage_backend == "remote"
        ), "pinning a feed silently switched the corpus's audio archiving to local"
        assert sub.cost_soft_cap_usd_per_run == 25.0

    def test_operator_explicit_values_survive_the_pin(self, base):
        """The precedence inversion, pinned directly: explicit fields beat profile defaults."""
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        sub = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="cloud_with_dgx_primary")
        )
        assert sub.litellm_api_base == self.REFERENCE_EXPLICIT["litellm_api_base"], (
            "the profile YAML's default overwrote the operator's explicit gateway — a pinned "
            "prod feed would send every LLM call to the wrong endpoint"
        )
        assert sub.cost_soft_cap_usd_per_run == 25.0

    def test_the_pin_still_moves_routing(self, base):
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        sub = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="cloud_with_dgx_primary")
        )
        assert sub.transcription_provider == "tailnet_dgx_whisper"
        assert sub.profile == "cloud_with_dgx_primary"

    def test_audio_preprocessing_preset_is_applied_not_just_named(self, base):
        """Reported-but-not-applied: the preset NAME was set while its settings were not."""
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        sub = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="cloud_with_dgx_primary")
        )
        ref = config_mod.Config(
            rss_url="https://p.example/f",
            profile="cloud_with_dgx_primary",
            **self.REFERENCE_EXPLICIT,
        )
        assert sub.audio_preprocessing_profile == ref.audio_preprocessing_profile
        assert sub.preprocessing_silence_threshold == ref.preprocessing_silence_threshold
        assert sub.preprocessing_silence_duration == ref.preprocessing_silence_duration


class TestStageCouplingOnAPin:
    """A model name must belong to the provider serving it (#1874 review-3 F3).

    Demonstrated bug: pinning ``bakeoff_gemini_flash`` onto a ``cloud_balanced`` corpus
    produced ``summary_provider=gemini`` with ``summary_model='podcast-flash-0731'`` — a
    LiteLLM alias meaningless to Gemini — because the pin declares the provider but not the
    model, and the base profile's model looked operator-explicit. Deployment-scoped settings
    are deliberately NOT coupled: a pin overlays routing, it does not reset the deployment.
    """

    @pytest.fixture()
    def base(self, monkeypatch):
        for key in (
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "DEEPGRAM_API_KEY",
            "LITELLM_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.setenv(key, "dummy-for-validation")
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")
        from podcast_scraper import config as config_mod

        return config_mod.Config(
            rss_url="https://placeholder.example/f",
            profile="cloud_balanced",
            cost_soft_cap_usd_per_run=25.0,
        )

    def test_moving_the_summary_provider_drops_the_old_providers_model(self, base):
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        url = "https://p.example/f"
        merged = merge_feed_entry_into_config(
            base, RssFeedEntry(url=url, profile="bakeoff_gemini_flash")
        )
        alone = config_mod.Config(rss_url=url, profile="bakeoff_gemini_flash")

        assert merged.summary_provider == "gemini"
        assert merged.summary_model == alone.summary_model, (
            f"gemini inherited {merged.summary_model!r} from the corpus profile — a model "
            "name from a provider that is no longer serving this stage"
        )

    def test_deployment_settings_are_not_coupled_and_survive(self, base):
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        merged = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="bakeoff_gemini_flash")
        )
        assert merged.cost_soft_cap_usd_per_run == 25.0, (
            "the pin reset a deployment-scoped setting; coupling must apply to models bound "
            "to a moved provider, not to corpus policy"
        )

    def test_a_stage_that_does_not_move_keeps_its_model(self, base):
        """Only a CHANGED provider drops its model — an unchanged stage carries over."""
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        merged = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="cloud_with_dgx_primary")
        )
        assert merged.summary_provider == "litellm"
        assert merged.summary_model, "an unchanged summary stage lost its model"


class TestRoutingIsOwnedByThePin:
    """Routing comes wholly from the pin; deployment policy overlays from the corpus.

    Review 4 (#1874) swept every ordered profile pair across all 99 routing-shaped fields and
    found 1200 violations: wherever the PIN was silent about a stage, the BASE profile's
    routing survived. An airgapped pin kept the corpus's cloud KG provider; a cloud pin kept
    the corpus's DGX ASR ladder. A profile that does not mention a stage is not endorsing the
    previous profile's choice — it runs that stage on its own defaults.
    """

    @pytest.fixture()
    def base(self, monkeypatch):
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "DEEPGRAM_API_KEY",
            "DEEPSEEK_API_KEY",
            "LITELLM_API_KEY",
        ):
            monkeypatch.setenv(key, "dummy-for-validation")
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx-llm-1")
        from podcast_scraper import config as config_mod

        return config_mod.Config(
            rss_url="https://placeholder.example/f",
            profile="cloud_quality",
            cost_soft_cap_usd_per_run=25.0,
        )

    def test_a_silent_stage_falls_to_the_pins_own_default_not_the_corpus(self, base):
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        url = "https://p.example/f"
        merged = merge_feed_entry_into_config(base, RssFeedEntry(url=url, profile="airgapped"))
        alone = config_mod.Config(rss_url=url, profile="airgapped")

        for field in ("kg_extraction_provider", "diarization_provider", "gi_value_gate_provider"):
            assert getattr(merged, field) == getattr(alone, field), (
                f"an airgapped pin inherited {field}={getattr(merged, field)!r} from the "
                "corpus profile — a stage the pin never endorsed"
            )

    def test_deployment_policy_still_overlays(self, base):
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        merged = merge_feed_entry_into_config(
            base, RssFeedEntry(url="https://p.example/f", profile="airgapped")
        )
        assert (
            merged.cost_soft_cap_usd_per_run == 25.0
        ), "the pin reset a corpus deployment decision; only ROUTING is pin-owned"

    def test_operator_explicit_routing_still_wins(self, base):
        """The A1 guarantee survives: an explicitly-set endpoint is not profile-derived."""
        from podcast_scraper import config as config_mod
        from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

        # Constructed, not model_copy'd: the operator's value has to be genuinely explicit
        # (in model_fields_set) for the classifier to see it, which is what prod does when the
        # operator YAML carries it.
        explicit = config_mod.Config(
            rss_url="https://placeholder.example/f",
            profile="cloud_quality",
            litellm_api_base="http://100.64.0.7:4001/v1",
        )
        merged = merge_feed_entry_into_config(
            explicit, RssFeedEntry(url="https://p.example/f", profile="cloud_with_dgx_primary")
        )
        assert merged.litellm_api_base == "http://100.64.0.7:4001/v1"
