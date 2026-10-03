"""The DGX profiles hold rather than fall over, and the local provider refuses non-English (#2178).

THE HAZARD. The local ``whisper`` tier's model default is ``base.en``, and for a non-English
language ``normalize_whisper_model_name`` strips the ``.en`` and runs ``["base", "tiny"]``. As a
FALLBACK it therefore converts a DGX outage into a silently bad transcript — a plausible
transcript of the wrong words that summary, GI, KG and search all then trust.

THE FIX IS TWO LAYERS, deliberately:

1. ``resilience_failure_strategy: hold`` (ADR-122) on the DGX profiles. The chain is never
   traversed, so the local tier is never reached on an outage.
2. The provider refuses a non-English request outright, so the hazard cannot return through one of
   the eight local / dev / airgapped profiles where it is the PRIMARY transcriber.

Layer 1 alone would leave the dev profiles exposed; layer 2 alone would leave prod degrading to a
bad transcript. Each is tested separately below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
PROFILES = REPO / "config" / "profiles"

#: The profiles that make the DGX the primary transcriber.
DGX_PROFILES = ("prod_dgx_full", "dev_dgx_full", "eval_default")


def _profile(name: str) -> dict:
    return yaml.safe_load((PROFILES / f"{name}.yaml").read_text(encoding="utf-8")) or {}


class TestTheDgxProfilesHold:
    @pytest.mark.parametrize("name", DGX_PROFILES)
    def test_the_strategy_is_hold_not_failover(self, name: str) -> None:
        """`hold` backoff-retries the chosen model, trips after N, pauses and probes, then raises
        ResilienceFuseOpenError and halts the batch. It never switches backends."""
        assert _profile(name).get("resilience_failure_strategy") == "hold", (
            f"{name} would fall over to the local English-only whisper tier on a DGX outage, "
            "producing a confident wrong-language transcript instead of stopping"
        )

    @pytest.mark.parametrize("name", DGX_PROFILES)
    def test_the_dgx_is_still_the_primary(self, name: str) -> None:
        """Guards against a future edit that achieves 'no failover' by removing the DGX instead."""
        assert _profile(name).get("transcription_provider") == "tailnet_dgx_whisper"

    @pytest.mark.parametrize("name", DGX_PROFILES)
    def test_the_chain_still_satisfies_adr_096(self, name: str) -> None:
        """The chain stays POPULATED even though `hold` never traverses it.

        ADR-096's validator rejects a DGX-only chain — it requires a non-DGX escape hatch, and
        that invariant is about deployment topology, not about which strategy a run picks. Emptying
        the chain to express "no failover" would fail config load; `hold` expresses it correctly.
        """
        chain = _profile(name).get("transcription_fallback_providers") or []
        assert chain, f"{name} has an empty chain, which ADR-096's validator rejects"
        assert any(
            p not in {"tailnet_dgx_whisper", "moss"} for p in chain
        ), f"{name}'s chain is DGX-only, which ADR-096 forbids"

    def test_a_dgx_profile_actually_loads(self) -> None:
        """The config validator is what would reject an over-eager version of this change."""
        from podcast_scraper import config as config_mod

        cfg = config_mod.Config(
            rss="https://example.com/f.xml",
            profile="prod_dgx_full",
            dgx_tailnet_host="dgx-llm-1",
        )
        from podcast_scraper.providers.resilience import FailureStrategy
        from podcast_scraper.providers.resilience.policy import resolve_failure_strategy

        assert resolve_failure_strategy(cfg) is FailureStrategy.HOLD


class TestTheLocalProviderRefusesNonEnglish:
    def test_a_non_english_request_raises(self) -> None:
        from podcast_scraper.providers.ml.ml_provider import _guard_supported_languages

        with pytest.raises(ValueError) as exc:
            _guard_supported_languages("es")
        # The guard names the SUPPORTED SET rather than hardcoding a language in its prose
        # (D-44): widening `LOCAL_WHISPER_SUPPORTED_LANGUAGES` changes the message with no
        # test edit, which is the point of routing on a parameter.
        assert "supports ['en']" in str(exc.value)
        assert "'es'" in str(exc.value)
        assert "base.en" in str(exc.value), "the message must name the mechanism, not just refuse"

    @pytest.mark.parametrize("code", ["es", "de", "ja", "pt-BR", "es-ES"])
    def test_every_non_english_form_is_refused(self, code: str) -> None:
        from podcast_scraper.providers.ml.ml_provider import _guard_supported_languages

        with pytest.raises(ValueError):
            _guard_supported_languages(code)

    @pytest.mark.parametrize("code", ["en", "en-US", "EN", "en_GB"])
    def test_english_in_any_regional_form_proceeds(self, code: str) -> None:
        from podcast_scraper.providers.ml.ml_provider import _guard_supported_languages

        _guard_supported_languages(code)  # must not raise

    def test_an_unset_language_proceeds(self) -> None:
        """``None`` is "nobody resolved a language" — the honest pre-#2172 state of most of the
        corpus. Refusing it would stop the local profiles transcribing anything at all."""
        from podcast_scraper.providers.ml.ml_provider import _guard_supported_languages

        _guard_supported_languages(None)
        _guard_supported_languages("")

    def test_the_guard_is_wired_at_both_transcribe_entries(self) -> None:
        """Two entry points (``transcribe`` and ``transcribe_with_segments``); a guard on one is a
        guard on neither, since the pipeline uses the segments variant."""
        src = (REPO / "src/podcast_scraper/providers/ml/ml_provider.py").read_text(encoding="utf-8")
        assert src.count("_guard_supported_languages(effective_language)") == 2


class TestTheProviderIsStillPrimarySomewhere:
    def test_it_remains_the_primary_transcriber_in_local_profiles(self) -> None:
        """The provider STAYS — S0.7 removes the hazard, not the tier. If this ever reaches zero,
        the guard above is guarding nothing and the local/airgapped profiles have silently lost
        their transcriber."""
        primaries = [
            p.stem
            for p in sorted(PROFILES.glob("*.yaml"))
            if (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("transcription_provider")
            == "whisper"
        ]
        assert len(primaries) >= 5, f"expected the local tier to be primary somewhere: {primaries}"
