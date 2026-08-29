# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — alias="rss"; populate-by-name accepts either at runtime.
"""W1 (#1874): every packaged profile must construct with only the keys it declares.

WHY THIS EXISTS. ``cloud_with_dgx_primary`` carried a leftover ``transcription.fallback:
openai`` block after its fallback was moved to Deepgram everywhere else. Nothing caught it:
the registry drift test compares the YAML to the registry preset (both said Deepgram), and
no test ever CONSTRUCTED the profile. It surfaced only when a hand-run smoke happened to
resolve the profile without an OpenAI key present — i.e. by luck, one commit later.

The check that matters is NARROW, not blunt. A first version of this test set every dummy
provider key and asserted the profile validates — and would have sailed past the very bug it
was written for, because with an OpenAI key present a stale OpenAI reference validates fine.
So: resolve the profile once to learn which providers it actually ROUTES to, then rebuild it
with ONLY those providers' keys present. A profile that still demands a key for a provider it
no longer routes to now fails, which is exactly the leftover-block class. Runs in well under
a second across every profile — the cheapest test in the suite over the largest surface.

Deliberately NOT asserted here: that a profile's routing is *correct*. That is the registry
drift test's job. This asks only "does it stand up", which is the question nobody was asking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

# Every provider secret a packaged profile might reference. Set as a block so a profile that
# legitimately needs a new provider fails on its ROUTING, not on a key this list forgot.
_DUMMY_SECRETS = {
    "OPENAI_API_KEY": "dummy-for-validation",
    "ANTHROPIC_API_KEY": "dummy-for-validation",
    "GEMINI_API_KEY": "dummy-for-validation",
    "DEEPGRAM_API_KEY": "dummy-for-validation",
    "DEEPSEEK_API_KEY": "dummy-for-validation",
    "GROQ_API_KEY": "dummy-for-validation",
    "GROK_API_KEY": "dummy-for-validation",
    "MISTRAL_API_KEY": "dummy-for-validation",
    "LITELLM_API_KEY": "dummy-for-validation",
    "QWEN_API_KEY": "dummy-for-validation",
    "DASHSCOPE_API_KEY": "dummy-for-validation",
}


def _profile_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "profiles"
        if candidate.is_dir():
            return candidate
    raise AssertionError("config/profiles not found from the test file")


def _profile_names() -> list[str]:
    names = [
        p.stem for p in sorted(_profile_dir().glob("*.yaml")) if not p.stem.endswith(".example")
    ]
    assert names, "no packaged profiles discovered — the test would be vacuous"
    return names


# Provider -> the env var whose absence Config rejects. Providers with no key requirement
# (vllm, ollama, tailnet_dgx*, local whisper, transformers, moss) are absent by design.
_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok": "GROK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "litellm": "LITELLM_API_KEY",
    "qwen": "QWEN_API_KEY",
}

# Every field that names a provider this profile routes to.
_ROUTING_FIELDS = (
    "transcription_provider",
    "transcription_fallback_provider",
    "diarization_provider",
    "summary_provider",
    "speaker_detector_provider",
    "quote_extraction_provider",
    "entailment_provider",
    "kg_extraction_provider",
    "gi_value_gate_provider",
)


def _routed_providers(cfg) -> set[str]:
    routed: set[str] = set()
    for field in _ROUTING_FIELDS:
        value = getattr(cfg, field, None)
        if isinstance(value, str) and value.strip():
            routed.add(value.strip())
    for value in getattr(cfg, "transcription_fallback_providers", None) or []:
        if isinstance(value, str) and value.strip():
            routed.add(value.strip())
    for value in getattr(cfg, "summary_fallback_providers", None) or []:
        if isinstance(value, str) and value.strip():
            routed.add(value.strip())
    return routed


@pytest.mark.parametrize("profile_name", _profile_names())
def test_profile_needs_only_the_keys_of_providers_it_routes_to(
    profile_name: str, monkeypatch
) -> None:
    for key, value in _DUMMY_SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")

    from podcast_scraper import config as config_mod

    # Pass 1 — everything present, purely to read back what the profile routes to.
    try:
        resolved = config_mod.Config(rss_url="https://example.test/feed.xml", profile=profile_name)
    except Exception as exc:  # noqa: BLE001 — the failure IS the finding
        raise AssertionError(f"profile {profile_name!r} does not construct at all: {exc}") from exc

    assert resolved.profile == profile_name, (
        f"{profile_name!r} resolved to {resolved.profile!r} — a profile that silently renames "
        "itself makes every downstream 'which profile ran this' answer wrong"
    )

    # Pass 2 — ONLY the keys its own routing justifies.
    routed = _routed_providers(resolved)
    needed = {_PROVIDER_ENV[p] for p in routed if p in _PROVIDER_ENV}
    for key in _DUMMY_SECRETS:
        if key not in needed:
            monkeypatch.delenv(key, raising=False)

    try:
        config_mod.Config(rss_url="https://example.test/feed.xml", profile=profile_name)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"profile {profile_name!r} demands a provider key its routing does not justify.\n"
            f"  routes to: {sorted(routed)}\n"
            f"  keys provided: {sorted(needed)}\n"
            f"  error: {exc}\n"
            "This is the leftover-reference class: cloud_with_dgx_primary kept a stale "
            "'transcription.fallback: openai' block after its fallback moved to Deepgram, and "
            "no test constructed the profile to notice."
        ) from exc


@pytest.mark.parametrize("profile_name", _profile_names())
def test_profile_names_a_transcription_route(profile_name: str, monkeypatch) -> None:
    """Every profile must say how it transcribes — the stage that dominates cost.

    A profile silently falling through to the Config default is the shape of the 2026-08-28
    routing bugs: it LOOKS configured and is not.
    """
    for key, value in _DUMMY_SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")

    from podcast_scraper import config as config_mod

    cfg = config_mod.Config(rss_url="https://example.test/feed.xml", profile=profile_name)
    assert (
        cfg.transcription_provider or ""
    ).strip(), f"profile {profile_name!r} resolves no transcription_provider"


@pytest.mark.parametrize("profile_name", _profile_names())
def test_fallback_provider_agrees_with_the_fallback_ladder(profile_name: str, monkeypatch) -> None:
    """The two representations of ASR fallback must name the same provider.

    THE ACTUAL SHAPE of the 2026-08-28 bug, and what the first two versions of this file both
    missed. ``cloud_with_dgx_primary`` carries the routing twice: a nested ``transcription:
    {primary, fallback}`` block that flattens into ``transcription_fallback_provider``, and a
    ``transcription_fallback_providers`` ladder. Moving the fallback to Deepgram updated the
    ladder and the registry but left the nested block on ``openai``, so the profile resolved
    with fallback_provider='openai' while its own ladder ended in 'deepgram' — and demanded an
    OpenAI key it had no reason to need.

    Neither "does it construct" nor "does it need only justified keys" can see that: once the
    nested block flattens, the stale value IS the routing, so it justifies its own key. Only
    comparing the two representations against each other catches it.
    """
    for key, value in _DUMMY_SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")

    from podcast_scraper import config as config_mod

    cfg = config_mod.Config(rss_url="https://example.test/feed.xml", profile=profile_name)
    single = (getattr(cfg, "transcription_fallback_provider", None) or "").strip()
    ladder = [
        str(p).strip()
        for p in (getattr(cfg, "transcription_fallback_providers", None) or [])
        if str(p).strip()
    ]
    if not single or not ladder:
        pytest.skip(f"{profile_name} declares only one fallback representation")

    assert single in ladder, (
        f"profile {profile_name!r} disagrees with itself about ASR fallback:\n"
        f"  transcription_fallback_provider = {single!r}\n"
        f"  transcription_fallback_providers = {ladder}\n"
        "Two representations of one routing drifted apart — the profile will demand the "
        "credentials of a provider its own ladder never reaches."
    )
