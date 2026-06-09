"""Guard: profiles that declare diarization must resolve with it ON.

``diarize``/``screenplay`` are coerced OFF when ``transcription_provider`` is not
in ``_DIARIZATION_ELIGIBLE_TRANSCRIPTION_PROVIDERS`` (config.py). That coercion is
correct, but it means a profile could silently drop diarization if its provider is
changed to a non-eligible one (or the coercion regresses) -- with only an INFO log.
This loads every diarize-declaring shipped profile through the real loader and
asserts the resolved Config still has diarization on (the profile↔runtime contract).
"""

from __future__ import annotations

import pytest

from podcast_scraper.config import Config, load_config_file

pytestmark = pytest.mark.unit

# Profiles whose YAML declares ``diarize: true`` with an eligible transcription
# provider (whisper / tailnet_dgx_whisper).
_DIARIZE_PROFILES = [
    "airgapped",
    "airgapped_thin",
    "dev",
    "local",
    "local_dgx_balanced",
    "local_dgx_full",
    "preprod_local_whisper",
    "cloud_with_dgx_primary",
]


@pytest.mark.parametrize("profile", _DIARIZE_PROFILES)
def test_diarize_declaring_profile_resolves_diarization_on(profile, monkeypatch):
    # Some diarize profiles keep an openai/deepgram *fallback* transcription provider,
    # whose construction validates an API key. The key is irrelevant to the diarize
    # resolution under test, so supply dummies so every profile constructs.
    for var in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "test-key")
    cfg = Config(**load_config_file(f"config/profiles/{profile}.yaml"))
    assert (
        cfg.diarize is True
    ), f"{profile} declares diarize: true but resolved to False (coerced off?)"
    assert cfg.screenplay is True, f"{profile} declares screenplay but resolved to False"
