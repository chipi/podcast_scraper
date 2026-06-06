"""Real end-to-end diarization: Whisper transcribe -> pyannote diarize -> screenplay.

The default local-Whisper path runs pyannote as a second pass and renders a
speaker-labelled screenplay. Every other diarization test mocks pyannote; this one
runs the *whole* default path on real audio so a break anywhere in
transcribe -> diarize -> align -> name-map -> screenplay is caught.

Requires the [ml] extra, cached Whisper + pyannote models, and an HF token with the
gated models accepted (``make preload-ml-models`` with a token). Skips — does not
fail — when those aren't provisioned, distinguishing "not provisioned" from a
regression.

Ground truth (tests/fixtures/FIXTURES_SPEC.md): ``p01_multi_e01`` = podcast p01
"Singletrack Sessions", host **Maya** + guest **Liam** = two distinct speakers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.ml_models, pytest.mark.diarization, pytest.mark.serial]

try:
    import pyannote.audio  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"pyannote.audio unavailable: {exc}", allow_module_level=True)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "p01_multi_e01.mp3"

_PROVISIONING_MARKERS = (
    "offlinemode",
    "offline mode",
    "gatedrepo",
    "localentrynotfound",
    "local cache",
    "cannot reach",
    "socketblocked",
    "connection",
    "no such file",
)


def _skip_if_unprovisioned(exc: Exception) -> None:
    haystack = f"{exc} {type(exc).__name__}".lower()
    if any(k in haystack for k in _PROVISIONING_MARKERS):
        pytest.skip(f"models not provisioned for offline e2e: {type(exc).__name__}")
    raise exc


def test_default_path_transcribe_diarize_screenplay() -> None:
    """Whisper + pyannote + screenplay on the two-voice fixture yield 2 labelled speakers."""
    assert _FIXTURE.is_file(), f"missing fixture: {_FIXTURE}"

    from podcast_scraper import config
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_from_segments,
    )
    from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result
    from podcast_scraper.providers.ml.ml_provider import MLProvider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        whisper_model="tiny.en",
        diarize=True,
        screenplay=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        try:
            provider = MLProvider(cfg)
            provider.initialize()
            result, _elapsed = provider.transcribe_with_segments(str(_FIXTURE))
            enriched = apply_diarization_to_result(
                result, str(_FIXTURE), cfg, ["Maya", "Liam"], cache_dir=tmp
            )
        except Exception as exc:  # noqa: BLE001 - provisioning vs regression
            _skip_if_unprovisioned(exc)
            return  # unreachable (skip raises), keeps type-checkers happy

    segments = enriched.get("segments") or []
    assert segments, "no transcript segments produced"

    # The default path must produce diarized labels (not a phantom single speaker).
    labels = {s.get("speaker_label") for s in segments if s.get("speaker_label")}
    assert len(labels) >= 2, f"expected >= 2 diarized speakers, got {sorted(labels)}"

    screenplay = format_diarized_screenplay_from_segments(segments)
    assert screenplay.strip(), "empty screenplay"
    # Each diarized speaker should surface as a "Label:" line in the screenplay.
    distinct_prefixes = {line.split(":", 1)[0] for line in screenplay.splitlines() if ":" in line}
    assert len(distinct_prefixes) >= 2, f"screenplay has < 2 speakers: {distinct_prefixes}"
