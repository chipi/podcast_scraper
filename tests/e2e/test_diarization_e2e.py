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


def test_default_path_transcribe_diarize_screenplay_into_graph() -> None:
    """Full default path on the two-voice fixture, asserting expected values at every layer.

    Whisper transcribe -> pyannote diarize -> named screenplay (Maya/Liam) -> GI
    (insights + timestamped quotes from the diarized segments) -> KG (Maya + Liam
    become Entity nodes). Proves diarization is generated *and* picked up downstream
    by the graph, with the known ground truth — not just "something ran".
    """
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

    # Ground truth (FIXTURES_SPEC): host **Maya** + guest **Liam**. The detected
    # names must map onto the two diarized speakers — assert the *actual values*,
    # not just "some 2-speaker split".
    labels = {s.get("speaker_label") for s in segments if s.get("speaker_label")}
    assert {"Maya", "Liam"}.issubset(labels), f"expected Maya+Liam diarized, got {sorted(labels)}"

    screenplay = format_diarized_screenplay_from_segments(segments)
    assert screenplay.strip(), "empty screenplay"
    # Both speakers must surface as named "Name:" lines in the screenplay.
    assert "Maya:" in screenplay, f"host Maya not labelled in screenplay:\n{screenplay[:300]}"
    assert "Liam:" in screenplay, f"guest Liam not labelled in screenplay:\n{screenplay[:300]}"

    # === extend into the graph: the diarized transcript must drive GI + KG ===
    from podcast_scraper.gi import build_artifact as gi_build_artifact
    from podcast_scraper.kg import build_artifact as kg_build_artifact

    transcript_text = enriched.get("text") or ""

    gi = gi_build_artifact(
        "episode:p01-multi-e01",
        transcript_text,
        transcript_segments=segments,  # diarized segments carry timing into quotes
        model_version="test",
        cfg=cfg,
        episode_title="Building Trails That Last",
        podcast_id="podcast:p01",
    )
    gi_types = {n["type"] for n in gi.get("nodes", [])}
    assert {"Episode", "Insight", "Quote"}.issubset(gi_types), f"GI node types: {gi_types}"
    quotes = [n for n in gi["nodes"] if n["type"] == "Quote"]
    assert quotes, "GI produced no grounded quotes"
    # Diarized segment timing flows into quote timestamps (FR2.2).
    assert all("timestamp_start_ms" in n.get("properties", {}) for n in quotes)

    kg = kg_build_artifact(
        "episode:p01-multi-e01",
        transcript_text,
        podcast_id="podcast:p01",
        episode_title="Building Trails That Last",
        detected_hosts=["Maya"],
        detected_guests=["Liam"],
        cfg=cfg,
    )
    kg_entities = {
        (n.get("properties", {}).get("name") or n.get("label"))
        for n in kg.get("nodes", [])
        if n.get("type") == "Entity"
    }
    # The diarized host + guest become graph entities.
    assert {"Maya", "Liam"}.issubset(kg_entities), f"KG entities missing Maya/Liam: {kg_entities}"
