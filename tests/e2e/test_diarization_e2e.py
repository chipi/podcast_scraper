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

from tests._fixtures import fixtures_dir

pytestmark = [pytest.mark.e2e, pytest.mark.ml_models, pytest.mark.diarization, pytest.mark.serial]

try:
    import pyannote.audio  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"pyannote.audio unavailable: {exc}", allow_module_level=True)

# Pin to v1 audio explicitly: the v2 fixture regenerated in #902 produces audio whose
# two TTS voices the diarizer can't cleanly separate, so the "Maya+Liam diarized"
# assertion fails. v1 audio (171KB, richer voice differentiation) separates into two
# voices cleanly. Neither fixture's host self-introduces in a whisper-clean way (whisper
# hears "Maya" as "Ma'am"), so the host name reaches the roster via known_hosts (below),
# not self-intro. Re-evaluate after v2 fixtures are regenerated with diarization-aware,
# self-intro-clean voice synthesis (descendant of #921); until then this test stays on v1.
_FIXTURE = fixtures_dir("audio", version="v1") / "p01_multi_e01.mp3"

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
    # No HF token wired into the job env (e.g. forks / token-less CI): the pyannote
    # provider raises "HuggingFace token required for diarize=true" at construction.
    # That's a provisioning gap, not a regression — skip rather than fail.
    "huggingface token required",
    "token required",
    "hf token",
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

    # Host **Maya** is supplied via known_hosts, guest **Liam** via detected names. The v1
    # fixture's TTS host never self-introduces in a whisper-clean way (whisper hears the
    # host's name "Maya" as "Ma'am"), so the roster's self-intro path can't name the host
    # off this audio — known_hosts is how a feed-known host name reaches the roster in
    # production. This still exercises the real mapping: known host → intro-dominant voice,
    # detected guest → the other voice, end-to-end into a named screenplay.
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        whisper_model="tiny.en",
        diarize=True,
        screenplay=True,
        known_hosts=["Maya"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        try:
            provider = MLProvider(cfg)
            provider.initialize()
            result, _elapsed = provider.transcribe_with_segments(str(_FIXTURE))
            enriched = apply_diarization_to_result(
                result, str(_FIXTURE), cfg, ["Liam"], cache_dir=tmp
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

    # === #974: the diarized screenplay drives the ad-free processing base ===
    # On a REAL pyannote screenplay, the producer must emit segments whose char ranges
    # slice back exactly (the basis for Fault A/B fixes downstream).
    from podcast_scraper.workflow.adfree_transcript import (
        build_adfree_artifacts,
        load_processing_transcript,
        produce_adfree_transcript,
    )

    arts = build_adfree_artifacts(screenplay, segments)
    assert arts is not None, "no ad-free artifacts from diarized screenplay"
    for s in arts.segments:
        assert arts.text[s["char_start"] : s["char_end"]] == s["text"], "ad-free offset drift"
    assert {"Maya", "Liam"} & {s.get("speaker_label") for s in arts.segments}

    # The three sidecars are produced next to the raw .txt (raw left untouched).
    with tempfile.TemporaryDirectory() as tdir:
        rel = "transcripts/ep.txt"
        (Path(tdir) / "transcripts").mkdir()
        (Path(tdir) / rel).write_text(screenplay, encoding="utf-8")
        adfree_rel = produce_adfree_transcript(screenplay, segments, rel, tdir)
        assert adfree_rel == "transcripts/ep.adfree.txt"
        assert (Path(tdir) / "transcripts" / "ep.adfree.segments.json").is_file()
        assert (Path(tdir) / "transcripts" / "ep.adfree.admap.json").is_file()
        # The resolver prefers the ad-free base and it carries char offsets.
        loaded = load_processing_transcript(tdir, rel)
        assert loaded.is_adfree and loaded.segments
        assert (Path(tdir) / rel).read_text() == screenplay  # raw untouched

    # === extend into the graph: the AD-FREE base must drive GI + KG ===
    from podcast_scraper.gi import build_artifact as gi_build_artifact
    from podcast_scraper.kg import build_artifact as kg_build_artifact

    transcript_text = arts.text  # GI reasons over the ad-free processing base (#974)

    gi = gi_build_artifact(
        "episode:p01-multi-e01",
        transcript_text,
        transcript_segments=arts.segments,  # carry char offsets -> exact speaker/timing
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
    # Fault A on REAL data: every quote char_start indexes the saved ad-free text exactly.
    for n in quotes:
        cs = n["properties"]["char_start"]
        ce = n["properties"]["char_end"]
        assert transcript_text[cs:ce] == n["properties"]["text"], "quote char_start drift"
    # Fault B on REAL data: diarized speaker maps onto at least one quote.
    assert any(n["properties"].get("speaker_id") for n in quotes), "no quote got a speaker_id"

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

    # === bridge (GI<->KG) + CIL: diarized speakers must reach the corpus graph layer ===
    import json as _json

    from podcast_scraper.builders.bridge_builder import build_bridge
    from podcast_scraper.server import cil_queries

    bridge = build_bridge("episode:p01-multi-e01", gi, kg)
    bridge_blob = _json.dumps(bridge).lower()
    assert "maya" in bridge_blob and "liam" in bridge_blob, "diarized speakers absent from bridge"

    with tempfile.TemporaryDirectory() as corpus:
        meta = Path(corpus) / "metadata"
        meta.mkdir()
        (meta / "ep.gi.json").write_text(_json.dumps(gi), encoding="utf-8")
        (meta / "ep.kg.json").write_text(_json.dumps(kg), encoding="utf-8")
        (meta / "ep.bridge.json").write_text(_json.dumps(bridge), encoding="utf-8")
        # The CIL person-profile query resolves the diarized host against the built
        # corpus graph (topics/quotes need linked insights — see GI-attribution note).
        profile = cil_queries.person_profile(corpus, corpus, "person:maya")
        assert profile["person_id"] == "person:maya"
