"""The artifacts record what language went to ASR and what the service said back (#2187).

WHY THIS EXISTS. V.6b's first real ASR run on non-English audio (Spanish, 2026-10-07) could not
answer two of its own questions — did the per-episode language reach the ASR service, and does
the service's reported language agree — because nothing recorded either: `asr.json` and the
manifest's `asr` stage carried only the model. The DGX provider merged the two into one
`language` field (`requested or detected`), which makes a disagreement invisible by construction.
These tests pin the split: the provider returns both halves, `asr.json` carries a `language`
block, and the manifest flags `asr_language_mismatch` when the primary subtags differ.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.tailnet_dgx import whisper_provider as wp
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)
from podcast_scraper.workflow import episode_processor, processing_manifest as pm

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_breaker():
    wp._whisper_breaker.reset()
    yield
    wp._whisper_breaker.reset()


def _cfg(**kw):
    base = dict(
        rss_url="https://example.com/feed.xml",
        run_id="run-xyz",
        transcription_provider="tailnet_dgx_whisper",
        dgx_whisper_model="turbo",
        diarization_provider="pyannote",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestTheProviderReturnsBothHalves:
    def _provider(self) -> TailnetDgxWhisperTranscriptionProvider:
        cfg = Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "transcription_fallback_provider": "openai",
                "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
                "openai_api_key": "sk-test",
            }
        )
        provider = TailnetDgxWhisperTranscriptionProvider(cfg)
        provider._initialized = True
        return provider

    @pytest.mark.parametrize(("asked", "server_says"), [("es", "es"), ("es", "pt"), (None, "de")])
    def test_requested_and_reported_are_kept_apart(self, tmp_path, asked, server_says) -> None:
        audio = tmp_path / "ep.mp3"
        audio.write_bytes(b"\x00\x01")
        provider = self._provider()

        def fake_call(*_a, **_k):
            provider._last_detected_language = server_says  # what the server's payload said
            return "texto", [{"start": 0.0, "end": 1.0, "text": "texto"}], 0.5

        with (
            patch(
                f"{wp.__name__}.check_faster_whisper_health",
                return_value=True,
            ),
            patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx", fake_call),
        ):
            result, _ = provider.transcribe_with_segments(str(audio), language=asked)

        assert result["language_requested"] == asked
        assert result["language_reported"] == server_says
        assert result["language"] == (asked or server_says)  # the merged field is unchanged


class TestTheRecord:
    def test_a_match(self) -> None:
        rec = episode_processor._asr_language_record(
            {"language_requested": "es", "language_reported": "es"}, _cfg()
        )
        assert rec == {"requested": "es", "reported": "es", "mismatch": False}

    def test_a_mismatch_compares_primary_subtags(self) -> None:
        rec = episode_processor._asr_language_record(
            {"language_requested": "pt-BR", "language_reported": "es"}, _cfg()
        )
        assert rec["mismatch"] is True
        same = episode_processor._asr_language_record(
            {"language_requested": "pt-BR", "language_reported": "pt"}, _cfg()
        )
        assert same["mismatch"] is False

    def test_reported_is_never_filled_from_requested(self) -> None:
        rec = episode_processor._asr_language_record({"language_requested": "es"}, _cfg())
        assert rec == {"requested": "es", "reported": None, "mismatch": None}

    def test_requested_falls_back_to_the_episodes_resolved_language(self) -> None:
        cfg = _cfg(feed_declared_language="de-DE")
        rec = episode_processor._asr_language_record({}, cfg)
        assert rec["requested"] == "de"
        assert rec["reported"] is None


class TestTheArtifacts:
    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def test_asr_json_carries_the_language_block(self) -> None:
        d, rel = self._setup()
        result = {
            "speech_audio_ratio": 0.93,
            "language_requested": "es",
            "language_reported": "pt",
        }
        episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
        rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
        assert rec["language"] == {"requested": "es", "reported": "pt", "mismatch": True}

    def test_the_manifest_flags_a_mismatch(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "language_requested": "es", "language_reported": "pt"}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        asr = data["stages"]["asr"]
        assert asr["metrics"]["language_requested"] == "es"
        assert asr["metrics"]["language_reported"] == "pt"
        assert "asr_language_mismatch" in data["quality_flags"]

    def test_the_manifest_does_not_flag_a_match(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "language_requested": "es", "language_reported": "es"}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_language_mismatch" not in data.get("quality_flags", [])


class TestUntranscribedSpeechIsRecorded:
    """The diarization pipeline's `asr_untranscribed_speech` reaches asr.json and the manifest."""

    GAP = [{"start": 166.739, "end": 185.791, "duration_s": 19.052, "speaker": "SPEAKER_00"}]

    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def test_asr_json_lists_the_gaps(self) -> None:
        d, rel = self._setup()
        result = {"speech_audio_ratio": 0.93, "asr_untranscribed_speech": self.GAP}
        episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
        rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
        assert rec["untranscribed_speech"] == self.GAP

    def test_the_manifest_flags_and_counts_them(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "asr_untranscribed_speech": self.GAP}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_untranscribed_speech" in data["quality_flags"]
        assert data["stages"]["asr"]["metrics"]["untranscribed_speech_count"] == 1
        assert data["stages"]["asr"]["metrics"]["untranscribed_speech_s"] == 19.052

    def test_no_gaps_is_recorded_as_zero_not_flagged(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "asr_untranscribed_speech": []}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_untranscribed_speech" not in data.get("quality_flags", [])
        assert data["stages"]["asr"]["metrics"]["untranscribed_speech_count"] == 0


class TestSpeechRecoveryIsRecorded:
    """#2187 A2: what was re-transcribed reaches asr.json and the manifest, flagged and counted."""

    RECOVERY = [
        {
            "start": 29.7,
            "end": 34.1,
            "speaker": "SPEAKER_00",
            "status": "recovered",
            "words": 12,
            "rejected": [],
        },
        {
            "start": 34.6,
            "end": 48.9,
            "speaker": "SPEAKER_00",
            "status": "rejected",
            "words": 0,
            "rejected": ["low_confidence"],
        },
    ]

    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def test_asr_json_lists_each_attempt(self) -> None:
        d, rel = self._setup()
        result = {"speech_audio_ratio": 0.93, "asr_speech_recovery": self.RECOVERY}
        episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
        rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
        assert rec["speech_recovery"] == self.RECOVERY

    def test_the_manifest_flags_and_counts_recovered_speech(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "asr_speech_recovery": self.RECOVERY}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_speech_recovered" in data["quality_flags"]
        assert data["stages"]["asr"]["metrics"]["recovered_speech_count"] == 1
        assert data["stages"]["asr"]["metrics"]["recovered_words"] == 12

    def test_attempts_that_recovered_nothing_are_not_flagged(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "asr_speech_recovery": self.RECOVERY[1:]}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_speech_recovered" not in data.get("quality_flags", [])
        assert data["stages"]["asr"]["metrics"]["recovered_speech_count"] == 0

    def test_the_clip_transcriber_uses_the_providers_clip_call_and_the_episodes_language(
        self,
    ) -> None:
        calls = []

        class _Provider:
            def transcribe_clip(self, path, language=None):
                calls.append(("clip", path, language))
                return {"segments": []}

            def transcribe_with_segments(self, path, language=None, **kw):
                calls.append(("episode", path, language))
                return {"segments": [], "text": ""}, 0.1

        cfg = Config(rss="https://example.com/f.xml").model_copy(
            update={"feed_declared_language": "es"}
        )
        transcriber = episode_processor._gap_clip_transcriber(cfg, _Provider())
        assert transcriber is not None
        assert transcriber("/tmp/gap_000.wav") == {"segments": []}
        assert calls == [("clip", "/tmp/gap_000.wav", "es")]

    def test_a_provider_without_a_clip_call_gets_no_recovery(self) -> None:
        """Never the episode path: its guardrail, retries and breaker treat a near-empty clip as
        an outage (V.6b French, 2026-10-08: breaker OPEN, then up to 900 s holding the lock)."""

        class _Provider:
            def transcribe_with_segments(self, path, language=None, **kw):
                raise AssertionError("a gap clip must never go through the episode path")

        cfg = Config(rss="https://example.com/f.xml")
        assert episode_processor._gap_clip_transcriber(cfg, _Provider()) is None

    def test_a_provider_without_a_clip_call_is_said_once_per_feature(self, caplog) -> None:
        # Review A3: the coverage-gate and failover wrappers do not forward transcribe_clip, so a
        # profile change turned both features off with no trace.
        class _Wrapped:
            pass

        episode_processor._CLIPLESS_PROVIDERS_REPORTED.discard(("_Wrapped", "gap recovery"))
        cfg = Config(rss="https://example.com/f.xml")
        with caplog.at_level("INFO", logger=episode_processor.logger.name):
            for _ in range(3):
                episode_processor._gap_clip_transcriber(cfg, _Wrapped())
        said = [r for r in caplog.records if "gap recovery is off" in r.getMessage()]
        assert len(said) == 1 and "_Wrapped has no transcribe_clip" in said[0].getMessage()


class TestTheDgxClipCall:
    """The DGX provider's ``transcribe_clip`` makes one request and nothing else."""

    def _provider(self) -> TailnetDgxWhisperTranscriptionProvider:
        cfg = Config(
            rss="https://example.com/f.xml",
            transcription_provider="tailnet_dgx_whisper",
            dgx_tailnet_host="dgx-llm-1",
            transcription_fallback_providers=["tailnet_dgx_whisper", "whisper"],
        )
        return TailnetDgxWhisperTranscriptionProvider(cfg)

    def test_a_near_empty_clip_is_an_answer_not_an_outage(self, tmp_path) -> None:
        clip = tmp_path / "gap.wav"
        clip.write_bytes(b"\0" * 64)
        provider = self._provider()
        response = {
            "text": "Merci.",
            "segments": [{"start": 0.0, "end": 0.6, "text": "Merci."}],
            "language": "fr",
        }

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return response

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a) -> None:
                return None

            def post(self, url, data=None, files=None):
                sent.append(data)
                return _Resp()

        sent: list = []
        with (
            patch.object(wp, "hardened_http_client", lambda *a, **k: _Client()),
            patch.object(wp.resilience, "probe_audio_duration_sec", lambda p: 5.5),
        ):
            out = provider.transcribe_clip(str(clip), language="fr")
        # 1 word for 5.5 s is under the episode floor (6): through the episode path this raised.
        assert [s["text"] for s in out["segments"]] == ["Merci."]
        assert len(sent) == 1 and sent[0]["language"] == "fr" and "prompt" not in sent[0]

    def test_a_clip_does_not_overwrite_the_episodes_detected_language(self, tmp_path) -> None:
        # `_last_detected_language` is read by the EPISODE call after it releases the lock; a clip
        # (another episode's, at transcription_parallelism > 1) must not be able to stamp it.
        clip = tmp_path / "gap.wav"
        clip.write_bytes(b"\0" * 64)
        provider = self._provider()
        provider._last_detected_language = "es"

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"text": "Hello.", "segments": [], "language": "en"}

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a) -> None:
                return None

            def post(self, url, data=None, files=None):
                return _Resp()

        with (
            patch.object(wp, "hardened_http_client", lambda *a, **k: _Client()),
            patch.object(wp.resilience, "probe_audio_duration_sec", lambda p: 2.0),
        ):
            provider.transcribe_clip(str(clip), language="es")
        assert provider._last_detected_language == "es"

    def test_a_transport_error_propagates_once(self, tmp_path) -> None:
        clip = tmp_path / "gap.wav"
        clip.write_bytes(b"\0" * 64)
        provider = self._provider()
        posts: list = []

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a) -> None:
                return None

            def post(self, url, data=None, files=None):
                posts.append(1)
                raise ConnectionError("dgx down")

        with patch.object(wp, "hardened_http_client", lambda *a, **k: _Client()):
            with pytest.raises(ConnectionError):
                provider.transcribe_clip(str(clip), language="fr")
        assert posts == [1], "one request: no retry policy around a clip"


class TestTheAsrResultIsInspected:
    """#2187: every fresh ASR result passes ``_inspect_asr_result`` on its way out of
    ``_transcribe_with_segments_maybe_chunked``: invented lines removed, punctuation breaks noted.
    """

    PUNCTUATED = "So the ports moved north over a decade. Why? The delta silted up. " * 12
    UNPUNCTUATED = "so the ports moved north over a decade and the trading families followed " * 5

    def _asr(self) -> dict:
        texts = [" Sous-titrage Société Radio-Canada"] + [
            self.PUNCTUATED if minute < 30 else self.UNPUNCTUATED for minute in range(1, 40)
        ]
        segs = [{"start": 0.0, "end": 30.0, "text": texts[0]}] + [
            {"start": m * 60.0, "end": m * 60.0 + 59.0, "text": texts[m]} for m in range(1, 40)
        ]
        return {"text": " ".join(texts), "segments": segs, "language": "en"}

    def test_through_the_real_transcription_path(self, tmp_path) -> None:
        asr = self._asr()

        class _Provider:
            def transcribe_with_segments(self, path, language=None, **kw):
                return dict(asr), 2.0

        media = tmp_path / "a.mp3"
        media.write_bytes(b"\0" * 1024)
        from podcast_scraper.models.entities import TranscriptionJob

        cfg = Config(rss="https://example.com/f.xml")
        job = TranscriptionJob(idx=1, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result, elapsed = episode_processor._transcribe_with_segments_maybe_chunked(
            str(media),
            cfg=cfg,
            job=job,
            transcription_provider=_Provider(),
            pipeline_metrics=None,
            episode_duration_seconds=2400,
            call_metrics=None,
        )
        assert elapsed == 2.0
        assert result["asr_invented_lines"] == [
            {"start": 0.0, "end": 30.0, "text": "Sous-titrage Société Radio-Canada"}
        ]
        assert "Sous-titrage" not in result["text"]
        assert len(result["segments"]) == 39
        assert result["asr_unpunctuated_windows"] == [[1800.0, 2400.0]]

    def test_a_clean_result_gains_nothing(self) -> None:
        clean = {
            "text": self.PUNCTUATED,
            "segments": [{"start": 0, "end": 9, "text": self.PUNCTUATED}],
        }
        out, _ = episode_processor._inspect_asr_result(clean, 1.0)
        assert out is clean


class TestInventedLinesAndBrokenPunctuationAreRecorded:
    INVENTED = [{"start": 0.0, "end": 30.0, "text": "Sous-titrage Société Radio-Canada"}]
    WINDOWS = [[1800.0, 2400.0], [2400.0, 3000.0]]

    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def test_asr_json_lists_both(self) -> None:
        d, rel = self._setup()
        result = {
            "speech_audio_ratio": 0.9,
            "asr_invented_lines": self.INVENTED,
            "asr_unpunctuated_windows": self.WINDOWS,
        }
        episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
        rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
        assert rec["invented_lines_removed"] == self.INVENTED
        assert rec["unpunctuated_windows"] == self.WINDOWS

    def test_the_manifest_flags_and_counts_both(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {
            "speech_audio_ratio": 0.9,
            "asr_invented_lines": self.INVENTED,
            "asr_unpunctuated_windows": self.WINDOWS,
        }
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert {"asr_invented_lines_removed", "asr_partly_unpunctuated"} <= set(
            data["quality_flags"]
        )
        assert data["stages"]["asr"]["metrics"]["invented_lines_removed"] == 1
        assert data["stages"]["asr"]["metrics"]["unpunctuated_windows"] == 2

    def test_neither_is_flagged_when_absent(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        episode_processor._write_processing_manifest(
            {"speech_audio_ratio": 0.9}, _cfg(), job, rel, d
        )
        data = json.load(open(pm.manifest_path(d, rel)))
        flags = set(data.get("quality_flags", []))
        assert not {"asr_invented_lines_removed", "asr_partly_unpunctuated"} & flags
        assert data["stages"]["asr"]["metrics"]["invented_lines_removed"] == 0


def test_stretched_words_reach_asr_json_and_raise_no_flag() -> None:
    """#2187: evidence for inspection — recorded, never counted as untranscribed speech."""
    from podcast_scraper.models.entities import TranscriptionJob

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "transcripts"))
    open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
    rel = "transcripts/0006 - X.txt"
    stretched = [
        {"start": 209.5, "end": 213.98, "duration_s": 4.48, "word": "case,", "speech_s": 3.77}
    ]
    result = {"speech_audio_ratio": 0.9, "asr_stretched_words": stretched}
    episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
    rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
    assert rec["stretched_words"] == stretched
    job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
    episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
    data = json.load(open(pm.manifest_path(d, rel)))
    assert "asr_untranscribed_speech" not in data.get("quality_flags", [])


class TestBrokenPunctuationIsRepaired:
    """#2187: a flagged window goes back through the provider's clip call with the episode
    language's prompt, and is replaced when that is a real repair."""

    FLAT = "so the ports moved north over a decade and the trading families followed them " * 2

    def _asr(self) -> dict:
        punct = "So the ports moved north. Why? The delta silted up, and trade moved on. " * 30
        segs: list = [{"start": 0.0, "end": 590.0, "text": punct}]
        segs += [
            {"start": 600.0 + 10 * i, "end": 609.0 + 10 * i, "text": self.FLAT} for i in range(20)
        ]
        return {"text": " ".join([punct] + [self.FLAT] * 20), "segments": segs, "language": "es"}

    class _Provider:
        def __init__(self) -> None:
            self.calls: list = []

        def transcribe_clip(self, path, language=None, prompt=None):
            """The window's own words, punctuated — what a real repair returns."""
            self.calls.append((language, prompt))
            flat = TestBrokenPunctuationIsRepaired.FLAT.split()
            line = " ".join(flat[:7]).capitalize() + ". " + " ".join(flat[7:]) + "."
            segs = [{"start": 10.0 * i, "end": 10.0 * i + 9.0, "text": line} for i in range(20)]
            return {"text": " ".join(s["text"] for s in segs), "segments": segs}

    def _inspect(self, monkeypatch, **cfg_update):
        from podcast_scraper.transcription import punctuation_repair

        monkeypatch.setattr(punctuation_repair, "cut_clip", lambda *a: None)
        provider = self._Provider()
        cfg = Config(rss="https://example.com/f.xml").model_copy(
            update={"feed_declared_language": "es", **cfg_update}
        )
        out, _ = episode_processor._inspect_asr_result(
            self._asr(), 1.0, media_path="/tmp/a.mp3", cfg=cfg, provider=provider
        )
        return out, provider

    def test_the_window_is_repaired_with_the_spanish_prompt(self, monkeypatch) -> None:
        out, provider = self._inspect(monkeypatch)
        from podcast_scraper.transcription.punctuation import WINDOW_PROMPTS

        assert provider.calls == [("es", WINDOW_PROMPTS["es"])]
        assert [r["status"] for r in out["asr_punctuation_repair"]] == ["repaired"]
        assert "asr_unpunctuated_windows" not in out

    def test_switched_off_it_is_only_recorded(self, monkeypatch) -> None:
        out, provider = self._inspect(monkeypatch, transcription_repair_unpunctuated_windows=False)
        assert provider.calls == []
        assert out["asr_unpunctuated_windows"] == [[600.0, 1200.0]]
        assert "asr_punctuation_repair" not in out

    def test_the_manifest_flags_and_counts_repairs(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        rel = "transcripts/0006 - X.txt"
        repair = [
            {"start": 600.0, "end": 799.0, "status": "repaired"},
            {"start": 1200.0, "end": 1790.0, "status": "refused", "reason": "echoed_prompt"},
        ]
        result = {"speech_audio_ratio": 0.9, "asr_punctuation_repair": repair}
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_punctuation_repaired" in data["quality_flags"]
        assert data["stages"]["asr"]["metrics"]["punctuation_windows_repaired"] == 1
        episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
        rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
        assert rec["punctuation_repair"] == repair

    def test_an_invented_line_in_a_repaired_window_is_removed(self, monkeypatch) -> None:
        """A repaired window is a fresh decode: a credit line inside it goes, recorded with any
        removed before the repair."""
        from podcast_scraper.transcription import punctuation_repair

        monkeypatch.setattr(punctuation_repair, "cut_clip", lambda *a: None)
        base = self._Provider()

        class _WithCredit:
            def transcribe_clip(self, path, language=None, prompt=None):
                out = base.transcribe_clip(path, language=language, prompt=prompt)
                credit = {
                    "start": 199.0,
                    "end": 199.5,
                    "text": " Sous-titrage Société Radio-Canada",
                }
                return {"text": out["text"], "segments": out["segments"] + [credit]}

        asr = self._asr()
        asr["segments"].insert(0, {"start": 0.0, "end": 0.0, "text": "Gracias por ver el video."})
        cfg = Config(rss="https://example.com/f.xml").model_copy(
            update={"feed_declared_language": "es"}
        )
        out, _ = episode_processor._inspect_asr_result(
            asr, 1.0, media_path="/tmp/a.mp3", cfg=cfg, provider=_WithCredit()
        )
        assert [r["status"] for r in out["asr_punctuation_repair"]] == ["repaired"]
        assert [x["text"] for x in out["asr_invented_lines"]] == [
            "Gracias por ver el video.",
            "Sous-titrage Société Radio-Canada",
        ]
        assert "Sous-titrage" not in out["text"]
