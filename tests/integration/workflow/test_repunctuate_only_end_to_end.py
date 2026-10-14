"""``repunctuate_only`` driven through the REAL CLI and pipeline, on a two-episode corpus (#2187).

The unit tests prove each step of the stage. This proves the steps are WIRED together the way prod
runs them -- ``reprocess-prod.yml`` passes ``--pipeline-stage repunctuate_only
--reprocess-episode-ids <file> --skip-existing --no-transcript-cache`` -- because the failures this
stage could have (a provider wrapper that hides the clip call, a ``skip_existing`` that drops the
episode, a second metadata record in the fresh run dir, stale ad-free text feeding GI) all live in
the composition, not in any one function.

The corpus has two episodes: one with a 10-minute window that lost its punctuation, and one that is
clean. The audio is a local archive keyed by guid; the DGX is a fake with the clip call; the clip
cutter is replaced (no ffmpeg); the GI/KG builders are the real ones behind a recorder (their LLM
and grounding models are stubbed away by giving them the insight text directly). The only HTTP
allowed is the RSS fetch: a request for audio from the publisher fails the test.

Synthetic text only: the words are a fixed list of common nouns, never an episode's.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from podcast_scraper import cli
from podcast_scraper.corpus_delta import episode_derivation_fingerprint
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_from_segments,
)
from podcast_scraper.rss import downloader
from podcast_scraper.transcription import punctuation_repair
from podcast_scraper.utils import audio_cache, filesystem
from podcast_scraper.workflow import processing_manifest as pm, run_index

pytestmark = [pytest.mark.integration]

RSS_URL = "https://example.com/repunct/feed.xml"
FLAGGED_GUID = "guid-flagged-episode"
CLEAN_GUID = "guid-clean-episode"
WINDOW_S = 600.0
SEG_S = 10.0
SEGS_PER_WINDOW = 60

VOCAB = (
    "market budget garden window river system policy engine season record signal harvest bridge "
    "pattern circuit meadow lantern compass voyage quarry timber copper ledger orchard summit "
    "harbor kernel fabric glacier mosaic beacon cinder pillar tundra velvet anchor basalt "
    "cobalt dune ember fjord grove hollow island jungle lagoon marble nectar oasis prairie "
    "quartz reef savanna thicket umber valley willow yarrow zenith"
).split()
ALICE = {
    "speaker": "SPEAKER_00",
    "speaker_label": "Alice Host",
    "speaker_role": "host",
    "voice_type": "person",
}
BOB = {
    "speaker": "SPEAKER_01",
    "speaker_label": "Bob Guest",
    "speaker_role": "guest",
    "voice_type": "person",
}


def _punctuate(words: List[str]) -> str:
    out = []
    for i in range(0, len(words), 10):
        sentence = words[i : i + 10]
        sentence[0] = sentence[0].capitalize()
        sentence[-1] += "."
        out.append(" ".join(sentence))
    return " ".join(out)


def _segments(salt: int, broken_window: int | None) -> List[Dict[str, Any]]:
    segs = []
    for g in range(3 * SEGS_PER_WINDOW):
        words = [VOCAB[(salt + g * 7 + i) % len(VOCAB)] for i in range(20)]
        broken = g // SEGS_PER_WINDOW == broken_window
        segs.append(
            {
                "start": g * SEG_S,
                "end": g * SEG_S + SEG_S,
                "text": " ".join(words) if broken else _punctuate(words),
                **(ALICE if (g // 6) % 2 == 0 else BOB),
            }
        )
    return segs


def _rss() -> str:
    items = "".join(
        f"<item><title>{title}</title><guid>{guid}</guid>"
        f'<enclosure url="https://cdn.example.com/{guid}.mp3" type="audio/mpeg"/></item>'
        for title, guid in (("Ep One", FLAGGED_GUID), ("Ep Two", CLEAN_GUID))
    )
    return (
        "<?xml version='1.0'?><rss version='2.0'><channel><title>A Show</title>"
        f"<language>en</language>{items}</channel></rss>"
    )


class FakeDgx:
    """The DGX Whisper provider's clip call, answering from the flagged episode's stored words."""

    def __init__(self, old_segments: List[Dict[str, Any]]) -> None:
        self.old = old_segments
        self.clip_calls: List[tuple] = []
        self.punctuated = True

    def initialize(self) -> None:
        pass

    def transcribe_clip(self, audio_path: str, language=None, prompt=None):
        window = json.loads(Path(audio_path).read_text(encoding="utf-8"))
        a, b = window["start"], window["end"]
        self.clip_calls.append((a, b, language))
        inside = [s for s in self.old if a <= s["start"] < b]
        new = []
        for k in range(0, len(inside), 2):
            pair = inside[k : k + 2]
            words = " ".join(s["text"] for s in pair).split()
            new.append(
                {
                    "start": pair[0]["start"] - a,
                    "end": pair[-1]["end"] - a,
                    "text": _punctuate(words) if self.punctuated else " ".join(words),
                }
            )
        return {"text": " ".join(s["text"] for s in new), "segments": new}


class Corpus:
    """Two episodes already on disk, in a prior run of a feed, the way a finished corpus is."""

    def __init__(self, root: Path) -> None:
        self.root = root
        feed_dir = Path(filesystem.corpus_feed_output_dir(str(root), RSS_URL))
        self.run = feed_dir / "run_20260101-000000_prior"
        (self.run / "transcripts").mkdir(parents=True)
        (self.run / "metadata").mkdir(parents=True)
        self.feed_dir = feed_dir
        self.flagged_segments = _segments(1, broken_window=1)
        self.clean_segments = _segments(5, broken_window=None)
        self.stems = {FLAGGED_GUID: "0001 - Ep One", CLEAN_GUID: "0002 - Ep Two"}
        self._write(FLAGGED_GUID, self.flagged_segments, flagged=[[WINDOW_S, 2 * WINDOW_S]])
        self._write(CLEAN_GUID, self.clean_segments, flagged=[])
        self.archive = root / "audio-archive"
        for guid in (FLAGGED_GUID, CLEAN_GUID):
            path = audio_cache.cache_path_for_guid(self.archive, guid, ".mp3")
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ID3 synthetic audio for " + guid.encode())

    def f(self, guid: str, suffix: str, where: str = "transcripts") -> Path:
        return self.run / where / f"{self.stems[guid]}{suffix}"

    def _write(self, guid: str, segs: List[Dict[str, Any]], flagged: List[List[float]]) -> None:
        stem = self.stems[guid]
        self.f(guid, ".txt").write_text(format_diarized_screenplay_from_segments(segs), "utf-8")
        self.f(guid, ".segments.json").write_text(json.dumps(segs), "utf-8")
        asr: Dict[str, Any] = {
            "model": "Systran/faster-whisper-large-v3-turbo",
            "speech_coverage": 0.93,
            "language": {"requested": "en", "reported": "en", "mismatch": False},
            "punctuation_repair": [{"start": 600.0, "end": 1200.0, "status": "refused"}],
        }
        if flagged:
            asr["unpunctuated_windows"] = flagged
        self.f(guid, ".asr.json").write_text(json.dumps(asr, indent=2), "utf-8")
        rel = f"transcripts/{stem}.txt"
        pm.update_stage(
            str(self.run),
            rel,
            "asr",
            pm.stage_block(
                ran=True,
                model="Systran/faster-whisper-large-v3-turbo",
                metrics={"unpunctuated_windows": len(flagged), "punctuation_windows_repaired": 0},
            ),
            quality_flags=["asr_partly_unpunctuated"] if flagged else [],
        )
        # What a previous full run left that is derived from the OLD text. The ad-free base is the
        # one that bites: GI reads it in preference to the transcript.
        self.f(guid, ".adfree.txt").write_text("STALE AD-FREE TEXT FROM BEFORE THE REPAIR", "utf-8")
        self.f(guid, ".adfree.segments.json").write_text("[]", "utf-8")
        self.f(guid, ".cleaned.txt").write_text("STALE CLEANED TEXT", "utf-8")
        self.f(guid, ".metadata.json", "metadata").write_text(
            json.dumps(
                {
                    "feed": {"title": "A Show", "language": "en"},
                    "episode": {
                        "guid": guid,
                        "episode_id": f"id-{guid}",
                        "title": stem[7:],
                        "duration_seconds": 1800,
                        "published_date": "Mon, 01 Jan 2026 00:00:00 +0000",
                    },
                    "content": {
                        "transcript_file_path": rel,
                        "transcript_source": "whisper_transcription",
                    },
                }
            ),
            "utf-8",
        )

    def bundle(self, guid: str) -> EpisodeArtifactBundle:
        meta = self.f(guid, ".metadata.json", "metadata")
        return EpisodeArtifactBundle(
            metadata_path=meta,
            gi_path=self.f(guid, ".gi.json", "metadata"),
            kg_path=self.f(guid, ".kg.json", "metadata"),
            bridge_path=None,
            episode_id=guid,
            stem=self.stems[guid],
        )

    def snapshot(self, guid: str) -> Dict[str, Any]:
        prefix = self.stems[guid]
        return {
            str(p.relative_to(self.run)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(self.run.rglob("*"))
            if p.is_file() and p.name.startswith(prefix)
        }


@pytest.fixture
def world(tmp_path, monkeypatch):
    run_index.reset_corpus_metadata_index_cache_for_tests()
    corpus = Corpus(tmp_path / "corpus")
    dgx = FakeDgx(corpus.flagged_segments)
    factory_calls: List[dict] = []

    def _factory(_cfg, **kwargs):
        factory_calls.append(kwargs)
        return dgx

    from podcast_scraper.workflow import orchestration

    monkeypatch.setattr(orchestration, "_create_transcription_provider_factory", _factory)
    monkeypatch.setattr(
        punctuation_repair,
        "cut_clip",
        lambda _audio, start, end, out: Path(out).write_text(
            json.dumps({"start": start, "end": end}), encoding="utf-8"
        ),
    )

    # The real GI and KG builders behind a recorder. GI is handed one insight whose text carries a
    # hash of the transcript it was given, so its artifact changes exactly when its input does.
    handed: Dict[str, List[str]] = {"gi": [], "kg": []}
    import podcast_scraper.gi as gi_mod
    import podcast_scraper.kg as kg_mod

    real_gi, real_kg = gi_mod.build_artifact, kg_mod.build_artifact

    def _gi(episode_id, transcript_text, **kw):
        handed["gi"].append(transcript_text)
        digest = hashlib.sha256(transcript_text.encode()).hexdigest()[:12]
        kw.update(insight_texts=[f"An insight grounded in text {digest}"], insight_provider=None)
        return real_gi(episode_id, transcript_text, **kw)

    def _kg(episode_id, transcript_text, **kw):
        handed["kg"].append(transcript_text)
        return real_kg(episode_id, transcript_text, **kw)

    monkeypatch.setattr(gi_mod, "build_artifact", _gi)
    monkeypatch.setattr(kg_mod, "build_artifact", _kg)

    http: List[str] = []

    def _http(url, user_agent=None, timeout=None, stream=False):
        normalized = downloader.normalize_url(url)
        http.append(normalized)
        if normalized != downloader.normalize_url(RSS_URL):
            raise AssertionError(f"Unexpected HTTP request (the audio is in the archive): {url}")
        from tests.conftest import create_rss_response

        return create_rss_response(_rss(), RSS_URL)

    ids_file = tmp_path / "worklist.txt"
    ids_file.write_text(f"{FLAGGED_GUID}\n{CLEAN_GUID}\n", encoding="utf-8")
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        "\n".join(
            [
                "transcription_provider: tailnet_dgx_whisper",
                "transcription_fallback_provider: openai",
                "openai_api_key: sk-test-not-a-key",
                "dgx_tailnet_host: dgx-llm-1.tail-test.ts.net",
                f"audio_cache_dir: {corpus.archive}",
                "preprocessing_enabled: false",
                "auto_speakers: false",
                "generate_metadata: true",
                "generate_summaries: false",
                "generate_gi: true",
                "gi_require_grounding: false",
                "generate_kg: true",
                "kg_extraction_source: metadata_only",
                "diarize: true",
                "screenplay: true",
                "save_adfree_transcript: true",
                "single_feed_uses_corpus_layout: true",
            ]
        ),
        encoding="utf-8",
    )

    def run(*extra: str) -> int:
        run_index.reset_corpus_metadata_index_cache_for_tests()
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=_http),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=_http),
        ):
            return cli.main(
                [
                    RSS_URL,
                    "--config",
                    str(settings),
                    "--output-dir",
                    str(corpus.root),
                    "--pipeline-stage",
                    "repunctuate_only",
                    "--reprocess-episode-ids",
                    str(ids_file),
                    "--skip-existing",
                    "--no-transcript-cache",
                    *extra,
                ]
            )

    return SimpleWorld(corpus, dgx, factory_calls, handed, http, run)


class SimpleWorld:
    def __init__(self, corpus, dgx, factory_calls, handed, http, run) -> None:
        self.corpus = corpus
        self.dgx = dgx
        self.factory_calls = factory_calls
        self.handed = handed
        self.http = http
        self.run = run


def _fresh_run_dirs(corpus: Corpus) -> List[Path]:
    return [p for p in corpus.feed_dir.glob("run_*") if p != corpus.run]


def test_the_flagged_episode_is_repaired_and_the_clean_one_is_not_touched(world):
    corpus = world.corpus
    clean_before = corpus.snapshot(CLEAN_GUID)
    old_flagged = json.loads(corpus.f(FLAGGED_GUID, ".segments.json").read_text())

    exit_code = world.run()

    assert exit_code == 0

    # THE CLEAN EPISODE: byte- and mtime-identical, never fetched, never decoded.
    assert corpus.snapshot(CLEAN_GUID) == clean_before
    assert not any(CLEAN_GUID in str(u) for u in world.http)

    # THE FLAGGED EPISODE: one window re-decoded, spliced, labelled, re-rendered.
    assert world.dgx.clip_calls == [(WINDOW_S, 2 * WINDOW_S, "en")]
    new = json.loads(corpus.f(FLAGGED_GUID, ".segments.json").read_text())
    inside = lambda s: WINDOW_S <= s["start"] < 2 * WINDOW_S  # noqa: E731
    assert [s for s in new if not inside(s)] == [s for s in old_flagged if not inside(s)]
    repaired = [s for s in new if inside(s)]
    assert repaired and all(s["punctuation_repaired"] for s in repaired)
    assert {s["speaker_label"] for s in repaired} == {"Alice Host", "Bob Guest"}
    text = corpus.f(FLAGGED_GUID, ".txt").read_text()
    assert text == format_diarized_screenplay_from_segments(new)

    # The records are patched, not rebuilt.
    asr = json.loads(corpus.f(FLAGGED_GUID, ".asr.json").read_text())
    assert "unpunctuated_windows" not in asr and asr["speech_coverage"] == 0.93
    assert [a.get("pass") for a in asr["punctuation_repair"]] == [None, "repunctuate_only"]
    manifest = json.loads(corpus.f(FLAGGED_GUID, ".manifest.json").read_text())
    assert "asr_partly_unpunctuated" not in manifest["quality_flags"]
    assert "asr_punctuation_repaired" in manifest["quality_flags"]
    assert manifest["stages"]["asr"]["model"] == "Systran/faster-whisper-large-v3-turbo"


def test_the_cascade_reads_the_new_text_and_not_a_stale_derivative(world):
    corpus = world.corpus
    before = {
        g: episode_derivation_fingerprint(corpus.bundle(g)) for g in (FLAGGED_GUID, CLEAN_GUID)
    }
    assert world.run() == 0

    new_text = corpus.f(FLAGGED_GUID, ".segments.json").read_text()
    assert new_text and not corpus.f(FLAGGED_GUID, ".cleaned.txt").exists()
    # GI and KG were handed the repaired transcript, once, and nothing carries the stale text.
    assert len(world.handed["gi"]) == 1 and len(world.handed["kg"]) == 1
    for seen in (world.handed["gi"][0], world.handed["kg"][0]):
        assert "STALE" not in seen
        assert "Alice Host:" in seen and ". " in seen, "the punctuated screenplay"

    after = {
        g: episode_derivation_fingerprint(corpus.bundle(g)) for g in (FLAGGED_GUID, CLEAN_GUID)
    }
    assert after[FLAGGED_GUID] != before[FLAGGED_GUID], "the corpus delta sees the episode change"
    assert after[CLEAN_GUID] == before[CLEAN_GUID]


def test_it_writes_one_record_where_the_episode_lives(world):
    """No second metadata record in the fresh run dir (#2075): the metadata and its GI/KG sit
    beside the transcript they describe."""
    corpus = world.corpus
    assert world.run() == 0

    for fresh in _fresh_run_dirs(corpus):
        assert not list(fresh.glob("metadata/*.metadata.json")), fresh
        assert not list(fresh.glob("transcripts/*")), fresh
    assert corpus.f(FLAGGED_GUID, ".gi.json", "metadata").exists()
    assert corpus.f(FLAGGED_GUID, ".kg.json", "metadata").exists()
    assert not corpus.f(CLEAN_GUID, ".gi.json", "metadata").exists(), "the clean one was not run"


def test_the_provider_was_built_without_its_wrappers_and_the_publisher_was_never_asked(world):
    assert world.run() == 0

    assert world.factory_calls == [{"_wrap_fallback": False}]
    assert world.http and all(u == downloader.normalize_url(RSS_URL) for u in world.http)


def test_a_second_run_finds_nothing_left_to_repair(world):
    """Repeatable: the windows are measured from the transcript, so a repaired one is not flagged
    again and the second pass touches nothing."""
    corpus = world.corpus
    assert world.run() == 0
    settled = {g: corpus.snapshot(g) for g in (FLAGGED_GUID, CLEAN_GUID)}
    calls = len(world.dgx.clip_calls)

    assert world.run() == 0

    assert len(world.dgx.clip_calls) == calls, "no window left to decode"
    assert {g: corpus.snapshot(g) for g in (FLAGGED_GUID, CLEAN_GUID)} == settled


def test_the_run_records_what_it_did_for_every_episode(world):
    """Ledger: the repaired one is ok, the clean one is a counted skip, nothing is missing."""
    assert world.run() == 0

    rows = {}
    for fresh in _fresh_run_dirs(world.corpus):
        for index in fresh.glob("index.json"):
            rows.update({e["episode_id"]: e for e in json.loads(index.read_text())["episodes"]})
    assert rows, "the run wrote no index"
    by_status = {e["status"] for e in rows.values()}
    assert by_status <= {"ok", "skipped"} and "failed" not in by_status
    assert any(e.get("error_type") == "NothingToRepunctuate" for e in rows.values())
    assert len(rows) == 2


def test_a_partly_clean_worklist_whose_flagged_episode_is_refused_still_succeeds(world):
    """The flagged window is refused, the clean episode needs nothing: one of the two requests is
    met, so this is a partial repair (a success whose unfinished id is logged at ERROR), not the
    total miss that fails the run. The refusal is in the ledger and nothing was written."""
    corpus = world.corpus
    before = {g: corpus.snapshot(g) for g in (FLAGGED_GUID, CLEAN_GUID)}
    world.dgx.punctuated = False

    exit_code = world.run()

    assert len(world.dgx.clip_calls) == 1, "the flagged window was decoded and refused"
    assert {g: corpus.snapshot(g) for g in (FLAGGED_GUID, CLEAN_GUID)} == before
    rows = {}
    for fresh in _fresh_run_dirs(corpus):
        for index in fresh.glob("index.json"):
            rows.update({e["episode_id"]: e for e in json.loads(index.read_text())["episodes"]})
    assert {e.get("error_type") for e in rows.values()} == {
        "NoWindowAccepted",
        "NothingToRepunctuate",
    }
    assert exit_code == 0


def test_a_worklist_of_only_refused_episodes_exits_nonzero(world, tmp_path):
    corpus = world.corpus
    world.dgx.punctuated = False
    only_flagged = tmp_path / "only-flagged.txt"
    only_flagged.write_text(f"{FLAGGED_GUID}\n", encoding="utf-8")
    before = corpus.snapshot(FLAGGED_GUID)

    # argparse keeps the LAST --reprocess-episode-ids, so this narrows the fixture's work-list.
    exit_code = world.run("--reprocess-episode-ids", str(only_flagged))

    assert exit_code == 1, "a repair that repaired none of what it was asked to is a failed run"
    assert corpus.snapshot(FLAGGED_GUID) == before


def test_a_dry_run_decodes_and_writes_nothing(world):
    corpus = world.corpus
    before = {g: corpus.snapshot(g) for g in (FLAGGED_GUID, CLEAN_GUID)}

    exit_code = world.run("--dry-run")

    assert exit_code == 0
    assert world.dgx.clip_calls == []
    assert {g: corpus.snapshot(g) for g in (FLAGGED_GUID, CLEAN_GUID)} == before
