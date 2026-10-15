"""``pipeline_stage=repunctuate_only``: re-decode only the windows flagged unpunctuated (#2187).

48 punctuation-repair windows on prod were refused by guards that have since been corrected; about
25 now pass. Running whole episodes again (Whisper, diarization, naming) to reach them would cost
an episode each, so this stage reads the stored transcript, measures its windows again, decodes
only those from the archived audio, and splices in what passes -- speakers carried over from the
segments it replaces.

The contracts that matter, each tested on a synthetic episode laid out the way a finished corpus is
(transcript + segments + asr.json + manifest + derived files):

* an episode with nothing to repair is left COMPLETELY alone -- no audio fetched, no file written;
* an accepted window changes that window and nothing outside it;
* the records that describe the transcript are PATCHED, never rebuilt from this pass's result;
* every refusal and skip writes a ledger row, because a stage that does less than it was asked and
  says nothing is the failure this whole arc keeps finding.

Synthetic text only: the words are a fixed list of common nouns, never an episode's.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, Dict, List, Optional
from unittest.mock import patch

import pytest

from podcast_scraper import config
from podcast_scraper.models import Episode, TranscriptionJob
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_from_segments,
)
from podcast_scraper.transcription import punctuation_repair
from podcast_scraper.utils import audio_cache
from podcast_scraper.workflow import (
    episode_processor as ep,
    metrics as metrics_mod,
    processing_manifest as pm,
)
from podcast_scraper.workflow.stages import processing

pytestmark = pytest.mark.unit

GUID = "guid-repunct-1"
WINDOW_S = 600.0
SEG_S = 10.0
SEGS_PER_WINDOW = 60
ORIGINAL_MODEL = "Systran/faster-whisper-large-v3-turbo"

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
    "stated_speaker": "Alice Host",
}
BOB = {
    "speaker": "SPEAKER_01",
    "speaker_label": "Bob Guest",
    "speaker_role": "guest",
    "voice_type": "person",
}


# --- the synthetic episode --------------------------------------------------------------------


def _words(index: int, n: int = 20) -> List[str]:
    return [VOCAB[(index * 7 + i) % len(VOCAB)] for i in range(n)]


def _punctuate(words: List[str]) -> str:
    """Capitalised sentences of ten words, the way a punctuated decode reads."""
    out: List[str] = []
    for i in range(0, len(words), 10):
        sentence = words[i : i + 10]
        sentence[0] = sentence[0].capitalize()
        sentence[-1] += "."
        out.append(" ".join(sentence))
    return " ".join(out)


def _segment(g: int, broken: bool) -> Dict[str, Any]:
    """Global segment ``g``: ten seconds, no gaps, six to a turn, two alternating voices."""
    words = _words(g)
    voice = ALICE if (g // 6) % 2 == 0 else BOB
    return {
        "start": g * SEG_S,
        "end": g * SEG_S + SEG_S,
        "text": (
            " ".join(words) if broken else _punctuate(words[:10]) + " " + _punctuate(words[10:])
        ),
        "avg_logprob": -0.21,
        **voice,
    }


def _segments(broken: tuple, n_windows: int = 3) -> List[Dict[str, Any]]:
    return [
        _segment(w * SEGS_PER_WINDOW + i, w in broken)
        for w in range(n_windows)
        for i in range(SEGS_PER_WINDOW)
    ]


@dataclass
class Disk:
    run: Path
    stem: str
    out: Path  # the NEW run dir a reprocess is handed: empty, as in production

    def f(self, suffix: str) -> Path:
        return self.run / "transcripts" / f"{self.stem}{suffix}"

    @property
    def txt(self) -> Path:
        return self.f(".txt")

    @property
    def seg(self) -> Path:
        return self.f(".segments.json")

    @property
    def asr(self) -> Path:
        return self.f(".asr.json")

    @property
    def manifest(self) -> Path:
        return self.f(".manifest.json")

    def snapshot(self) -> Dict[str, Any]:
        """Every file under the old run: bytes AND mtime, so a rewrite of identical bytes shows."""
        return {
            str(p.relative_to(self.run)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(self.run.rglob("*"))
            if p.is_file()
        }


def _write_disk(
    tmp_path: Path,
    *,
    broken: tuple = (1,),
    n_windows: int = 3,
    asr_language: Optional[str] = "en",
    tagged_language: Optional[str] = None,
    char_offsets: bool = False,
) -> Disk:
    run = tmp_path / "corpus" / "run_20260101-000000_prior"
    (run / "transcripts").mkdir(parents=True)
    (run / "metadata").mkdir(parents=True)
    out = tmp_path / "corpus" / "run_20260102-000000_fresh"
    out.mkdir(parents=True)
    d = Disk(run, "0001 - Ep", out)

    segs = _segments(broken, n_windows)
    if char_offsets:
        cursor = 0
        for s in segs:
            s["char_start"], s["char_end"] = cursor, cursor + len(s["text"])
            cursor += len(s["text"]) + 1
    text = format_diarized_screenplay_from_segments(segs)

    if tagged_language:
        # A translated episode: canonical = the English render, the SOURCE pair is tagged.
        d.f(f".{tagged_language}.txt").write_text(text, encoding="utf-8")
        d.f(f".{tagged_language}.segments.json").write_text(json.dumps(segs), encoding="utf-8")
        d.txt.write_text("ENGLISH RENDER OF THE SOURCE\n", encoding="utf-8")
        d.seg.write_text(
            json.dumps([{"start": 0.0, "end": 1.0, "text": "english"}]), encoding="utf-8"
        )
    else:
        d.txt.write_text(text, encoding="utf-8")
        d.seg.write_text(json.dumps(segs, indent=0), encoding="utf-8")

    flagged = [[w * WINDOW_S, (w + 1) * WINDOW_S] for w in broken]
    asr: Dict[str, Any] = {
        "model": ORIGINAL_MODEL,
        "speech_coverage": 0.93,
        "speech_audio_ratio": 0.9,
        "failed_over": False,
        "language": {"requested": asr_language, "reported": asr_language, "mismatch": False},
        "silent_gap_recovery": [{"start": 5.0, "end": 9.0, "status": "skipped"}],
        "digital_silence_trimmed": {"lead_s": 2.0, "tail_s": 0.0},
        "punctuation_repair": [
            {"start": 600.0, "end": 1200.0, "status": "refused", "reason": "lost_words"}
        ],
        "unpunctuated_windows": flagged,
    }
    d.asr.write_text(json.dumps(asr, indent=2), encoding="utf-8")

    rel = "transcripts/0001 - Ep.txt"
    root = str(run)
    pm.update_stage(
        root,
        rel,
        "asr",
        pm.stage_block(
            ran=True,
            method="tailnet_dgx_whisper",
            model=ORIGINAL_MODEL,
            method_version=pm.METHOD_VERSIONS["asr"],
            duration_s=412.5,
            cost_usd=0.0,
            metrics={
                "speech_coverage": 0.93,
                "language_requested": asr_language,
                "unpunctuated_windows": len(flagged),
                "punctuation_windows_repaired": 0,
            },
            failover={"from": "turbo", "to": "large-v3"},
        ),
        episode_id="ep-1",
        feed_id="https://example.com/feed.xml",
        run_id="run-original",
        quality_flags=["asr_partly_unpunctuated", "asr_failover"],
    )
    pm.update_stage(root, rel, "diarization", pm.stage_block(ran=True, metrics={"num_speakers": 2}))
    pm.update_stage(
        root, rel, "naming", pm.stage_block(ran=True, metrics={"named": 2, "unresolved": 0})
    )

    for suffix in (
        ".cleaned.txt",
        ".anon.txt",
        ".adfree.txt",
        ".adfree.segments.json",
        ".adfree.admap.json",
        ".adfree.turns.json",
        ".turns.json",
    ):
        d.f(suffix).write_text("STALE FROM THE OLD TEXT", encoding="utf-8")
    d.f(".speakers.diagnostics.json").write_text('{"voices": []}', encoding="utf-8")
    return d


def _episode(disk: Disk, *, media_url: Optional[str] = None) -> Episode:
    item = ET.Element("item")
    ET.SubElement(item, "guid").text = GUID
    episode = Episode(
        idx=1, title="Ep", title_safe="Ep", item=item, transcript_urls=[], media_url=media_url
    )
    episode.on_disk_transcript = str(disk.txt)
    episode.on_disk_idx = 1
    return episode


def _job(episode: Episode) -> TranscriptionJob:
    return TranscriptionJob(
        idx=1, ep_title="Ep", ep_title_safe="Ep", temp_media="", episode=episode
    )


def _cfg(**over: Any) -> config.Config:
    base: Dict[str, Any] = dict(
        rss="https://example.com/feed.xml",
        transcription_provider="tailnet_dgx_whisper",
        transcription_fallback_provider="openai",
        openai_api_key="sk-test-not-a-key",
        dgx_tailnet_host="dgx-llm-1.tail-test.ts.net",
        pipeline_stage="repunctuate_only",
        preprocessing_enabled=False,
        feed_declared_language="en",
        diarize=True,
        screenplay=True,
        save_adfree_transcript=True,
        dgx_whisper_model="Systran/faster-whisper-large-v3",
    )
    base.update(over)
    return config.Config(**base)


# --- the DGX stand-in, the audio archive, and the clip cutter ---------------------------------


def _clip_answer(old: List[Dict[str, Any]], *, punctuated: bool = True) -> Dict[str, Any]:
    """What the DGX would return for a window: the same words, two old segments per new one."""
    a = old[0]["start"]
    segments = []
    for k in range(0, len(old), 2):
        pair = old[k : k + 2]
        words = " ".join(s["text"] for s in pair).split()
        text = _punctuate(words) if punctuated else " ".join(words)
        segments.append({"start": pair[0]["start"] - a, "end": pair[-1]["end"] - a, "text": text})
    return {"text": " ".join(s["text"] for s in segments), "segments": segments}


def _window_of(disk: Disk, w: int) -> List[Dict[str, Any]]:
    return [
        s
        for s in json.loads(disk.seg.read_text(encoding="utf-8"))
        if w * WINDOW_S <= s["start"] < (w + 1) * WINDOW_S
    ]


class Clips:
    """``transcribe_clip`` answering the scripted windows in order; records every call."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: List[tuple] = []

    def transcribe_clip(self, audio_path: str, language=None, prompt=None):
        self.calls.append((audio_path, language, prompt))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Archive:
    """An audio archive holding the scripted keys; records what was asked of it."""

    def __init__(self, *held: str) -> None:
        self.held = set(held)
        self.asked: List[tuple] = []

    def exists(self, rel: str) -> bool:
        self.asked.append(("exists", rel))
        return rel in self.held

    def download(self, rel: str, dest: str) -> bool:
        self.asked.append(("download", rel))
        Path(dest).write_bytes(b"ID3 synthetic audio")
        return True


def _archive_key(ext: str = ".mp3") -> str:
    key = audio_cache.rel_key_for_guid(GUID, ext)
    assert key is not None
    return key


@pytest.fixture
def cuts(monkeypatch):
    """Replace the ffmpeg clip cutter; the list records every (start, end) it was asked for."""
    asked: List[tuple] = []

    def _cut(_audio: str, start: float, end: float, out_path: str) -> None:
        asked.append((start, end))
        Path(out_path).write_bytes(b"RIFF synthetic clip")

    monkeypatch.setattr(punctuation_repair, "cut_clip", _cut)
    return asked


@pytest.fixture
def archive(monkeypatch):
    held = Archive(_archive_key())
    monkeypatch.setattr(audio_cache, "resolve_backend", lambda *_a, **_k: held)
    return held


def _run(disk: Disk, provider: Any, cfg: Optional[config.Config] = None, **episode_kw: Any):
    """Run the stage for the episode and return ``(result, ledger rows)``."""
    pipeline_metrics = metrics_mod.Metrics()
    result = ep._repunctuate_existing_transcript(
        _job(_episode(disk, **episode_kw)),
        cfg or _cfg(),
        None,
        str(disk.out),
        provider,
        pipeline_metrics,
    )
    rows = [(s.status, s.error_type) for s in pipeline_metrics.episode_statuses]
    return result, rows, pipeline_metrics


# --- an episode with nothing to repair is left completely alone -------------------------------


def test_a_clean_episode_is_left_completely_untouched(tmp_path, cuts, monkeypatch):
    disk = _write_disk(tmp_path, broken=())
    before = disk.snapshot()
    held = Archive(_archive_key())
    monkeypatch.setattr(audio_cache, "resolve_backend", lambda *_a, **_k: held)
    clips = Clips()

    result, rows, _ = _run(disk, clips)

    assert result == (False, None, 0)
    assert disk.snapshot() == before, "bytes AND mtimes: not even a rewrite of identical bytes"
    assert held.asked == [] and cuts == [] and clips.calls == [], "nothing fetched, cut or decoded"
    assert rows == [("skipped", "NothingToRepunctuate")]
    assert list(disk.out.iterdir()) == [], "and nothing written into the fresh run either"


def test_a_plain_transcript_is_refused_and_left_untouched(tmp_path, cuts, monkeypatch):
    """No speaker labels: the served text is not the join of the segments (88 of 96 plain prod
    transcripts differ from it), so a re-render would change words outside the window."""
    disk = _write_disk(tmp_path)  # window 1 is flagged
    plain = [
        {k: v for k, v in s.items() if not k.startswith("speaker") and k != "stated_speaker"}
        for s in json.loads(disk.seg.read_text(encoding="utf-8"))
    ]
    disk.seg.write_text(json.dumps(plain, indent=0), encoding="utf-8")
    before = disk.snapshot()
    held = Archive(_archive_key())
    monkeypatch.setattr(audio_cache, "resolve_backend", lambda *_a, **_k: held)
    clips = Clips()

    result, rows, _ = _run(disk, clips)

    assert result == (False, None, 0)
    assert disk.snapshot() == before
    assert held.asked == [] and cuts == [] and clips.calls == []
    assert rows == [("skipped", "PlainTranscript")]


def test_a_wholly_unpunctuated_transcript_is_not_reported_clean(tmp_path, cuts, archive):
    """``unpunctuated_windows`` answers [] for 'fine' and for 'everything is wrong'. The second is a
    full re-transcription, and a ledger that called it clean would hide the episode for good."""
    disk = _write_disk(tmp_path, broken=(0, 1, 2))
    before = disk.snapshot()

    result, rows, _ = _run(disk, Clips())

    assert result == (False, None, 0)
    assert rows == [("skipped", "WholeTranscriptUnpunctuated")]
    assert disk.snapshot() == before and archive.asked == []


def test_windows_that_are_all_refused_leave_the_episode_untouched(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    before = disk.snapshot()
    clips = Clips(_clip_answer(_window_of(disk, 1), punctuated=False))

    result, rows, pipeline_metrics = _run(disk, clips)

    assert result == (False, None, 0)
    assert len(clips.calls) == 1, "the flagged window WAS re-decoded"
    assert rows == [("skipped", "NoWindowAccepted")]
    assert "still_unpunctuated" in pipeline_metrics.episode_statuses[0].error_message
    assert disk.snapshot() == before, "a refusal writes nothing: not asr.json, not the manifest"


def test_a_failing_clip_call_keeps_the_transcript(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    before = disk.snapshot()

    result, rows, _ = _run(disk, Clips(ConnectionError("dgx down")))

    assert result == (False, None, 0)
    assert rows == [("skipped", "NoWindowAccepted")]
    assert disk.snapshot() == before


# --- the work-list: "make this episode right" is met when it already is ---------------------------


def _worklist_outcome(disk: Disk, provider: Any):
    from podcast_scraper.workflow.worklist_report import reset_worklist_report

    report = reset_worklist_report()
    report.request([GUID])
    report.mark_matched([GUID])
    _run(disk, provider)
    return report


def test_a_clean_episode_finishes_its_worklist_entry_so_a_repeat_run_does_not_fail(
    tmp_path, cuts, archive
):
    """Without this a repeat of a finished repair read 'repaired 0 of N' and exited 1."""
    report = _worklist_outcome(_write_disk(tmp_path, broken=()), Clips())

    assert report.completed == {GUID} and report.already_right == {GUID}
    assert not report.repaired_nothing


def test_the_work_list_entry_is_finished_even_with_no_metrics_object(tmp_path, cuts, archive):
    """Whether the request counts as met must not depend on a ledger being passed: a clean episode
    would silently go back to 'repaired 0/N' and fail the repeat run."""
    from podcast_scraper.workflow.worklist_report import reset_worklist_report

    report = reset_worklist_report()
    report.request([GUID])
    disk = _write_disk(tmp_path, broken=())

    result = ep._repunctuate_existing_transcript(
        _job(_episode(disk)), _cfg(), None, str(disk.out), Clips(), None
    )

    assert result == (False, None, 0)
    assert report.already_right == {GUID} and not report.repaired_nothing


def test_a_refused_window_does_not_finish_its_worklist_entry(tmp_path, cuts, archive):
    """The episode is still wrong; a run that repaired none of what it was asked to is a failure."""
    disk = _write_disk(tmp_path)
    report = _worklist_outcome(disk, Clips(_clip_answer(_window_of(disk, 1), punctuated=False)))

    assert report.completed == set() and report.repaired_nothing
    # ...but it ran and the guards refused: declined, never "did NOT finish" (2026-10-15).
    assert report.declined_ids == [GUID] and report.incomplete == []
    assert "left as they were by design (NoWindowAccepted 1)" in report.summary()


def test_a_wholly_unpunctuated_episode_does_not_finish_its_worklist_entry(tmp_path, cuts, archive):
    report = _worklist_outcome(_write_disk(tmp_path, broken=(0, 1, 2)), Clips())

    assert report.completed == set()


# --- an accepted window changes that window and nothing outside it ----------------------------


def _accept_window_1(tmp_path, **disk_kw):
    disk = _write_disk(tmp_path, **disk_kw)
    clips = Clips(_clip_answer(_window_of(disk, 1)))
    return disk, clips


def test_an_accepted_window_replaces_only_that_window(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path)
    old = json.loads(disk.seg.read_text(encoding="utf-8"))

    result, rows, _ = _run(disk, clips)

    assert result == (True, os.path.relpath(disk.txt, disk.out), 0)
    assert rows == [], "no skip or failure row: the cascade that follows marks the episode ok"
    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    inside = [s for s in new if WINDOW_S <= s["start"] < 2 * WINDOW_S]
    assert [s for s in new if s not in inside] == [
        s for s in old if not WINDOW_S <= s["start"] < 2 * WINDOW_S
    ], "outside the window every segment is the same dict, field for field"
    assert [json.dumps(s) for s in new if s not in inside] == [
        json.dumps(s) for s in old if not WINDOW_S <= s["start"] < 2 * WINDOW_S
    ], "...and serialises byte for byte"
    assert len(inside) == SEGS_PER_WINDOW // 2 and all(s["punctuation_repaired"] for s in inside)
    assert cuts == [(600.0, 1200.0)], "one clip, the flagged window cut at its segment edges"
    assert len(clips.calls) == 1


def test_spliced_segments_carry_the_voice_of_the_segments_they_replaced(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path)
    old = json.loads(disk.seg.read_text(encoding="utf-8"))

    _run(disk, clips)

    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    inside = [s for s in new if s.get("punctuation_repaired")]
    assert inside
    for seg in inside:
        mid = (seg["start"] + seg["end"]) / 2
        was = next(o for o in old if o["start"] <= mid < o["end"])
        for field in ("speaker", "speaker_label", "speaker_role", "voice_type", "stated_speaker"):
            assert seg.get(field) == was.get(field), (field, seg["start"])
    assert {s["speaker_label"] for s in inside} == {"Alice Host", "Bob Guest"}


def test_a_segment_repaired_by_an_earlier_pass_is_not_relabelled(tmp_path, cuts, archive):
    """Spliced means 'not one of the stored segments', not 'carries ``punctuation_repaired``': the
    tag survives in the file, and a one-second backchannel an earlier repair put inside a host's
    long segment would be handed to the host by overlap."""
    disk, clips = _accept_window_1(tmp_path)
    segs = json.loads(disk.seg.read_text(encoding="utf-8"))
    backchannel = {
        "start": 102.0,
        "end": 103.0,
        "text": "Yeah.",
        **BOB,
        "punctuation_repaired": True,
    }
    at = next(i for i, s in enumerate(segs) if s["start"] == 110.0)
    segs[at:at] = [backchannel]
    for s in segs:
        if 100.0 <= s["start"] < 110.0:
            s.update(ALICE)
    disk.seg.write_text(json.dumps(segs), encoding="utf-8")

    _run(disk, clips)

    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    assert backchannel in new, "kept exactly as stored: same speaker, same tag"


def test_the_screenplay_is_re_rendered_from_the_new_segments(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path)
    old_text = disk.txt.read_text(encoding="utf-8")

    _run(disk, clips)

    text = disk.txt.read_text(encoding="utf-8")
    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    assert text == format_diarized_screenplay_from_segments(new)
    assert text.startswith("Alice Host: ") and "\nBob Guest: " in text
    assert text != old_text
    from podcast_scraper.transcription.punctuation import sentence_ends_per_1000_words

    assert sentence_ends_per_1000_words(text) > 25, "window 1 is punctuated now, not just present"


def test_no_segment_keeps_a_stale_character_offset(tmp_path, cuts, archive):
    """Offsets index the text BEFORE the splice. A kept one would point GI at the wrong words."""
    disk, clips = _accept_window_1(tmp_path, char_offsets=True)

    _run(disk, clips)

    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    assert not any("char_start" in s or "char_end" in s for s in new)


def test_a_plain_render_gets_fresh_offsets_that_index_the_new_text(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path, char_offsets=True)
    # No speaker labels to render a screenplay from: the plain, one-line-per-segment render.
    segs = json.loads(disk.seg.read_text(encoding="utf-8"))
    plain = [
        {k: v for k, v in s.items() if not k.startswith("speaker") and k != "stated_speaker"}
        | {"voice_type": None}
        for s in segs
    ]
    for s in plain:
        s.pop("voice_type")
    disk.seg.write_text(json.dumps(plain), encoding="utf-8")
    disk.txt.write_text("\n".join(s["text"] for s in plain), encoding="utf-8")

    _run(disk, Clips(_clip_answer(_window_of(disk, 1))), _cfg(diarize=False, screenplay=False))

    text = disk.txt.read_text(encoding="utf-8")
    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    assert all(text[s["char_start"] : s["char_end"]] == s["text"] for s in new)


# --- the records that describe the transcript are patched, not rebuilt ------------------------


def test_asr_json_gets_this_pass_appended_and_keeps_everything_else(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path)
    before = json.loads(disk.asr.read_text(encoding="utf-8"))

    _run(disk, clips)

    after = json.loads(disk.asr.read_text(encoding="utf-8"))
    assert "unpunctuated_windows" not in after, "no window is left flagged"
    kept = {
        k: v for k, v in after.items() if k not in ("punctuation_repair", "unpunctuated_windows")
    }
    assert kept == {
        k: v for k, v in before.items() if k not in ("punctuation_repair", "unpunctuated_windows")
    }, "model, coverage, language, recovery and trim records are untouched"
    assert after["punctuation_repair"][:1] == before["punctuation_repair"], "history is kept"
    attempt = after["punctuation_repair"][-1]
    assert len(after["punctuation_repair"]) == 2
    assert attempt["pass"] == "repunctuate_only" and attempt["status"] == "repaired"
    assert attempt["repair_model"] == "Systran/faster-whisper-large-v3"
    assert attempt["original_model"] == ORIGINAL_MODEL, "model drift is visible per attempt"
    assert attempt["audio_source"] == "archive"
    assert attempt["audio_identity_verified"] is False, "the archived copy may be another encode"


def test_the_manifest_drops_the_repaired_flag_and_keeps_the_rest_of_the_asr_block(
    tmp_path, cuts, archive
):
    disk, clips = _accept_window_1(tmp_path)
    before = json.loads(disk.manifest.read_text(encoding="utf-8"))

    _run(disk, clips)

    after = json.loads(disk.manifest.read_text(encoding="utf-8"))
    assert "asr_partly_unpunctuated" not in after["quality_flags"]
    assert "asr_punctuation_repaired" in after["quality_flags"]
    assert "asr_failover" in after["quality_flags"], "a flag this pass knows nothing about stays"
    asr_before, asr_after = before["stages"]["asr"], after["stages"]["asr"]
    assert asr_after["metrics"]["unpunctuated_windows"] == 0
    assert asr_after["metrics"]["punctuation_windows_repaired"] == 1
    for key in ("method", "model", "method_version", "duration_s", "cost_usd", "failover"):
        assert asr_after[key] == asr_before[key], key
    assert asr_after["metrics"]["speech_coverage"] == 0.93
    assert after["stages"]["diarization"] == before["stages"]["diarization"]
    assert after["stages"]["naming"] == before["stages"]["naming"]


def test_a_partial_repair_keeps_the_flag_and_the_window_it_could_not_fix(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path, broken=(0, 1))
    clips = Clips(
        _clip_answer(_window_of(disk, 0)),
        _clip_answer(_window_of(disk, 1), punctuated=False),
    )

    result, _rows, _ = _run(disk, clips)

    assert result[0] is True
    asr = json.loads(disk.asr.read_text(encoding="utf-8"))
    assert asr["unpunctuated_windows"] == [[600.0, 1200.0]]
    attempts = [a for a in asr["punctuation_repair"] if a.get("pass") == "repunctuate_only"]
    assert [a["status"] for a in attempts] == ["repaired", "refused"]
    m = json.loads(disk.manifest.read_text(encoding="utf-8"))
    assert "asr_partly_unpunctuated" in m["quality_flags"], "one window is still unpunctuated"
    assert m["stages"]["asr"]["metrics"]["unpunctuated_windows"] == 1
    assert m["stages"]["asr"]["metrics"]["punctuation_windows_repaired"] == 1


def test_running_it_again_has_nothing_left_to_do(tmp_path, cuts, archive):
    """Repeatable by construction: windows come from the transcript, not from a record of tries."""
    disk, clips = _accept_window_1(tmp_path)
    _run(disk, clips)
    after_first = disk.snapshot()

    result, rows, _ = _run(disk, Clips())

    assert result == (False, None, 0) and rows == [("skipped", "NothingToRepunctuate")]
    assert disk.snapshot() == after_first


# --- the files derived from the old text -------------------------------------------------------


def test_derived_files_follow_the_new_text(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path)

    _run(disk, clips)

    assert not disk.f(".cleaned.txt").exists(), "rewritten by the next summary, never served stale"
    adfree = disk.f(".adfree.txt")
    assert adfree.exists() and adfree.read_text(encoding="utf-8") != "STALE FROM THE OLD TEXT"
    assert "STALE" not in disk.f(".turns.json").read_text(encoding="utf-8"), "turns rewritten"
    turns = json.loads(disk.f(".turns.json").read_text(encoding="utf-8"))
    assert turns, "and describe the new transcript"
    assert disk.f(".speakers.diagnostics.json").read_text(encoding="utf-8") == '{"voices": []}'


def test_a_stale_adfree_base_is_removed_when_nothing_will_regenerate_it(tmp_path, cuts, archive):
    """GI reads ``.adfree.txt`` in preference to the transcript. Left in place after the splice it
    would extract insights and quotes from words that have since been replaced."""
    disk, clips = _accept_window_1(tmp_path)

    _run(disk, clips, _cfg(save_adfree_transcript=False))

    for suffix in (".adfree.txt", ".adfree.segments.json", ".adfree.admap.json"):
        assert not disk.f(suffix).exists(), suffix
    anon = disk.f(".anon.txt")  # regenerated from the new segments, because it differs from .txt
    assert anon.exists() and "STALE" not in anon.read_text(encoding="utf-8")


def test_an_invented_line_in_a_repaired_window_is_dropped_and_recorded(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    answer = _clip_answer(_window_of(disk, 1))
    answer["segments"].append({"start": 599.0, "end": 599.5, "text": "Thanks for watching!"})
    answer["segments"][-1]["start"] = 598.0
    answer["segments"][-1]["end"] = 599.0

    result, _rows, _ = _run(disk, Clips(answer))

    assert result[0] is True
    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    assert not any("Thanks for watching" in s["text"] for s in new)
    asr = json.loads(disk.asr.read_text(encoding="utf-8"))
    assert [r["text"] for r in asr["invented_lines_removed"]] == ["Thanks for watching!"]


def test_a_write_failure_leaves_the_transcript_exactly_as_it_was(tmp_path, cuts, archive):
    disk, clips = _accept_window_1(tmp_path)
    before = {p: disk.f(p).read_bytes() for p in (".txt", ".segments.json", ".asr.json")}
    real_replace = os.replace

    def _fail_on_the_text(src, dst):
        if str(dst) == str(disk.txt):
            raise OSError("disk full")
        return real_replace(src, dst)

    with patch.object(ep.os, "replace", _fail_on_the_text):
        result, rows, _ = _run(disk, clips)

    assert result == (False, None, 0) and rows == [("failed", "WriteFailed")]
    assert {p: disk.f(p).read_bytes() for p in before} == before, "text, segments and asr.json"
    assert not list(disk.run.rglob("*.repunct.*")), "no staged file left behind"


# --- where the audio comes from -----------------------------------------------------------------


def test_the_audio_is_fetched_by_guid_from_the_archive_and_needs_no_media_url(
    tmp_path, cuts, archive
):
    disk, clips = _accept_window_1(tmp_path)

    result, _rows, _ = _run(disk, clips, media_url=None)

    assert result[0] is True
    assert ("exists", _archive_key(".mp3")) in archive.asked
    assert ("download", _archive_key(".mp3")) in archive.asked
    audio, language, prompt = clips.calls[0]
    assert language == "en" and prompt and prompt.startswith("Hello"), "the English prompt"


def test_the_publisher_is_the_fallback_when_the_archive_misses(tmp_path, cuts, monkeypatch):
    disk, clips = _accept_window_1(tmp_path)
    monkeypatch.setattr(audio_cache, "resolve_backend", lambda *_a, **_k: Archive())

    def _download(episode, _cfg, dest, _metrics, _out):
        Path(dest).write_bytes(b"ID3 from the publisher")
        return True, 22, 0.0

    monkeypatch.setattr(ep, "_download_or_reuse_media", _download)

    result, _rows, _ = _run(disk, clips, media_url="https://example.com/ep.mp3")

    assert result[0] is True
    attempt = json.loads(disk.asr.read_text(encoding="utf-8"))["punctuation_repair"][-1]
    assert attempt["audio_source"] == "download"


def test_no_archived_audio_is_a_recorded_failure_and_writes_nothing(tmp_path, cuts, monkeypatch):
    disk, clips = _accept_window_1(tmp_path)
    before = disk.snapshot()
    monkeypatch.setattr(audio_cache, "resolve_backend", lambda *_a, **_k: Archive())

    result, rows, _ = _run(disk, clips, media_url=None)

    assert result == (False, None, 0)
    assert rows == [("failed", "NoArchivedAudio")]
    assert clips.calls == [] and disk.snapshot() == before


def test_audio_that_is_fetched_is_removed_afterwards(tmp_path, cuts, archive, monkeypatch):
    import tempfile

    disk, clips = _accept_window_1(tmp_path)
    made: List[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*a, **k):
        path = real_mkdtemp(*a, **k)
        made.append(path)
        return path

    monkeypatch.setattr(ep.tempfile, "mkdtemp", _spy)

    _run(disk, clips)

    repunct = [p for p in made if os.path.basename(p).startswith("repunct_")]
    assert repunct and not any(os.path.exists(p) for p in repunct)


# --- refusals ----------------------------------------------------------------------------------


def test_a_changed_language_is_a_recorded_failure(tmp_path, cuts, archive):
    """The transcript was made as Spanish; decoding its windows as English would splice in a
    translation-flavoured hallucination under the original speakers."""
    disk = _write_disk(tmp_path, asr_language="es")
    before = disk.snapshot()

    result, rows, _ = _run(disk, Clips(), _cfg(feed_declared_language="en"))

    assert result == (False, None, 0) and rows == [("failed", "LanguageChanged")]
    assert disk.snapshot() == before and archive.asked == []


def test_a_provider_without_a_clip_call_is_a_recorded_failure(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    before = disk.snapshot()

    result, rows, _ = _run(disk, SimpleNamespace(transcribe_with_segments=lambda *a, **k: None))

    assert result == (False, None, 0) and rows == [("failed", "NoClipProvider")]
    assert disk.snapshot() == before and archive.asked == []


def test_a_language_with_no_punctuated_prompt_is_a_skip_before_any_fetch(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path, asr_language="ja")
    before = disk.snapshot()

    result, rows, _ = _run(disk, Clips(), _cfg(feed_declared_language="ja"))

    assert result == (False, None, 0) and rows == [("skipped", "NoPromptForLanguage")]
    assert disk.snapshot() == before and archive.asked == []


def test_a_transcript_without_asr_provenance_is_a_skip_before_any_fetch(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    disk.asr.unlink()
    before = disk.snapshot()

    result, rows, _ = _run(disk, Clips())

    assert result == (False, None, 0) and rows == [("skipped", "NoAsrProvenance")]
    assert disk.snapshot() == before and archive.asked == []


def test_no_segments_file_is_a_recorded_failure(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    disk.seg.unlink()

    result, rows, _ = _run(disk, Clips())

    assert result == (False, None, 0) and rows == [("failed", "NoSegments")]


def test_an_episode_whose_own_transcript_is_gone_is_a_recorded_failure(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path)
    disk.txt.unlink()

    result, rows, _ = _run(disk, Clips())

    assert result == (False, None, 0) and rows == [("failed", "TranscriptUnresolved")]


# --- a translated episode: the canonical pair is the English render ----------------------------


def test_a_translated_episode_is_read_from_its_source_pair_without_swapping(
    tmp_path, cuts, archive
):
    """Repunctuating the English render would patch words nobody said. Nothing is accepted here, so
    nothing may move: the English render stays where readers expect it."""
    disk = _write_disk(tmp_path, asr_language="es", tagged_language="es")
    before = disk.snapshot()
    source = json.loads(disk.f(".es.segments.json").read_text(encoding="utf-8"))
    clips = Clips(_clip_answer(source[SEGS_PER_WINDOW : 2 * SEGS_PER_WINDOW], punctuated=False))

    result, rows, _ = _run(disk, clips, _cfg(feed_declared_language="es"))

    assert result == (False, None, 0) and rows == [("skipped", "NoWindowAccepted")]
    assert len(clips.calls) == 1, "the source segments' flagged window WAS found and decoded"
    assert clips.calls[0][1] == "es" and clips.calls[0][2].startswith("Hola")
    assert disk.snapshot() == before, "no swap, no invalidation, no write on a refusal"


def test_a_translated_episode_swaps_back_only_when_a_window_is_accepted(tmp_path, cuts, archive):
    disk = _write_disk(tmp_path, asr_language="es", tagged_language="es")
    source = json.loads(disk.f(".es.segments.json").read_text(encoding="utf-8"))
    clips = Clips(_clip_answer([s for s in source if WINDOW_S <= s["start"] < 2 * WINDOW_S]))

    result, _rows, _ = _run(disk, clips, _cfg(feed_declared_language="es"))

    assert result[0] is True
    assert not disk.f(".es.txt").exists() and not disk.f(".es.segments.json").exists()
    assert "ENGLISH RENDER" not in disk.txt.read_text(encoding="utf-8"), "the stale render is gone"
    new = json.loads(disk.seg.read_text(encoding="utf-8"))
    assert len(new) == len(source) - SEGS_PER_WINDOW // 2 and any(
        s.get("punctuation_repaired") for s in new
    )
    assert disk.txt.read_text(encoding="utf-8") == format_diarized_screenplay_from_segments(new)


# --- how the stage is reached -------------------------------------------------------------------


def test_the_audio_route_hands_it_a_job_without_downloading_anything(tmp_path):
    disk = _write_disk(tmp_path)

    with patch.object(ep, "_download_or_reuse_media", side_effect=AssertionError("fetched")):
        job = ep.download_media_for_transcription(
            _episode(disk, media_url="https://example.com/ep.mp3"),
            _cfg(),
            str(tmp_path / "tmp"),
            str(disk.out),
            None,
        )

    assert job is not None and job.temp_media == ""


def test_the_early_skip_check_never_claims_a_present_episode_for_it(tmp_path):
    """``skip_existing`` says 'already have it'; for this stage that is the precondition."""
    disk = _write_disk(tmp_path)

    verdict = ep.presence_skip_evidence(
        _episode(disk), _cfg(skip_existing=True), str(disk.out), None, str(tmp_path / "tmp")
    )

    assert verdict is None


def test_the_dispatch_hands_the_stage_to_the_repunctuation_path(tmp_path, monkeypatch):
    disk = _write_disk(tmp_path)
    monkeypatch.setattr(ep, "_repunctuate_existing_transcript", lambda *a, **k: "dispatched")

    got = ep._maybe_dispatch_reprocess_stage(
        _job(_episode(disk)), _cfg(), None, str(disk.out), None, None
    )

    assert got == "dispatched"


# --- the cheap precheck: a clean episode costs a file read, not an LLM call ---------------------


def _prepare(disk: Disk, episodes: List[Episode], cfg: config.Config):
    pipeline_metrics = metrics_mod.Metrics()
    resources = SimpleNamespace(
        temp_dir=None, transcription_jobs=None, transcription_jobs_lock=None
    )
    with (
        patch.object(processing, "_check_episode_size_skip", return_value=processing._NO_SIZE_SKIP),
        patch.object(processing, "_detect_speakers_for_episode", return_value=None) as detect,
    ):
        args = processing.prepare_episode_download_args(
            episodes,
            cfg,
            str(disk.out),
            None,
            cast(Any, resources),
            cast(Any, SimpleNamespace(cached_hosts=set())),
            pipeline_metrics,
        )
    return args, detect, pipeline_metrics


def test_a_clean_episode_is_ruled_out_before_speaker_detection(tmp_path, caplog):
    clean = _write_disk(tmp_path / "a", broken=())
    flagged = _write_disk(tmp_path / "b", broken=(1,))
    ep_clean, ep_flagged = _episode(clean), _episode(flagged)
    ep_flagged.idx = 2

    with caplog.at_level("WARNING"):
        args, detect, pipeline_metrics = _prepare(clean, [ep_clean, ep_flagged], _cfg())

    assert [a[0] for a in args] == [ep_flagged], "only the flagged episode is queued"
    assert [c.args[0] for c in detect.call_args_list] == [ep_flagged], "no LLM call for the clean"
    assert ("skipped", "NothingToRepunctuate") in [
        (s.status, s.error_type) for s in pipeline_metrics.episode_statuses
    ]
    assert "produced ZERO processing jobs" not in caplog.text


def test_a_corpus_that_is_all_clean_is_not_reported_as_a_stage_that_built_no_work(tmp_path, caplog):
    clean = _write_disk(tmp_path, broken=())

    with caplog.at_level("WARNING"):
        args, detect, _ = _prepare(clean, [_episode(clean)], _cfg())

    assert args == [] and detect.call_count == 0
    assert "produced ZERO processing jobs" not in caplog.text, "an all-clean corpus is the goal"


def test_the_precheck_leaves_the_decision_to_the_stage_when_it_cannot_tell(tmp_path):
    disk = _write_disk(tmp_path, broken=(1,))
    pipeline_metrics = metrics_mod.Metrics()
    episode = _episode(disk)

    assert ep.repunctuate_nothing_to_do(episode, _cfg(), pipeline_metrics) is False, "has a window"
    episode.on_disk_transcript = None
    assert ep.repunctuate_nothing_to_do(episode, _cfg(), pipeline_metrics) is False, "no record"
    episode.on_disk_transcript = str(disk.txt)
    disk.seg.unlink()
    assert ep.repunctuate_nothing_to_do(episode, _cfg(), pipeline_metrics) is False, "no segments"
    assert pipeline_metrics.episode_statuses == []
    other = _cfg(pipeline_stage="relabel_only", transcription_provider="whisper")
    assert ep.repunctuate_nothing_to_do(episode, other, pipeline_metrics) is False, "other stage"


def test_the_stage_fetches_nothing_for_an_episode_the_precheck_ruled_out(tmp_path):
    """The stage and the precheck read the same segments and so agree; the run never reaches
    ``_archived_audio_for_repunctuate`` for a clean episode."""
    disk = _write_disk(tmp_path, broken=())
    pipeline_metrics = metrics_mod.Metrics()
    assert ep.repunctuate_nothing_to_do(_episode(disk), _cfg(), pipeline_metrics) is True

    with patch.object(ep, "_archived_audio_for_repunctuate", side_effect=AssertionError("fetched")):
        result, _rows, _ = _run(disk, Clips())

    assert result == (False, None, 0)
