"""A publisher transcript that names its turns must produce a PLACED roster — no audio, no model.

THE DEFECT. ``transcript_formats.cues`` lifts WebVTT ``<v Speaker>`` voice spans into
``seg["speaker"]``, but on the transcript-DOWNLOAD path nothing turned them into a roster. The
names went nowhere: the episode arrived as one undifferentiated voice, ``content.speakers`` was
derived from segments carrying no ``speaker_label``, and nothing was ever ``placed``. Measured on
the production corpus, EVERY episode that used a publisher transcript ended with a single voice —
128 of 128 (recorded in ``cues.py``).

WHY IT MATTERED HERE. ``tests/stack-test`` mounts only the RSS fixtures that ship a transcript for
every item, deliberately, so Whisper never runs. Every stack-test episode took that path, was never
diarized, and after the #2075 operator decision ("an episode never diarized, or diarized with
nobody named, casts nobody") the KG correctly emitted zero ``Person`` nodes — turning the whole
Person-rail surface red while the pipeline behaved exactly as specified.

HOW THE FIRST ATTEMPT BROKE MAIN, because these tests exist to make that unrepeatable. be1ed96d
called the roster helper with the ``Episode``, on the belief that it carried
``detected_speaker_names`` / ``metadata_named`` / ``feed_hosts``. It does not — those belong to
``TranscriptionJob`` — so every transcript download died with ``AttributeError`` and
``transcripts_saved=0``. The unit test written alongside it passed, because its fixture SET those
attributes on the Episode by hand and so manufactured the very thing whose absence was the bug.

Two rules follow, and both are asserted below: the roster inputs are passed EXPLICITLY (a
keyword-only signature cannot be satisfied by accident), and the wiring test drives the real entry
point ``process_transcript_download`` with a real ``Episode``, never a decorated stand-in.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from podcast_scraper import config as config_module
from podcast_scraper.models.entities import Episode
from podcast_scraper.transcript_formats import parse_webvtt
from podcast_scraper.workflow import episode_processor as epx

pytestmark = [pytest.mark.unit]


_NAMED = """WEBVTT

00:00:00.000 --> 00:00:09.000
<v Maya>Welcome back to Singletrack Sessions. I'm joined by Liam, who builds trail for the \
Cascadia Alliance. Liam, thanks for being here.</v>

00:00:09.000 --> 00:00:15.000
<v Liam>Thanks, Maya. Glad to be here — this is something I think about a lot.</v>

00:00:15.000 --> 00:00:24.000
<v Maya>Let's start there. What is the most underrated piece of trail building?</v>

00:00:24.000 --> 00:00:36.000
<v Liam>Drainage. If water leaves on its own the trail lasts a decade, and everything else \
follows from that one decision.</v>
"""

#: A publisher who ships cues but never labels them — the ordinary case, which must not change.
_UNNAMED = """WEBVTT

00:00:00.000 --> 00:00:09.000
Welcome back to Singletrack Sessions.

00:00:09.000 --> 00:00:15.000
Thanks for having me.
"""

_FEED_HOSTS = ["Maya"]
_DETECTED = ["Maya", "Liam"]


def _episode() -> Any:
    """A REAL Episode, exactly as the pipeline builds one — nothing added."""
    return Episode(
        idx=1,
        title="Building Trails That Last",
        title_safe="building-trails",
        item=ET.Element("item"),
        transcript_urls=[("http://feed.example/t.vtt", "text/vtt")],
    )


def _roster(vtt: str, *, feed_hosts: Optional[List[str]] = None) -> Tuple[dict, List[dict]]:
    plain, segments = parse_webvtt(vtt)
    out = epx._roster_native_segments(
        {"text": plain, "segments": segments},
        config_module.Config(),
        idx=1,
        detected_speaker_names=_DETECTED,
        metadata_named=_DETECTED,
        feed_hosts=_FEED_HOSTS if feed_hosts is None else feed_hosts,
        detection_ran=True,
    )
    return out, list(out.get("segments") or [])


def test_the_voice_spans_become_named_placed_voices() -> None:
    """The names the file states must reach the segments, which is where ``content.speakers`` is
    derived from (``metadata_generation``, #876)."""
    out, segs = _roster(_NAMED)
    labelled = [s for s in segs if s.get("speaker_label")]
    assert labelled, (
        "a transcript that names every turn produced no speaker_label at all — the voice spans "
        "were parsed and then discarded, which is the 128-of-128 defect"
    )
    assert {str(s["speaker_label"]) for s in labelled} == {"Maya", "Liam"}
    assert out.get("diarization_num_speakers") == 2


def test_the_roles_come_from_the_roster_not_from_cue_order() -> None:
    """A publisher transcript states WHO speaks, never who HOSTS, so roles must be resolved as
    evidence — the guess-from-ordering bug the roster exists to replace."""
    _, segs = _roster(_NAMED)
    roles = {
        str(s.get("speaker_label")): s.get("speaker_role") for s in segs if s.get("speaker_label")
    }
    assert roles == {"Maya": "host", "Liam": "guest"}, roles


def test_the_publisher_names_its_voices_without_any_feed_host() -> None:
    """THE GATE THIS REPLACED. An earlier attempt skipped the roster entirely when no feed host
    was known, because without one the roster could only emit bare ``SPEAKER_NN`` ids — true only
    while the publisher's names were being discarded before it ever saw them.

    With the names carried through, the file alone is sufficient: identity comes from the source,
    roles from the conversation. The skip was also never inert — it left the raw ``speaker`` on the
    segments, where ``adfree`` promoted it and ``metadata_generation`` defaulted a role-less name
    to host + placed, publishing the publisher's GUEST as a placed host.
    """
    _out, segs = _roster(_NAMED, feed_hosts=[])
    labels = {str(s.get("speaker_label")) for s in segs if s.get("speaker_label")}
    assert labels == {
        "Maya",
        "Liam",
    }, f"the transcript names these voices; a missing feed host must not silence it: {labels}"


def test_a_transcript_without_voice_spans_is_untouched() -> None:
    """Gated on the DATA, so an ordinary unlabelled transcript keeps its behaviour exactly."""
    out, segs = _roster(_UNNAMED)
    assert not any(s.get("speaker_label") for s in segs)
    assert not out.get("speaker_diagnostics")


def test_it_never_invents_a_voice_the_file_did_not_name() -> None:
    """#876: a heuristic may identify a voice, never author a name."""
    _, segs = _roster(_NAMED)
    stated = {s["speaker"] for s in parse_webvtt(_NAMED)[1] if s.get("speaker")}
    for s in segs:
        label = s.get("speaker_label")
        if label:
            assert str(label) in stated, f"{label!r} is a name the transcript never said"


# --------------------------------------------------------------------------------------------
# The wiring. These drive the real entry point; the ones above would all have passed on the
# broken code, because the helper was never the thing that was wrong.
# --------------------------------------------------------------------------------------------


def test_episode_does_not_carry_the_roster_fields() -> None:
    """The exact false belief that broke main in be1ed96d, written down.

    ``Episode`` and ``TranscriptionJob`` live in the same module and the grep that "confirmed"
    this matched the wrong class. If someone later adds these to ``Episode``, this test should be
    deleted deliberately — not discovered by a production crash.
    """
    ep = _episode()
    for field in (
        "detected_speaker_names",
        "metadata_named",
        "feed_hosts",
        "speaker_detection_ran",
    ):
        assert not hasattr(ep, field), (
            f"Episode now has {field!r}; the download path was written on the assumption that it "
            "does not, and passes these explicitly instead"
        )


def test_the_download_path_rosters_a_named_transcript(tmp_path, monkeypatch) -> None:
    """END TO END through `process_transcript_download` with a real Episode.

    This is the test the first attempt did not have. It would have failed on be1ed96d with
    ``AttributeError: 'Episode' object has no attribute 'detected_speaker_names'`` instead of
    letting CI discover it.
    """
    monkeypatch.setattr(
        epx,
        "_fetch_transcript_content",
        lambda url, cfg: (_NAMED.encode("utf-8"), "text/vtt"),
    )
    cfg = config_module.Config(output_dir=str(tmp_path))

    ok, rel_path, source, _ = epx.process_transcript_download(
        _episode(),
        "http://feed.example/t.vtt",
        "text/vtt",
        cfg,
        str(tmp_path),
        None,
        detected_speaker_names=_DETECTED,
        metadata_named=_DETECTED,
        feed_hosts=_FEED_HOSTS,
    )

    assert ok and rel_path, f"the download failed outright: ok={ok} path={rel_path}"
    assert source == "direct_download"

    segments_files = list(Path(tmp_path).rglob("*.segments.json"))
    assert segments_files, "no .segments.json was saved, so content.speakers has nothing to read"

    import json

    saved = json.loads(segments_files[0].read_text(encoding="utf-8"))
    rows = saved if isinstance(saved, list) else saved.get("segments") or []
    roles = {s.get("speaker_label"): s.get("speaker_role") for s in rows if s.get("speaker_label")}
    assert roles == {
        "Maya": "host",
        "Liam": "guest",
    }, f"the saved segments must carry the roster's names and roles, got {roles}"


def test_the_download_path_leaves_an_unlabelled_transcript_alone(tmp_path, monkeypatch) -> None:
    """The no-op case, driven through the same entry point: no labels, and no crash."""
    monkeypatch.setattr(
        epx,
        "_fetch_transcript_content",
        lambda url, cfg: (_UNNAMED.encode("utf-8"), "text/vtt"),
    )
    cfg = config_module.Config(output_dir=str(tmp_path))

    ok, rel_path, _, _ = epx.process_transcript_download(
        _episode(),
        "http://feed.example/t.vtt",
        "text/vtt",
        cfg,
        str(tmp_path),
        None,
        detected_speaker_names=_DETECTED,
        metadata_named=_DETECTED,
        feed_hosts=_FEED_HOSTS,
    )
    assert ok and rel_path

    import json

    for path in Path(tmp_path).rglob("*.segments.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = saved if isinstance(saved, list) else saved.get("segments") or []
        assert not any(s.get("speaker_label") for s in rows)


def test_the_download_path_survives_a_feed_with_no_hosts(tmp_path, monkeypatch) -> None:
    """A feed whose hosts could not be resolved must still download its transcript.

    The regression this guards is the shape of the one that broke main: an input the download path
    cannot always supply, reached for unconditionally.
    """
    monkeypatch.setattr(
        epx,
        "_fetch_transcript_content",
        lambda url, cfg: (_NAMED.encode("utf-8"), "text/vtt"),
    )
    cfg = config_module.Config(output_dir=str(tmp_path))

    ok, rel_path, _, _ = epx.process_transcript_download(
        _episode(), "http://feed.example/t.vtt", "text/vtt", cfg, str(tmp_path), None
    )
    assert ok and rel_path, "a transcript download must not depend on host detection succeeding"


_NO_TURNS = """WEBVTT

00:00:00.000 --> 00:00:09.000
Welcome back to the show, and today we are talking about trail building.

00:00:09.000 --> 00:00:15.000
It is something I have been thinking about for a long time.
"""


def _download(tmp_path, monkeypatch, body: str, **cfg_kw):
    monkeypatch.setattr(
        epx, "_fetch_transcript_content", lambda url, cfg: (body.encode("utf-8"), "text/vtt")
    )
    cfg = config_module.Config(output_dir=str(tmp_path), **cfg_kw)
    return epx.process_transcript_download(
        _episode(),
        "http://feed.example/t.vtt",
        "text/vtt",
        cfg,
        str(tmp_path),
        None,
        detected_speaker_names=_DETECTED,
        metadata_named=_DETECTED,
        feed_hosts=_FEED_HOSTS,
    )


def test_a_turnless_transcript_is_accepted_by_default(tmp_path, monkeypatch) -> None:
    """The gate is OPT-IN. Turning it on moves real feeds onto ASR, which costs GPU time, so the
    default must not change behaviour for anyone who has not asked for it."""
    ok, rel_path, source, _ = _download(tmp_path, monkeypatch, _NO_TURNS)
    assert ok and rel_path
    assert source == "direct_download"


def test_a_turnless_transcript_is_refused_when_the_operator_requires_speakers(
    tmp_path, monkeypatch
) -> None:
    """Refused BEFORE anything is written, and reported with a distinct source so the caller can
    tell a deliberate refusal from a failed download."""
    ok, rel_path, source, _ = _download(
        tmp_path, monkeypatch, _NO_TURNS, require_transcript_speakers=True
    )
    assert not ok
    assert rel_path is None
    assert source == epx.TRANSCRIPT_LACKS_SPEAKERS
    assert not list(
        Path(tmp_path).rglob("*.txt")
    ), "nothing may be written for a refused transcript"


def test_a_transcript_that_separates_turns_is_kept_even_when_speakers_are_required(
    tmp_path, monkeypatch
) -> None:
    """The gate asks whether turns are SEPARATED, not whether the labels are names. Anonymous
    labels are diarization we did not have to compute, and must not be thrown away."""
    anonymous = _NAMED.replace("<v Maya>", "<v Speaker 1>").replace("<v Liam>", "<v Speaker 2>")
    ok, rel_path, source, _ = _download(
        tmp_path, monkeypatch, anonymous, require_transcript_speakers=True
    )
    assert ok and rel_path
    assert source == "direct_download"


def test_the_caller_really_reaches_transcription(tmp_path, monkeypatch) -> None:
    """DRIVE IT, do not grep for it.

    The first version of this test asserted that the string "TRANSCRIPT_LACKS_SPEAKERS" appeared
    in the caller's source — which would pass on code that read the sentinel and then did nothing
    with it. A refusal that does not reach ASR leaves the episode with no transcript at all, the
    opposite of the intent, so the thing to assert is that the audio path was actually entered.
    """
    import queue as _queue

    monkeypatch.setattr(
        epx, "_fetch_transcript_content", lambda url, cfg: (_NO_TURNS.encode("utf-8"), "text/vtt")
    )
    reached: List[str] = []

    def _spy(*args, **kwargs):
        reached.append("download_media_for_transcription")
        return None

    monkeypatch.setattr(epx, "download_media_for_transcription", _spy)

    cfg = config_module.Config(
        output_dir=str(tmp_path), require_transcript_speakers=True, transcribe_missing=True
    )
    epx.process_episode_download(
        _episode(),
        cfg,
        str(tmp_path),
        str(tmp_path),
        None,
        _queue.Queue(),
        None,
        detected_speaker_names=_DETECTED,
        metadata_named=_DETECTED,
        feed_hosts=_FEED_HOSTS,
    )

    assert reached == ["download_media_for_transcription"], (
        "a refused transcript must fall through to the audio path; without it the episode is "
        "simply dropped"
    )
