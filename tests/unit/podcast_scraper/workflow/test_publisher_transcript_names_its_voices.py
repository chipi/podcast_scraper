"""A publisher transcript that names its turns must produce a PLACED roster — no audio, no model.

THE DEFECT. ``transcript_formats.cues`` lifts WebVTT ``<v Speaker>`` voice spans into
``seg["speaker"]``, but on the transcript-DOWNLOAD path nothing turned them into a roster. The
names went nowhere: the episode arrived as one undifferentiated voice, ``content.speakers`` was
derived from segments carrying no ``speaker_label``, and nothing was ever ``placed``. Measured on
the production corpus, EVERY episode that used a publisher transcript ended with a single voice —
128 of 128 (recorded in ``cues.py``).

WHY IT MATTERED HERE. ``tests/stack-test`` mounts only the RSS fixtures that ship a transcript for
every item, deliberately, so Whisper never runs. So every stack-test episode took that path, was
never diarized, and after the #2075 operator decision ("an episode never diarized, or diarized with
nobody named, casts nobody") the KG correctly emitted zero ``Person`` nodes — turning the whole
Person-rail surface red while the pipeline was behaving exactly as specified. The stack could not
exercise speaker attribution at all; what it had been "covering" was the show-notes GUESS path that
#876/#2075 removed.

THE FIX is not to make the stack run a diarizer — that is the expensive processing the transcript
fast path exists to avoid. It is to treat a transcript that already says who is speaking as what it
is: a diarization we did not have to compute, routed through the same role authority as every other
source via ``precomputed_diarization``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, List, Tuple

import pytest

from podcast_scraper import config as config_module
from podcast_scraper.models.entities import Episode
from podcast_scraper.transcript_formats import parse_webvtt
from podcast_scraper.workflow.episode_processor import _apply_native_speaker_roster

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

#: The same episode from a publisher who ships cues but never labels them — the ordinary case, and
#: the one that must keep behaving exactly as it did.
_UNNAMED = """WEBVTT

00:00:00.000 --> 00:00:09.000
Welcome back to Singletrack Sessions.

00:00:09.000 --> 00:00:15.000
Thanks for having me.
"""


def _episode() -> Any:
    ep = Episode(
        idx=1,
        title="Building Trails That Last",
        title_safe="building-trails",
        item=ET.Element("item"),
        transcript_urls=[],
    )
    ep.feed_hosts = ["Maya"]
    ep.detected_speaker_names = ["Maya", "Liam"]
    ep.metadata_named = ["Maya", "Liam"]
    ep.speaker_detection_ran = True
    return ep


def _roster(vtt: str) -> Tuple[dict, List[dict]]:
    plain, segments = parse_webvtt(vtt)
    out = _apply_native_speaker_roster(
        {"text": plain, "segments": segments}, config_module.Config(), _episode()
    )
    return out, list(out.get("segments") or [])


def test_the_voice_spans_become_named_placed_voices() -> None:
    """The names the file states must reach the segments, which is where ``content.speakers`` is
    derived from (``metadata_generation``, #876). Without this the roster never runs and every
    label is dropped."""
    out, segs = _roster(_NAMED)

    labelled = [s for s in segs if s.get("speaker_label")]
    assert labelled, (
        "a transcript that names every turn produced no speaker_label at all — the voice spans "
        "were parsed and then discarded, which is the 128-of-128 defect"
    )
    assert {str(s["speaker_label"]) for s in labelled} == {"Maya", "Liam"}
    assert out.get("diarization_num_speakers") == 2


def test_the_roles_come_from_the_roster_not_from_cue_order() -> None:
    """Naming is only half of it. A publisher transcript states WHO speaks, never who HOSTS, so
    the roles must be resolved as evidence — the guess-from-ordering class of bug the roster
    exists to replace."""
    _, segs = _roster(_NAMED)
    roles = {
        str(s.get("speaker_label")): s.get("speaker_role") for s in segs if s.get("speaker_label")
    }
    assert roles == {
        "Maya": "host",
        "Liam": "guest",
    }, f"expected the feed's host to be seated as host and the introduced name as guest: {roles}"


def test_a_transcript_without_voice_spans_is_untouched() -> None:
    """The gate is on the DATA, not on a provider name, so the ordinary unlabelled transcript must
    keep its existing behaviour exactly — no roster pass, no invented speaker, no diagnostics."""
    out, segs = _roster(_UNNAMED)
    assert not any(s.get("speaker_label") for s in segs)
    assert not out.get("speaker_diagnostics")
    assert [s.get("text") for s in segs] == [
        s.get("text") for s in parse_webvtt(_UNNAMED)[1]
    ], "an unlabelled transcript must pass through untouched"


def test_the_download_path_actually_calls_the_roster() -> None:
    """Pin the WIRING, not just the helper.

    Every other test in this file exercises ``_apply_native_speaker_roster`` directly — and that
    helper already existed while the defect was live, because the transcript-download branch never
    called it. So they would all have passed on the broken code. This is the assertion that fails
    if the call is removed from ``process_transcript_download``.
    """
    import inspect

    from podcast_scraper.workflow import episode_processor

    src = inspect.getsource(episode_processor.process_transcript_download)
    assert "_apply_native_speaker_roster" in src, (
        "process_transcript_download must route a speaker-labelled transcript through the roster; "
        "without it the parsed <v Speaker> spans are saved as segments nobody ever names and the "
        "episode is one undifferentiated voice (128 of 128 on the production corpus)"
    )
    assert "_save_speaker_diagnostics_file" in src, (
        "the roster's diagnostics must be persisted beside the segments, or an operator cannot "
        "explain a voice it declined to name without re-running the pipeline"
    )


def test_it_never_invents_a_voice_the_file_did_not_name() -> None:
    """#876: a heuristic may identify a voice, never author a name. Every label must be one the
    transcript actually stated."""
    _, segs = _roster(_NAMED)
    stated = {s["speaker"] for s in parse_webvtt(_NAMED)[1] if s.get("speaker")}
    for s in segs:
        label = s.get("speaker_label")
        if label:
            assert str(label) in stated, f"{label!r} is a name the transcript never said"
