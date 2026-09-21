"""A name the PUBLISHER stated must survive ``relabel_only`` — including the second relabel.

THE DEFECT. ``relabel_only`` exists to re-resolve speaker names from evidence, so it deliberately
throws every stored identity away first: ``_anon(_identity(s))`` remaps each distinct
``speaker_label`` to a fresh ``SPEAKER_NN`` and hands the roster an anonymous clustering. That is
exactly right for names WE inferred — the whole point is to re-derive them — and exactly wrong for
names the SOURCE stated. A publisher transcript that writes ``<v Maya>`` on every turn is not a
guess to be re-derived; it is a fact about the file. Anonymising it meant a relabel of such an
episode fell back to the feed's host list and could publish the feed's name on a voice the
transcript had already named — the ac1751b4 defect, arriving one stage later.

THE CARRIER. ``_roster_native_segments`` writes ``stated_speaker`` onto the durable segments
beside ``speaker_label``: OUR answer and the SOURCE's answer, stored separately, because only one
of them is ours to revise. ``_relabel_existing_transcript`` reads it back into
``stated_voice_names``.

WHY THE SECOND RELABEL IS ITS OWN TEST. The roster is handed a stripped ``{start, end, text}``
view, so what it returns carries no ``stated_speaker``. Writing that straight back would make the
first relabel look correct and the second silently fall back to the feed — a bug that only appears
on the reprocess-of-a-reprocess nobody runs by hand.

EVERY ASSERTION HERE HAS THE FEED DISAGREEING WITH THE FILE. Agreement cannot distinguish
"respected the publisher" from "re-derived and got lucky" — that is precisely how the test written
alongside be1ed96d passed while the code was broken.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper import config as config_module
from podcast_scraper.models import TranscriptionJob
from podcast_scraper.models.entities import Episode
from podcast_scraper.transcript_formats import parse_webvtt
from podcast_scraper.workflow import episode_processor as epx

pytestmark = [pytest.mark.unit]


#: The feed's hosts name a DIFFERENT person from either voice in the file.
_FEED_DESC = "Each week, journalists Kevin Roose and Casey Newton explore the world of tech."

_NAMED_VTT = """WEBVTT

00:00:00.000 --> 00:00:09.000
<v Maya>Welcome back to Singletrack Sessions. I'm joined by Liam, who builds trail for the \
Cascadia Alliance.</v>

00:00:09.000 --> 00:00:15.000
<v Liam>Thanks, Maya. Glad to be here — this is something I think about a lot.</v>

00:00:15.000 --> 00:00:24.000
<v Maya>Let's start there. What is the most underrated piece of trail building?</v>

00:00:24.000 --> 00:00:36.000
<v Liam>Drainage. If water leaves on its own the trail lasts a decade.</v>
"""


def _cfg(**kw: Any) -> config_module.Config:
    return config_module.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
        speaker_resolution_llm=False,  # deterministic + airgapped (no LLM in tests/CI)
        **kw,
    )


def _rostered_segments() -> List[Dict[str, Any]]:
    """The segments a transcript DOWNLOAD would have stored for ``_NAMED_VTT``."""
    plain, segments = parse_webvtt(_NAMED_VTT)
    out = epx._roster_native_segments(
        {"text": plain, "segments": segments},
        _cfg(),
        idx=1,
        detected_speaker_names=["Maya", "Liam"],
        metadata_named=["Maya", "Liam"],
        feed_hosts=["Maya"],
        detection_ran=True,
    )
    return list(out.get("segments") or [])


# ------------------------------------------------------------------------------------------
# The carrier itself.
# ------------------------------------------------------------------------------------------


def test_the_publishers_own_label_is_stored_beside_our_answer() -> None:
    rows = _rostered_segments()
    assert rows, "the roster returned no segments at all"
    assert {r.get("stated_speaker") for r in rows if r.get("stated_speaker")} == {
        "Maya",
        "Liam",
    }, "the publisher's own labels were not persisted, so a relabel has nothing to read back"


def test_our_answer_is_still_written_separately() -> None:
    """``stated_speaker`` ADDS a field; it must not displace the resolved label the rest of the
    pipeline reads (``adfree``, ``metadata_generation``, GI quote attribution)."""
    rows = _rostered_segments()
    assert {r.get("speaker_label") for r in rows if r.get("speaker_label")} == {"Maya", "Liam"}


def test_a_non_person_label_is_not_stored_as_a_stated_name() -> None:
    """A sponsor read separates a turn without naming a person. Storing ``Ad`` here would hand it
    to the next relabel at TOP precedence — the one place a bad label does the most damage."""
    vtt = _NAMED_VTT + (
        "\n00:00:36.000 --> 00:00:44.000\n<v Ad>This episode is brought to you by a sponsor.</v>\n"
    )
    plain, segments = parse_webvtt(vtt)
    out = epx._roster_native_segments(
        {"text": plain, "segments": segments},
        _cfg(),
        idx=1,
        detected_speaker_names=["Maya", "Liam"],
        metadata_named=["Maya", "Liam"],
        feed_hosts=["Maya"],
        detection_ran=True,
    )
    stated = {
        r.get("stated_speaker") for r in (out.get("segments") or []) if r.get("stated_speaker")
    }
    assert stated == {"Maya", "Liam"}, f"a production credit was stored as a stated name: {stated}"


# ------------------------------------------------------------------------------------------
# Through relabel_only, with the feed disagreeing.
# ------------------------------------------------------------------------------------------


def _write_corpus(base: Path, run_tag: str, segs: List[Dict[str, Any]]) -> Tuple[Path, str]:
    """Lay down a finished-corpus run dir carrying the segments a download produced."""
    old_run = base / f"run_{run_tag}"
    tdir = old_run / "transcripts"
    mdir = old_run / "metadata"
    tdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    stem = f"0001 - Ep_{run_tag}"
    (tdir / f"{stem}.txt").write_text(
        "\n".join(f"{s.get('speaker_label')}: {s.get('text', '')}" for s in segs),
        encoding="utf-8",
    )
    (tdir / f"{stem}.segments.json").write_text(json.dumps(segs), encoding="utf-8")
    (mdir / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {
                    "title": "Hard Fork",
                    "description": _FEED_DESC,
                    "authors": ["The New York Times"],
                },
                "content": {"transcript_source": "direct_download"},
            }
        ),
        encoding="utf-8",
    )
    return old_run, stem


def _job(episode: Optional[Any] = None) -> TranscriptionJob:
    return TranscriptionJob(
        idx=1,
        ep_title="Building Trails That Last",
        ep_title_safe="building-trails",
        temp_media="",
        detected_speaker_names=None,
        metadata_named=None,
        episode=episode,
    )


def _relabel(base: Path, run_tag: str, new_tag: str) -> Tuple[bool, List[Dict[str, Any]], str]:
    new_run = base / f"run_{new_tag}"
    new_run.mkdir(parents=True)
    episode = Episode(
        idx=1,
        title="Building Trails That Last",
        title_safe="building-trails",
        item=ET.Element("item"),
        transcript_urls=[],
    )
    ok, _rel, _n = epx._relabel_existing_transcript(
        _job(episode), _cfg(pipeline_stage="relabel_only"), run_tag, str(new_run), None, None
    )
    stem = f"0001 - Ep_{run_tag}"
    tdir = base / f"run_{run_tag}" / "transcripts"
    rows = json.loads((tdir / f"{stem}.segments.json").read_text(encoding="utf-8"))
    text = (tdir / f"{stem}.txt").read_text(encoding="utf-8")
    return ok, rows, text


def test_a_relabel_keeps_the_names_the_publisher_stated(tmp_path: Path) -> None:
    """THE regression test. The feed's hosts are Kevin Roose and Casey Newton; the file says Maya
    and Liam. Re-resolving from the feed would publish a name the source contradicts (#876)."""
    base = tmp_path / "feed"
    _write_corpus(base, "20260101-000000_t", _rostered_segments())

    ok, rows, text = _relabel(base, "20260101-000000_t", "20260102-000000_t")

    assert ok is True, "the relabel did not run at all, so it proves nothing"
    labels = {r.get("speaker_label") for r in rows if r.get("speaker_label")}
    assert labels == {"Maya", "Liam"}, f"the relabel overrode the publisher's names: {labels}"
    assert "Kevin Roose" not in text and "Casey Newton" not in text, text[:400]


def test_the_stated_names_are_rewritten_so_the_next_relabel_still_has_them(
    tmp_path: Path,
) -> None:
    """The roster returns a stripped view. Without re-attaching, this field is gone after one
    relabel and the SECOND falls back to the feed."""
    base = tmp_path / "feed"
    _write_corpus(base, "20260101-000000_t", _rostered_segments())

    _ok, rows, _text = _relabel(base, "20260101-000000_t", "20260102-000000_t")

    assert {r.get("stated_speaker") for r in rows if r.get("stated_speaker")} == {
        "Maya",
        "Liam",
    }, "the relabel dropped the publisher's labels from the stored segments"


def test_a_second_relabel_still_keeps_them(tmp_path: Path) -> None:
    """Reprocess-of-a-reprocess. This is the run that would have silently regressed."""
    base = tmp_path / "feed"
    _write_corpus(base, "20260101-000000_t", _rostered_segments())

    _relabel(base, "20260101-000000_t", "20260102-000000_t")
    ok, rows, text = _relabel(base, "20260101-000000_t", "20260103-000000_t")

    assert ok is True
    labels = {r.get("speaker_label") for r in rows if r.get("speaker_label")}
    assert labels == {"Maya", "Liam"}, f"the second relabel fell back to the feed: {labels}"
    assert "Kevin Roose" not in text and "Casey Newton" not in text, text[:400]


def test_an_ordinary_inferred_name_is_still_re_derived(tmp_path: Path) -> None:
    """The other direction, so this is not a blanket freeze. A corpus with NO ``stated_speaker``
    — an ASR + diarize episode — must still have v2's resolved name anonymised and re-resolved,
    which is the entire purpose of relabel_only."""
    base = tmp_path / "feed"
    segs = [
        {
            "start": 0.0,
            "end": 60.0,
            "speaker": None,
            "speaker_label": "Amy Lawrence",
            "text": "Welcome back. I'm Kevin Russo, tech columnist, here with Casey.",
        },
        {
            "start": 60.0,
            "end": 120.0,
            "speaker": None,
            "speaker_label": "SPEAKER_01",
            "text": "Thanks Kevin. Let's get into the agents story.",
        },
    ]
    _write_corpus(base, "20260101-000000_t", segs)

    ok, _rows, text = _relabel(base, "20260101-000000_t", "20260102-000000_t")

    assert ok is True
    assert "Amy Lawrence" not in text, "v2's inferred name was frozen; relabel must re-derive it"


# ------------------------------------------------------------------------------------------
# The retranscript_only hand-off — the post-deploy repair path for #2130.
# ------------------------------------------------------------------------------------------


def _retranscript_segments() -> List[Dict[str, Any]]:
    """EXACTLY what ``_retranscript_existing_transcript`` writes: the parsed cues, verbatim.

    ``{start, end, text, speaker}`` and nothing else — no ``stated_speaker`` (that key is written
    by the DOWNLOAD path, which never ran for these episodes) and no ``speaker_label`` (nothing
    has resolved anything yet).
    """
    _plain, segments = parse_webvtt(_NAMED_VTT)
    return [dict(s) for s in segments]


def test_the_retranscript_handoff_keeps_the_publishers_names(tmp_path: Path) -> None:
    """THE post-deploy regression, found by simulating the stage rather than reading it.

    ``retranscript_only`` repairs the 129 production episodes whose voice spans were never stored
    (measured on the 2,257-episode snapshot: those 129 have no ``.segments.json`` at all, and all
    129 do record a transcript URL, so this is the stage that can repair them). It re-fetches,
    re-parses, writes the cues, and hands off to the relabel below.

    Before this was fixed the hand-off discarded every stated name and published ``SPEAKER_00`` /
    ``SPEAKER_01`` — the repair ran, logged "rewrote ... with 2 speaker(s); relabelling", and named
    nobody, which is the entire thing #2130 exists to fix.
    """
    base = tmp_path / "feed"
    _write_corpus(base, "20260101-000000_t", _retranscript_segments())

    ok, rows, text = _relabel(base, "20260101-000000_t", "20260102-000000_t")

    assert ok is True
    labels = {r.get("speaker_label") for r in rows if r.get("speaker_label")}
    assert labels == {"Maya", "Liam"}, (
        f"the retranscript hand-off published {sorted(labels)} — the publisher stated Maya and "
        "Liam, and a bare SPEAKER_NN means the repair named nobody"
    )
    assert "Kevin Roose" not in text and "Casey Newton" not in text, text[:300]


def test_a_bare_cue_label_is_still_only_a_cluster_id(tmp_path: Path) -> None:
    """The guard on the new fallback. Providers that tag turns positionally (``Speaker 1``, SRT's
    ``Speaker N``) state SEPARATION, not identity — publishing those would mint a KG person called
    "Speaker 1"."""
    segs = _retranscript_segments()
    for i, s in enumerate(segs):
        s["speaker"] = f"Speaker {i % 2 + 1}"
    base = tmp_path / "feed"
    _write_corpus(base, "20260101-000000_t", segs)

    _ok, rows, _text = _relabel(base, "20260101-000000_t", "20260102-000000_t")

    labels = {r.get("speaker_label") for r in rows if r.get("speaker_label")}
    assert all(lbl.startswith("SPEAKER_") for lbl in labels), labels


def test_a_previous_runs_resolved_label_is_still_re_derived(tmp_path: Path) -> None:
    """The line the fallback must not cross. ``speaker`` is what the SOURCE said; ``speaker_label``
    is OUR answer from a previous run, and re-deriving it is what relabel is FOR. Treating the
    latter as stated would freeze v2's names forever — the exact bug this stage was built to undo.
    """
    segs = [
        {
            "start": 0.0,
            "end": 60.0,
            "speaker": None,
            "speaker_label": "Amy Lawrence",
            "text": "Welcome back. I'm Kevin Russo, tech columnist, here with Casey.",
        },
        {
            "start": 60.0,
            "end": 120.0,
            "speaker": None,
            "speaker_label": "SPEAKER_01",
            "text": "Thanks Kevin. Let's get into the agents story.",
        },
    ]
    base = tmp_path / "feed"
    _write_corpus(base, "20260101-000000_t", segs)

    _ok, _rows, text = _relabel(base, "20260101-000000_t", "20260102-000000_t")

    assert "Amy Lawrence" not in text, "a previous run's resolved name was frozen as if stated"
