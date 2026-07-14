"""The eval harness must run the pipeline we SHIP, voice gates and all.

GI reads the diarized segments to decide two things it cannot decide from text alone:

* an advertisement is never grounded — ad copy is *written* to be quotable, which makes it the
  most fluent false insight on offer;
* an insight from a voice nobody can name is not surfaceable.

``run_experiment`` never passed ``transcript_segments`` to ``build_artifact``, so both gates were
dead in every eval run — the same "works, wired to nothing" defect the gates themselves were added
to close. A head-to-head scored on that harness compares two models on a pipeline neither of them
would actually run under.

The offsets are the trap. ``char_start``/``char_end`` index the RAW screenplay, and the eval applies
a preprocessing profile before summarising. Hand GI the cleaned text with raw offsets and every
voice lookup lands on whichever speaker the shift happened to hit: measured on the real corpus,
~0-8% of offsets still resolve. A gate that mis-attributes is worse than a gate that is absent, so
misaligned segments are refused outright rather than trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.experiment.run_experiment import (  # noqa: E402
    gi_transcript_and_segments,
)

pytestmark = pytest.mark.unit

AD = "SPEAKER_04: there is more than one side to every story with the flip side podcast"
HOST = "Katie Martin: you want to figure out what financial markets are up to"


def _screenplay() -> tuple[str, List[Dict[str, Any]]]:
    """A raw screenplay and the segments whose offsets index it — as the pipeline emits them."""
    text = f"{AD}\n{HOST}\n"
    segments: List[Dict[str, Any]] = []
    cursor = 0
    for line, voice in ((AD, "commercial"), (HOST, "person")):
        segments.append(
            {
                "text": line,
                "char_start": cursor,
                "char_end": cursor + len(line),
                "speaker_label": line.split(":")[0],
                "voice_type": voice,
            }
        )
        cursor += len(line) + 1
    return text, segments


def _write(tmp_path: Path, text: str, segments: List[Dict[str, Any]] | None) -> Path:
    transcript = tmp_path / "ep01.txt"
    transcript.write_text(text, encoding="utf-8")
    if segments is not None:
        (tmp_path / "ep01.segments.json").write_text(json.dumps(segments), encoding="utf-8")
    return transcript


def _cleaned(raw: str) -> str:
    """Stand-in for a preprocessing profile: it drops the sponsor line, shifting every offset."""
    return raw.replace(AD + "\n", "")


def test_the_segments_arrive_WITH_the_text_they_index(tmp_path: Path) -> None:
    """The whole point of the paired return: GI cannot be handed one without the other."""
    raw, segments = _screenplay()
    gi_text, loaded = gi_transcript_and_segments(
        _write(tmp_path, raw, segments), raw, _cleaned(raw)
    )

    assert loaded is not None
    assert [s["voice_type"] for s in loaded] == ["commercial", "person"]
    assert gi_text == raw, (
        "GI was handed the PREPROCESSED text while the offsets index the raw screenplay — "
        "every voice lookup lands on whichever speaker the shift happened to hit"
    )
    for seg in loaded:
        assert gi_text[seg["char_start"] : seg["char_end"]] == seg["text"]


def test_an_episode_with_no_segments_reads_the_preprocessed_text_as_before(tmp_path: Path) -> None:
    """Older datasets carry no diarization. They run as before — ungated, but not broken."""
    raw, _ = _screenplay()
    cleaned = _cleaned(raw)
    gi_text, loaded = gi_transcript_and_segments(_write(tmp_path, raw, None), raw, cleaned)

    assert loaded is None
    assert gi_text == cleaned


def test_segments_that_do_not_index_the_transcript_are_REFUSED(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """THE BUG THIS EXISTS FOR.

    Preprocessing shifts the transcript; the offsets do not move with it. Trusting them would hand
    the host's words to the advertiser and vice versa — measured on the real corpus, only 0-8% of
    offsets still resolved. Refuse them, and say so.
    """
    raw, segments = _screenplay()
    shifted = "A preamble the cleaner left behind. " + raw

    with caplog.at_level("ERROR"):
        gi_text, loaded = gi_transcript_and_segments(
            _write(tmp_path, shifted, segments), shifted, _cleaned(shifted)
        )

    assert loaded is None, "misaligned offsets were trusted — quotes will name the wrong speaker"
    assert "wrong speaker" in caplog.text
    assert gi_text == _cleaned(shifted)


def test_the_grounded_insights_branch_actually_CALLS_it() -> None:
    """A loader nobody calls is the very defect this file is about.

    Both ``build_artifact`` call sites in the grounded_insights branch must forward the segments.
    The text/offset pairing is guarded above; this guards the wiring.
    """
    source = (_REPO_ROOT / "scripts" / "eval" / "experiment" / "run_experiment.py").read_text(
        encoding="utf-8"
    )
    _, _, gi_branch = source.partition('elif cfg.task == "grounded_insights":')
    gi_branch, _, _ = gi_branch.partition('elif cfg.task == "knowledge_graph"')

    assert "gi_transcript_and_segments(" in gi_branch
    calls = gi_branch.count("gil_payload = build_artifact(")
    assert calls == 2, f"expected the stub + provider call sites, found {calls}"
    assert gi_branch.count("transcript_segments=episode_segments") == calls, (
        "a build_artifact call in the eval GI branch does not forward the diarized segments — "
        "the advertisement and surfaceable gates are dead in this harness"
    )
