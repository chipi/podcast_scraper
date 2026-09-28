"""The transcript resolver: purpose selects a coordinate space (#2170 / S2.1a).

Pins the contract the fifteen readers are being routed onto, and the one input where the
new resolver deliberately answers differently from the function it replaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.workflow.transcript_resolution import (
    adfree_transcript_relpath,
    load_processing_transcript,
    load_transcript,
    resolve_segments_path,
    resolve_text_path,
    segments_relpath_candidates,
    text_relpath_candidates,
    TranscriptPurpose,
)

pytestmark = pytest.mark.unit

_REL = "transcripts/01 - ep.txt"


def _write(root: Path, rel: str, payload: object) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload), encoding="utf-8")


class TestCandidateOrder:
    """Pure ordering — the part that must NOT consult the disk, so it can be reasoned about."""

    def test_analysis_prefers_adfree(self) -> None:
        assert text_relpath_candidates(_REL, purpose=TranscriptPurpose.ANALYSIS) == [
            "transcripts/01 - ep.adfree.txt",
            "transcripts/01 - ep.txt",
        ]

    def test_timeline_prefers_raw(self) -> None:
        assert text_relpath_candidates(_REL, purpose=TranscriptPurpose.TIMELINE) == [
            "transcripts/01 - ep.txt",
            "transcripts/01 - ep.adfree.txt",
        ]

    def test_the_two_purposes_are_exact_opposites(self) -> None:
        """If these ever agree, the reason the resolver takes a purpose has evaporated."""
        analysis = text_relpath_candidates(_REL, purpose=TranscriptPurpose.ANALYSIS)
        timeline = text_relpath_candidates(_REL, purpose=TranscriptPurpose.TIMELINE)
        assert analysis == list(reversed(timeline))
        assert analysis[0] != timeline[0]

    def test_cleaned_is_a_middle_candidate_never_a_first_choice(self) -> None:
        """The recurrent-host scan wants it; nobody wants it ahead of a real body."""
        got = text_relpath_candidates(
            _REL, purpose=TranscriptPurpose.ANALYSIS, include_cleaned=True
        )
        assert got == [
            "transcripts/01 - ep.adfree.txt",
            "transcripts/01 - ep.cleaned.txt",
            "transcripts/01 - ep.txt",
        ]

    def test_segments_candidates_follow_the_body_order(self) -> None:
        assert segments_relpath_candidates(_REL, purpose=TranscriptPurpose.ANALYSIS) == [
            "transcripts/01 - ep.adfree.segments.json",
            "transcripts/01 - ep.segments.json",
        ]
        assert segments_relpath_candidates(_REL, purpose=TranscriptPurpose.TIMELINE) == [
            "transcripts/01 - ep.segments.json",
            "transcripts/01 - ep.adfree.segments.json",
        ]

    @pytest.mark.parametrize("rel", ["", "   ", None])
    def test_no_path_yields_no_candidates(self, rel: object) -> None:
        # `None` is not in the signature, but metadata can carry a null transcript path and
        # the callers pass it straight through, so it has to answer rather than raise.
        got = text_relpath_candidates(
            rel,  # type: ignore[arg-type]
            purpose=TranscriptPurpose.ANALYSIS,
        )
        assert got == []

    def test_a_derived_reference_resolves_like_its_canonical_base(self) -> None:
        """GI's ``transcript_ref`` points at whichever body it read, and comes back here.

        DECLARED DIFFERENCE from the old ``load_processing_transcript``: given
        ``…adfree.txt`` it appended a second suffix, looked for ``…adfree.adfree.txt``,
        missed, and fell through to the raw branch — returning the ad-free file while
        reporting ``is_adfree=False``. No caller passes such a path today (metadata always
        stores the plain ``.txt``), so this changes no live behaviour; it defines an input
        that previously produced a contradiction.
        """
        for given in ("transcripts/01 - ep.adfree.txt", "transcripts/01 - ep.cleaned.txt"):
            assert text_relpath_candidates(given, purpose=TranscriptPurpose.ANALYSIS) == [
                "transcripts/01 - ep.adfree.txt",
                "transcripts/01 - ep.txt",
            ]

    def test_backslashes_are_normalized(self) -> None:
        assert text_relpath_candidates(
            "transcripts\\01 - ep.txt", purpose=TranscriptPurpose.TIMELINE
        ) == ["transcripts/01 - ep.txt", "transcripts/01 - ep.adfree.txt"]


class TestResolutionAgainstDisk:
    def test_both_present_the_purposes_diverge(self, tmp_path: Path) -> None:
        """The case neither fixture corpus contains, and the whole point of the module."""
        _write(tmp_path, _REL, "raw body")
        _write(tmp_path, "transcripts/01 - ep.adfree.txt", "adfree body")
        _write(tmp_path, "transcripts/01 - ep.segments.json", [{"id": 0}])
        _write(tmp_path, "transcripts/01 - ep.adfree.segments.json", [{"char_start": 0}])

        assert (
            resolve_text_path(tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS)
            == tmp_path / "transcripts/01 - ep.adfree.txt"
        )
        assert (
            resolve_text_path(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE) == tmp_path / _REL
        )
        assert (
            resolve_segments_path(tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS)
            == tmp_path / "transcripts/01 - ep.adfree.segments.json"
        )
        assert (
            resolve_segments_path(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE)
            == tmp_path / "transcripts/01 - ep.segments.json"
        )

    def test_analysis_falls_back_to_raw_on_a_pre_974_corpus(self, tmp_path: Path) -> None:
        _write(tmp_path, _REL, "raw body")
        assert (
            resolve_text_path(tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS) == tmp_path / _REL
        )

    def test_timeline_falls_back_to_adfree_when_the_raw_body_is_gone(self, tmp_path: Path) -> None:
        """A real fixture state: viewer-validation-corpus ships no raw segment sidecars."""
        _write(tmp_path, "transcripts/01 - ep.adfree.txt", "adfree body")
        assert (
            resolve_text_path(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE)
            == tmp_path / "transcripts/01 - ep.adfree.txt"
        )

    def test_nothing_on_disk_resolves_to_none(self, tmp_path: Path) -> None:
        assert resolve_text_path(tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS) is None
        assert resolve_segments_path(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE) is None


class TestLoadTranscript:
    def test_segments_always_come_from_the_body_that_was_loaded(self, tmp_path: Path) -> None:
        """Mixing a body with the other variant's sidecar is the displacement bug again.

        Both sidecars exist here with telltale contents, so a cross-pairing is visible
        rather than merely possible.
        """
        _write(tmp_path, _REL, "raw body")
        _write(tmp_path, "transcripts/01 - ep.adfree.txt", "adfree body")
        _write(tmp_path, "transcripts/01 - ep.segments.json", [{"from": "raw"}])
        _write(tmp_path, "transcripts/01 - ep.adfree.segments.json", [{"from": "adfree"}])
        _write(tmp_path, "transcripts/01 - ep.adfree.admap.json", {"excised": [[0, 9]]})

        analysis = load_transcript(tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS)
        assert analysis.text == "adfree body"
        assert analysis.segments == [{"from": "adfree"}]
        assert analysis.transcript_ref == "transcripts/01 - ep.adfree.txt"
        assert analysis.is_adfree is True
        assert analysis.ad_map == {"excised": [[0, 9]]}

        timeline = load_transcript(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE)
        assert timeline.text == "raw body"
        assert timeline.segments == [{"from": "raw"}]
        assert timeline.transcript_ref == _REL
        assert timeline.is_adfree is False

    def test_the_ad_map_is_only_offered_alongside_the_ad_free_body(self, tmp_path: Path) -> None:
        """An ad-map describes excisions from the raw body. Handing it to a caller that
        loaded the raw body invites it to excise twice."""
        _write(tmp_path, _REL, "raw body")
        _write(tmp_path, "transcripts/01 - ep.adfree.admap.json", {"excised": [[0, 9]]})
        loaded = load_transcript(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE)
        assert loaded.is_adfree is False
        assert loaded.ad_map is None

    def test_a_wrapped_sidecar_is_not_accepted(self, tmp_path: Path) -> None:
        """``{"segments": [...]}`` yields None, exactly as before the refactor.

        ``gi/repair`` unwraps that shape; this loader never did. Widening it here would
        give GI and KG segments where they used to get None — silently.
        """
        _write(tmp_path, _REL, "raw body")
        _write(tmp_path, "transcripts/01 - ep.segments.json", {"segments": [{"id": 0}]})
        assert load_transcript(tmp_path, _REL, purpose=TranscriptPurpose.TIMELINE).segments is None

    def test_a_missing_transcript_is_empty_and_not_ad_free(self, tmp_path: Path) -> None:
        """Not ad-free, so a consumer excises for itself rather than trusting the emptiness."""
        loaded = load_transcript(tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS)
        assert loaded.text == ""
        assert loaded.segments is None
        assert loaded.transcript_ref == _REL
        assert loaded.is_adfree is False
        assert loaded.ad_map is None

    def test_load_processing_transcript_is_the_analysis_spelling(self, tmp_path: Path) -> None:
        """The two live callers (GI, KG) keep their name and their answer."""
        _write(tmp_path, _REL, "raw body")
        _write(tmp_path, "transcripts/01 - ep.adfree.txt", "adfree body")
        assert load_processing_transcript(str(tmp_path), _REL) == load_transcript(
            tmp_path, _REL, purpose=TranscriptPurpose.ANALYSIS
        )


def test_adfree_relpath_helper_is_unchanged() -> None:
    """Re-exported from here now; its four callers must see identical output."""
    assert adfree_transcript_relpath("transcripts/01 - ep.txt") == "transcripts/01 - ep.adfree.txt"
    assert adfree_transcript_relpath("transcripts/ep") == "transcripts/ep.adfree.txt"
