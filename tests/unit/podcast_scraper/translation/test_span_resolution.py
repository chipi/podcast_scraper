"""Span → unit resolution and the provenance check (S2.5 / RFC-124 §5.4 C-5).

These two functions are what put provenance on every claim (S2.11). If they answer wrongly a
claim carries the WRONG unit ids, which is worse than carrying none: an audit would trace a
quote to text that never produced it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.translation.artifacts import resolve_units_for_span, verify_span_excerpt

pytestmark = pytest.mark.unit


def _seg(unit_id: str, start: int, end: int, sent_id: str = "s1") -> Dict[str, Any]:
    return {"unit_id": unit_id, "sent_id": sent_id, "char_start": start, "char_end": end}


#: "Maya: One. Two.\nLiam: Three."  — spans chosen so a label prefix sits between two cues.
SEGMENTS: List[Dict[str, Any]] = [
    _seg("t0000.u01", 6, 10, "t0000.s01"),
    _seg("t0000.u01", 11, 15, "t0000.s02"),
    _seg("t0001.u01", 22, 28, "t0001.s01"),
]


class TestOverlapNotContainment:
    def test_a_span_inside_one_cue_resolves_to_its_unit(self) -> None:
        assert resolve_units_for_span(SEGMENTS, 6, 10) == ["t0000.u01"]

    def test_a_span_touching_a_label_prefix_still_resolves(self) -> None:
        """The case containment gets wrong. A quote span routinely starts on `Label: `, which
        belongs to no segment — under containment it would resolve to nothing and the claim
        would silently carry no provenance at all."""
        assert resolve_units_for_span(SEGMENTS, 0, 10) == ["t0000.u01"]

    def test_a_span_crossing_a_turn_boundary_returns_BOTH_units(self) -> None:
        """The honest answer. A quote that spans two speakers was produced by two units, and
        recording only one would attribute half of it to the wrong text."""
        assert resolve_units_for_span(SEGMENTS, 11, 28) == ["t0000.u01", "t0001.u01"]

    def test_inter_turn_whitespace_alone_resolves_to_nothing(self) -> None:
        """Between cue 2 (ends 15) and cue 3 (starts 22) there is only the newline and the next
        label. A span entirely inside that touches no unit, and saying so is correct."""
        assert resolve_units_for_span(SEGMENTS, 16, 21) == []

    def test_units_are_deduplicated_and_kept_in_document_order(self) -> None:
        """Two cues of the same unit must not report it twice, and order has to be stable or a
        provenance block would churn between runs for no reason."""
        assert resolve_units_for_span(SEGMENTS, 6, 15) == ["t0000.u01"]

    def test_an_empty_or_inverted_span_resolves_to_nothing(self) -> None:
        assert resolve_units_for_span(SEGMENTS, 10, 10) == []
        assert resolve_units_for_span(SEGMENTS, 20, 5) == []

    def test_cues_without_a_unit_id_are_ignored_not_guessed(self) -> None:
        """A source-language segment has no `unit_id`. Resolving one anyway would invent
        provenance for text no translation produced."""
        segs = [{"char_start": 0, "char_end": 100, "text": "x"}]
        assert resolve_units_for_span(segs, 0, 50) == []

    def test_the_boundary_is_half_open(self) -> None:
        """A span ending exactly where a cue starts does not touch it — the same half-open
        convention every char range in this codebase uses."""
        assert resolve_units_for_span(SEGMENTS, 0, 6) == []
        assert resolve_units_for_span(SEGMENTS, 0, 7) == ["t0000.u01"]


class TestTheProvenanceCheck:
    TEXT = "Maya: One. Two.\nLiam: Three.\n"

    def test_a_matching_excerpt_verifies(self) -> None:
        assert verify_span_excerpt(self.TEXT, 6, 10, "One.") is True

    def test_a_RE_TRANSLATION_is_caught_where_a_file_hash_would_not_be(self) -> None:
        """The failure this exists for. After a re-translation the offsets still look plausible
        and the file still exists — only the text at them has changed."""
        assert verify_span_excerpt(self.TEXT, 6, 10, "Uno.") is False

    def test_whitespace_differences_do_not_fail_it(self) -> None:
        """The renderer strips each cue, so an excerpt captured with padding is still the same
        text and must not read as a mismatch."""
        assert verify_span_excerpt(self.TEXT, 6, 10, "  One. ") is True

    def test_a_span_past_the_end_of_the_text_fails(self) -> None:
        """A shorter text than the offsets expect is exactly what a withheld or re-rendered
        English body looks like."""
        assert verify_span_excerpt(self.TEXT, 6, 9999, "One.") is False

    def test_an_empty_excerpt_never_verifies_real_text(self) -> None:
        assert verify_span_excerpt(self.TEXT, 6, 10, "") is False
