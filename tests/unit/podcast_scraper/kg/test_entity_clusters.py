"""Entity names are compared with diacritics folded (#2056)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestDiacriticsAreFoldedBeforeComparing:
    """``Björk`` and ``Bjork`` are one person (#2056).

    Transcripts and feed metadata disagree about diacritics constantly — ASR emits unaccented
    ASCII, the show notes carry the real spelling. Multi-token names already survived this by
    accident, because one differing character barely moves the similarity ratio:

        'Jürgen Schmidhuber' == 'Jurgen Schmidhuber'   -> already True (fuzzy)

    A SINGLE-token name does not, because `_is_acronymish` refuses to fuzzy-match short single
    tokens at all — the UPS/USPS guard. So the mononym case fell through the one gap where the
    ratio test cannot rescue it.

    Folding also makes the multi-token cases match EXACTLY rather than by ratio, which is a
    precision gain, not just a recall one: an exact hit short-circuits before any threshold is
    consulted.
    """

    def test_folding_is_exact_not_fuzzy(self) -> None:
        from podcast_scraper.kg.filters import _clean_entity_name

        assert _clean_entity_name("François Chollet") == _clean_entity_name("Francois Chollet")
