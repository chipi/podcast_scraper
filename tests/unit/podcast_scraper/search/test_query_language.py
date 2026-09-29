"""The query-language signal, and the decision NOT to act on it in v1 (S2.9 / D-14).

D-14 asked for the dense-leg switch to be "measured either way". There is no query corpus to
measure against, and the asymmetry decides the default without one:

- wrongly dropping the dense leg on an ENGLISH query is a regression for the 678 English
  episodes that are the corpus today — semantic search is what those users came for;
- wrongly keeping it on a SPANISH query is dilution — the keyword leg still reaches the Spanish
  through `segments_nonen`, and the dense leg just adds English results alongside.

So the cheap mistake is tolerated and the expensive one is not.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.query_language import assess_query_language

pytestmark = pytest.mark.unit


class TestItNeverChangesRetrievalInV1:
    @pytest.mark.parametrize(
        "query",
        ["que es la inflacion", "inflación", "what is inflation", "trail building", "", "   "],
    )
    def test_drop_dense_leg_is_always_false(self, query: str) -> None:
        """The decision, pinned. A future change is one line with a test behind it rather than a
        new concept introduced under time pressure."""
        assert assess_query_language(query).drop_dense_leg is False


class TestEnglishIsProtectedFirst:
    @pytest.mark.parametrize(
        "query",
        [
            "what is the inflation rate",
            "how does drainage work",
            "who was on the show about trails",
            "trails and drainage",
        ],
    )
    def test_an_english_function_word_suppresses_the_signal(self, query: str) -> None:
        """The half that matters most. A false "non-English" on an English query is the
        expensive mistake, so any English marker ends the assessment."""
        got = assess_query_language(query)
        assert got.looks_non_english is False
        assert "English function word" in got.evidence

    def test_english_wins_even_when_a_spanish_marker_is_also_present(self) -> None:
        """ "de" is Spanish AND appears in English names and phrases. English evidence is checked
        FIRST for exactly that reason."""
        got = assess_query_language("the history de facto")
        assert got.looks_non_english is False


class TestWhatItCanDetect:
    def test_spanish_function_words(self) -> None:
        got = assess_query_language("que es la inflacion")
        assert got.looks_non_english is True
        assert got.guess == "es"
        assert "function words" in got.evidence

    def test_diacritics_alone(self) -> None:
        """Weak evidence, but evidence: script detection gives none for tier 1 because every
        tier-1 language is Latin (D-14)."""
        got = assess_query_language("inflación")
        assert got.looks_non_english is True
        assert got.guess is None, "diacritics say non-English, not which language"
        assert "diacritics" in got.evidence

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("der und das", "de"),
            ("le les des", "fr"),
            ("il che una", "it"),
        ],
    )
    def test_other_tier_one_languages(self, query: str, expected: str) -> None:
        assert assess_query_language(query).guess == expected


class TestWhatItHonestlyCannot:
    @pytest.mark.parametrize("query", ["senderos", "inflacion", "drenaje"])
    def test_a_bare_non_english_noun_is_reported_as_AMBIGUOUS(self, query: str) -> None:
        """Not a guess. A single Spanish word with no diacritics and no function words is
        genuinely indistinguishable from an English proper noun or a loanword, and short
        queries are most queries — which is the whole reason the dense leg is not dropped.
        """
        got = assess_query_language(query)
        assert got.looks_non_english is False
        assert got.evidence == "no discriminating signal"

    def test_an_empty_query_says_so(self) -> None:
        assert assess_query_language("").evidence == "empty query"


class TestOverlappingMarkerSets:
    """The marker sets overlap by design, so the guess must go to the BEST match.

    "una" is Spanish and Italian; "de"/"que" are Spanish and Portuguese. Iterating in
    declaration order handed "il che una" to `es` on one shared token while `it` matched three.
    The guess is recorded for the measurement D-14 asked for, so a systematically wrong guess
    would poison exactly the data it exists to collect.
    """

    def test_the_language_with_the_most_markers_wins(self) -> None:
        got = assess_query_language("il che una")
        assert got.guess == "it", "3 Italian markers beat 1 shared with Spanish"

    def test_a_single_shared_token_still_produces_a_deterministic_guess(self) -> None:
        """One hit in several languages is a real tie. It must resolve the same way every run,
        or the recorded signal is noise."""
        first = assess_query_language("una").guess
        assert first is not None
        assert all(assess_query_language("una").guess == first for _ in range(5))

    def test_a_clear_spanish_query_is_still_spanish(self) -> None:
        got = assess_query_language("los senderos por la montana")
        assert got.guess == "es"
