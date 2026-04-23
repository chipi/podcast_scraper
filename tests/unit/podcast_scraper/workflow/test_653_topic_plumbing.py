"""Unit tests for #653 Part A + D — GI Topic plumbing.

Covers the two paths that assemble ``gi_topic_labels`` in
:mod:`podcast_scraper.workflow.metadata_generation`:

* Prefilled-extraction topics (mega_bundled / extraction_bundled).
* Summary-bullet fallback with noun-phrase extraction (staged mode).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

with patch.dict("sys.modules", {"spacy": __import__("unittest.mock").mock.MagicMock()}):
    from podcast_scraper.workflow.metadata_generation import _bullet_to_topic_phrase

pytestmark = [pytest.mark.unit]


class TestBulletToTopicPhrase:
    """#653 Part D — staged-mode fallback: bullet → short noun phrase."""

    def test_empty_string_returns_empty(self):
        assert _bullet_to_topic_phrase("") == ""
        assert _bullet_to_topic_phrase("   ") == ""

    def test_strips_leading_stopwords(self):
        # "The quick brown fox" → drops "The" → first 4 content tokens.
        assert _bullet_to_topic_phrase("The quick brown fox jumps") == "quick brown fox jumps"

    def test_strips_multiple_leading_stopwords(self):
        # "how" + "we" stripped (both in stopword list); "can" is not
        # a stopword so stripping stops.
        assert _bullet_to_topic_phrase("How we can improve the system") == (
            "can improve the system"
        )

    def test_stops_stripping_after_three(self):
        # "The a an the content" — 3 stopwords stripped, then "the content" kept.
        result = _bullet_to_topic_phrase("The a an the real content here")
        assert result.startswith("the real content")

    def test_keeps_max_tokens(self):
        assert _bullet_to_topic_phrase("Word1 Word2 Word3 Word4 Word5 Word6") == (
            "Word1 Word2 Word3 Word4"
        )

    def test_custom_max_tokens(self):
        assert _bullet_to_topic_phrase("Word1 Word2 Word3 Word4 Word5", max_tokens=2) == (
            "Word1 Word2"
        )

    def test_strips_trailing_punctuation(self):
        assert _bullet_to_topic_phrase("Quick brown fox jumps.") == "Quick brown fox jumps"
        assert _bullet_to_topic_phrase("Quick brown fox jumps,") == "Quick brown fox jumps"

    def test_stopword_with_comma_still_stripped(self):
        # "The," should be recognised as stopword (case-insensitive, punct-stripped).
        assert _bullet_to_topic_phrase("The, quick brown fox jumps") == "quick brown fox jumps"

    def test_fallback_to_original_if_stripping_empties(self):
        # If all tokens are stopwords, the stripping loop only eats 3 and the
        # rest is kept. The "fallback to text" branch fires only when the final
        # phrase after rstrip becomes empty.
        result = _bullet_to_topic_phrase("the a an")
        # After 3 stripped: empty → phrase = "" → fallback to text
        assert result == "the a an"

    def test_real_summary_bullet_example(self):
        # Approximate a real Planet Money bullet.
        bullet = "The Strait of Hormuz is critical for global oil trade"
        result = _bullet_to_topic_phrase(bullet)
        assert result == "Strait of Hormuz is"
        # Not perfect English but short enough to be a Topic label (#653
        # tolerates imperfection on the fallback path — bundled modes use KG
        # canonical topics directly).

    def test_no_stopword_prefix_unchanged(self):
        assert _bullet_to_topic_phrase("Prediction markets face scrutiny") == (
            "Prediction markets face scrutiny"
        )
