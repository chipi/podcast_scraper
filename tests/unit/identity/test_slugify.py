"""Unit tests for RFC-072 canonical CIL slugify (Phase 1)."""

import pytest

from podcast_scraper.identity.slugify import org_id, person_id, slugify, topic_id

pytestmark = [pytest.mark.unit]


def test_slugify_deterministic_idempotent() -> None:
    s = slugify("  Sam Altman  ")
    assert s == "sam-altman"
    assert slugify(s) == s


def test_slugify_diacritics_stripped_to_ascii() -> None:
    assert slugify("José") == "jose"
    assert slugify("Björk") == "bjork"


def test_slugify_empty_raises_with_original_in_message() -> None:
    with pytest.raises(ValueError, match="normalisation"):
        slugify("")
    with pytest.raises(ValueError) as exc:
        slugify("   ")
    assert "   " in str(exc.value)


def test_slugify_whitespace_only_raises() -> None:
    with pytest.raises(ValueError):
        slugify("\n\t  \n")


def test_slugify_all_punctuation_raises() -> None:
    with pytest.raises(ValueError):
        slugify("!!!")


def test_slugify_mixed_punctuation_and_words() -> None:
    assert slugify("  Car Loans!! ") == "car-loans"


def test_slugify_underscores_to_hyphens() -> None:
    assert slugify("foo_bar baz") == "foo-bar-baz"


def test_slugify_very_long_stable() -> None:
    raw = "word-" * 200
    out = slugify(raw)
    assert len(out) > 100
    assert out == slugify(raw)


def test_slugify_pure_non_ascii_emoji_raises() -> None:
    with pytest.raises(ValueError):
        slugify("🔥")


def test_person_org_topic_id_format() -> None:
    assert person_id("Lex Fridman") == "person:lex-fridman"
    assert org_id("OpenAI") == "org:openai"
    assert topic_id("AI Regulation") == "topic:ai-regulation"


def test_person_id_empty_raises() -> None:
    with pytest.raises(ValueError):
        person_id("  ")
