"""Unit tests for cleaning_v4 preprocessing functions.

Tests the new preprocessing functions added for cleaning_v4 profile:
- is_junk_line()
- strip_episode_header()
- anonymize_speakers()
- cleaning_v4 profile integration
"""

import pytest

from podcast_scraper.preprocessing import core
from podcast_scraper.preprocessing.profiles import apply_profile


@pytest.mark.unit
def test_is_junk_line_punctuation():
    """Test is_junk_line detects punctuation-heavy lines."""
    assert core.is_junk_line("////") is True
    assert core.is_junk_line("=-=-=-=-=-") is True
    assert core.is_junk_line("---") is True
    assert core.is_junk_line("Hello world") is False
    assert core.is_junk_line("") is False
    assert core.is_junk_line("   ") is False


@pytest.mark.unit
def test_is_junk_line_artifacts():
    """Test is_junk_line detects known artifacts."""
    assert core.is_junk_line("(Pause)") is True
    assert core.is_junk_line("[music]") is True
    assert core.is_junk_line("Desc-") is True
    assert core.is_junk_line("subscribe to our podcast") is True
    assert core.is_junk_line("Hello world") is False


@pytest.mark.unit
def test_anonymize_speakers_basic():
    """Test anonymize_speakers replaces names with A:, B:, C: labels."""
    text = "Maya: Hello\nLiam: Hi there\nMaya: How are you?"
    result = core.anonymize_speakers(text)
    assert "A: Hello" in result
    assert "B: Hi there" in result
    assert "A: How are you?" in result
    assert "Maya" not in result
    assert "Liam" not in result


@pytest.mark.unit
def test_anonymize_speakers_multiple_speakers():
    """Test anonymize_speakers handles multiple speakers correctly."""
    text = "Alice: First\nBob: Second\nCharlie: Third\nAlice: Fourth"
    result = core.anonymize_speakers(text)
    assert "A: First" in result
    assert "B: Second" in result
    assert "C: Third" in result
    assert "A: Fourth" in result
    assert "Alice" not in result
    assert "Bob" not in result
    assert "Charlie" not in result


@pytest.mark.unit
def test_anonymize_speakers_preserves_content():
    """Test anonymize_speakers preserves non-speaker content."""
    text = "Maya: Hello world\nThis is regular text\nLiam: Another turn"
    result = core.anonymize_speakers(text)
    assert "A: Hello world" in result
    assert "This is regular text" in result
    assert "B: Another turn" in result


@pytest.mark.unit
def test_strip_episode_header_markdown():
    """Test strip_episode_header removes markdown headers."""
    text = "# My Podcast\n## Episode 1\n\nMaya: Welcome!"
    result = core.strip_episode_header(text)
    assert "# My Podcast" not in result
    assert "## Episode 1" not in result
    assert "Maya: Welcome!" in result


@pytest.mark.unit
def test_strip_episode_header_host_guest():
    """Test strip_episode_header removes Host: and Guest: lines."""
    text = "Host: Maya\nGuest: Liam\n\nMaya: Welcome!"
    result = core.strip_episode_header(text)
    assert "Host: Maya" not in result
    assert "Guest: Liam" not in result
    assert "Maya: Welcome!" in result


@pytest.mark.unit
def test_strip_episode_header_guests_plural():
    """Test strip_episode_header removes Guests: (plural) lines."""
    text = "Guests: Alice, Bob, Charlie\n\nAlice: Hello"
    result = core.strip_episode_header(text)
    assert "Guests: Alice, Bob, Charlie" not in result
    assert "Alice: Hello" in result


@pytest.mark.unit
def test_cleaning_v4_profile_integration():
    """Test cleaning_v4 profile applies all new functions."""
    text = "# Show Title\nHost: Maya\n\nMaya: Hello\nLiam: Hi there\n////\n(Pause)"
    result = apply_profile(text, "cleaning_v4")
    # Should remove header
    assert "# Show Title" not in result
    assert "Host: Maya" not in result
    # Should anonymize speakers
    assert "Maya" not in result
    assert "Liam" not in result
    # Should remove junk
    assert "////" not in result
    assert "(Pause)" not in result
    # Should preserve content structure
    assert "A:" in result or "B:" in result


@pytest.mark.unit
def test_cleaning_v4_profile_registered():
    """Test cleaning_v4 profile is registered and callable."""
    from podcast_scraper.preprocessing.profiles import list_profiles

    profiles = list_profiles()
    assert "cleaning_v4" in profiles


@pytest.mark.unit
def test_cleaning_v4_vs_v3_difference():
    """Test cleaning_v4 produces different output than v3 (speaker anonymization)."""
    text = "Maya: Hello world\nLiam: How are you?"
    v3_result = apply_profile(text, "cleaning_v3")
    v4_result = apply_profile(text, "cleaning_v4")
    # v4 should anonymize, v3 should not
    assert "Maya" in v3_result or "Liam" in v3_result
    assert "Maya" not in v4_result
    assert "Liam" not in v4_result
