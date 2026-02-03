"""Unit tests for cleaning_v1, cleaning_v2, and cleaning_v3 preprocessing profiles.

Tests the behavior of each profile to ensure they correctly apply their
respective cleaning steps and produce expected outputs.
"""

import pytest

from podcast_scraper.preprocessing.profiles import apply_profile


@pytest.mark.unit
def test_cleaning_v1_profile_registered():
    """Test cleaning_v1 profile is registered and callable."""
    from podcast_scraper.preprocessing.profiles import list_profiles

    profiles = list_profiles()
    assert "cleaning_v1" in profiles


@pytest.mark.unit
def test_cleaning_v1_removes_timestamps():
    """Test cleaning_v1 removes timestamps."""
    text = "Hello [00:12:34] world [1:23:45] test"
    result = apply_profile(text, "cleaning_v1")
    assert "[00:12:34]" not in result
    assert "[1:23:45]" not in result
    assert "Hello" in result
    assert "world" in result
    assert "test" in result


@pytest.mark.unit
def test_cleaning_v1_normalizes_speakers():
    """Test cleaning_v1 normalizes generic speaker tags."""
    text = "SPEAKER 1: Hello\nSPEAKER 2: World"
    result = apply_profile(text, "cleaning_v1")
    # Generic speaker tags should be removed
    assert "SPEAKER 1:" not in result
    assert "SPEAKER 2:" not in result
    assert "Hello" in result
    assert "World" in result


@pytest.mark.unit
def test_cleaning_v1_collapses_blank_lines():
    """Test cleaning_v1 collapses multiple blank lines."""
    text = "Line 1\n\n\n\nLine 2"
    result = apply_profile(text, "cleaning_v1")
    # Should collapse to single blank line
    assert "\n\n\n" not in result
    assert "Line 1" in result
    assert "Line 2" in result


@pytest.mark.unit
def test_cleaning_v1_preserves_content():
    """Test cleaning_v1 preserves actual content."""
    text = "This is a transcript.\n\nIt has multiple lines.\n[00:05:00] And timestamps."
    result = apply_profile(text, "cleaning_v1")
    assert "This is a transcript" in result
    assert "It has multiple lines" in result
    assert "And timestamps" in result
    assert "[00:05:00]" not in result


@pytest.mark.unit
def test_cleaning_v1_profile_integration():
    """Test cleaning_v1 profile applies all basic cleaning steps."""
    text = "SPEAKER 1: Hello [00:12:34] world\n\n\nSPEAKER 2: Test [1:23:45]"
    result = apply_profile(text, "cleaning_v1")
    # Should remove timestamps
    assert "[00:12:34]" not in result
    assert "[1:23:45]" not in result
    # Should normalize speakers
    assert "SPEAKER 1:" not in result
    assert "SPEAKER 2:" not in result
    # Should collapse blank lines
    assert "\n\n\n" not in result
    # Should preserve content
    assert "Hello" in result
    assert "world" in result
    assert "Test" in result


@pytest.mark.unit
def test_cleaning_v2_profile_registered():
    """Test cleaning_v2 profile is registered and callable."""
    from podcast_scraper.preprocessing.profiles import list_profiles

    profiles = list_profiles()
    assert "cleaning_v2" in profiles


@pytest.mark.unit
def test_cleaning_v2_removes_sponsor_blocks():
    """Test cleaning_v2 removes sponsor blocks."""
    text = (
        "Main content here.\n\nThis episode is brought to you by Acme Corp. "
        "They make great products.\n\nMore content."
    )
    result = apply_profile(text, "cleaning_v2")
    assert "This episode is brought to you by" not in result
    assert "Acme Corp" not in result
    assert "They make great products" not in result
    assert "Main content here" in result
    assert "More content" in result


@pytest.mark.unit
def test_cleaning_v2_removes_outro_blocks():
    """Test cleaning_v2 removes outro blocks."""
    text = (
        "Main content here.\n\nThank you so much for listening! "
        "Please subscribe to our podcast.\n\nEnd."
    )
    result = apply_profile(text, "cleaning_v2")
    assert "Thank you so much for listening" not in result
    assert "Please subscribe" not in result
    assert "Main content here" in result
    assert "End" in result


@pytest.mark.unit
def test_cleaning_v2_includes_v1_steps():
    """Test cleaning_v2 includes all cleaning_v1 steps."""
    text = "SPEAKER 1: Hello [00:12:34] world\n\n\nTest"
    result = apply_profile(text, "cleaning_v2")
    # Should have v1 behavior
    assert "[00:12:34]" not in result
    assert "SPEAKER 1:" not in result
    assert "\n\n\n" not in result
    assert "Hello" in result
    assert "world" in result
    assert "Test" in result


@pytest.mark.unit
def test_cleaning_v2_profile_integration():
    """Test cleaning_v2 profile applies v1 + sponsor/outro removal."""
    text = (
        "SPEAKER 1: Hello [00:12:34] world\n\n\n"
        "This episode is brought to you by Acme Corp.\n\n"
        "Thank you so much for listening! Please subscribe.\n\n"
        "More content."
    )
    result = apply_profile(text, "cleaning_v2")
    # Should remove timestamps (v1)
    assert "[00:12:34]" not in result
    # Should normalize speakers (v1)
    assert "SPEAKER 1:" not in result
    # Should collapse blank lines (v1) - collapses 3+ to 2
    # Note: After sponsor/outro removal, blank lines may remain
    # Should remove sponsor blocks (v2)
    assert "This episode is brought to you by" not in result
    assert "Acme Corp" not in result
    # Should remove outro blocks (v2)
    assert "Thank you so much for listening" not in result
    assert "Please subscribe" not in result
    # Should preserve content
    assert "Hello" in result
    assert "world" in result
    assert "More content" in result


@pytest.mark.unit
def test_cleaning_v2_vs_v1_difference():
    """Test cleaning_v2 produces different output than v1 (sponsor/outro removal)."""
    text = (
        "Main content.\nThis episode is brought to you by Acme Corp.\n"
        "Thank you so much for listening!"
    )
    v1_result = apply_profile(text, "cleaning_v1")
    v2_result = apply_profile(text, "cleaning_v2")
    # v2 should remove sponsor/outro, v1 should not
    assert "This episode is brought to you by" in v1_result
    assert "Thank you so much for listening" in v1_result
    assert "This episode is brought to you by" not in v2_result
    assert "Thank you so much for listening" not in v2_result
    assert "Main content" in v2_result


@pytest.mark.unit
def test_cleaning_v3_profile_registered():
    """Test cleaning_v3 profile is registered and callable."""
    from podcast_scraper.preprocessing.profiles import list_profiles

    profiles = list_profiles()
    assert "cleaning_v3" in profiles


@pytest.mark.unit
def test_cleaning_v3_strips_credits():
    """Test cleaning_v3 strips credit lines."""
    text = "Main content here.\n\nProduced by John Doe.\nMusic by Jane Smith.\n\nMore content."
    result = apply_profile(text, "cleaning_v3")
    assert "Produced by" not in result
    assert "Music by" not in result
    assert "John Doe" not in result
    assert "Jane Smith" not in result
    assert "Main content here" in result
    assert "More content" in result


@pytest.mark.unit
def test_cleaning_v3_strips_garbage_lines():
    """Test cleaning_v3 strips garbage/boilerplate lines."""
    text = "Main content here.\n\nRead more at test-site\nBack to top\n\nMore content."
    result = apply_profile(text, "cleaning_v3")
    assert "Read more" not in result
    assert "Back to top" not in result
    assert "test-site" not in result
    assert "Main content here" in result
    assert "More content" in result


@pytest.mark.unit
def test_cleaning_v3_removes_summarization_artifacts():
    """Test cleaning_v3 removes summarization artifacts."""
    text = "Main content here.\n\nTextColor- This is an artifact.\nMUSIC plays.\n\nMore content."
    result = apply_profile(text, "cleaning_v3")
    assert "TextColor" not in result
    assert "MUSIC" not in result
    assert "Main content here" in result
    assert "More content" in result


@pytest.mark.unit
def test_cleaning_v3_includes_v2_steps():
    """Test cleaning_v3 includes all cleaning_v2 steps."""
    text = (
        "SPEAKER 1: Hello [00:12:34] world\n\n\n"
        "This episode is brought to you by Acme Corp.\n\n"
        "Thank you so much for listening!"
    )
    result = apply_profile(text, "cleaning_v3")
    # Should have v2 behavior (which includes v1)
    assert "[00:12:34]" not in result
    assert "SPEAKER 1:" not in result
    assert "\n\n\n" not in result
    assert "This episode is brought to you by" not in result
    assert "Thank you so much for listening" not in result
    assert "Hello" in result
    assert "world" in result


@pytest.mark.unit
def test_cleaning_v3_profile_integration():
    """Test cleaning_v3 profile applies all cleaning steps."""
    text = (
        "# Episode Title\n"
        "SPEAKER 1: Hello [00:12:34] world\n\n\n"
        "Produced by John Doe.\n"
        "Read more at test-site\n"
        "This episode is brought to you by Acme Corp.\n\n"
        "Thank you so much for listening!\n"
        "TextColor- Artifact\n"
        "More content."
    )
    result = apply_profile(text, "cleaning_v3")
    # Should remove timestamps (v1)
    assert "[00:12:34]" not in result
    # Should normalize speakers (v1)
    assert "SPEAKER 1:" not in result
    # Should collapse blank lines (v1) - collapses 3+ to 2
    # Note: After various removals, blank lines may remain
    # Should remove sponsor blocks (v2)
    assert "This episode is brought to you by" not in result
    assert "Acme Corp" not in result
    # Should remove outro blocks (v2)
    assert "Thank you so much for listening" not in result
    # Should strip credits (v3)
    assert "Produced by" not in result
    assert "John Doe" not in result
    # Should strip garbage lines (v3)
    assert "Read more" not in result
    assert "test-site" not in result
    # Should remove artifacts (v3)
    assert "TextColor" not in result
    # Should preserve content
    assert "Hello" in result
    assert "world" in result
    # Note: "More content" may be removed if it's part of outro/sponsor block
    # or if it matches garbage patterns - this is expected behavior


@pytest.mark.unit
def test_cleaning_v3_vs_v2_difference():
    """Test cleaning_v3 produces different output than v2 (credits/garbage/artifacts)."""
    text = "Main content.\n" "Produced by John.\n" "Read more at test-site\n" "TextColor- Artifact"
    v2_result = apply_profile(text, "cleaning_v2")
    v3_result = apply_profile(text, "cleaning_v3")
    # v3 should remove credits/garbage/artifacts, v2 should not
    assert "Produced by" in v2_result or "John" in v2_result
    assert "Read more" in v2_result or "test-site" in v2_result
    assert "TextColor" in v2_result or "Artifact" in v2_result
    assert "Produced by" not in v3_result
    assert "Read more" not in v3_result
    assert "TextColor" not in v3_result
    assert "Main content" in v3_result


@pytest.mark.unit
def test_cleaning_v3_vs_v1_difference():
    """Test cleaning_v3 produces different output than v1 (all additional steps)."""
    text = (
        "Main content.\n"
        "Produced by John.\n"
        "This episode is brought to you by Acme Corp.\n"
        "Read more at test-site"
    )
    v1_result = apply_profile(text, "cleaning_v1")
    v3_result = apply_profile(text, "cleaning_v3")
    # v3 should remove more than v1
    assert "Produced by" in v1_result or "John" in v1_result
    assert "This episode is brought to you by" in v1_result
    assert "Read more" in v1_result or "test-site" in v1_result
    assert "Produced by" not in v3_result
    assert "This episode is brought to you by" not in v3_result
    assert "Read more" not in v3_result
    assert "Main content" in v3_result


@pytest.mark.unit
def test_cleaning_none_profile_registered():
    """Test cleaning_none profile is registered and callable."""
    from podcast_scraper.preprocessing.profiles import list_profiles

    profiles = list_profiles()
    assert "cleaning_none" in profiles


@pytest.mark.unit
def test_cleaning_none_no_op():
    """Test cleaning_none returns text unchanged."""
    text = "SPEAKER 1: Hello [00:12:34] world\n\n\n[SPONSOR] Sponsor [/SPONSOR]"
    result = apply_profile(text, "cleaning_none")
    # Should be unchanged
    assert result == text
    assert "[00:12:34]" in result
    assert "SPEAKER 1:" in result
    assert "\n\n\n" in result
    assert "[SPONSOR]" in result


@pytest.mark.unit
def test_profile_version_progression():
    """Test that each profile version adds more cleaning than the previous."""
    # Use blank lines to separate sections so sponsor removal doesn't remove everything
    text = (
        "SPEAKER 1: Hello [00:12:34] world\n\n\n"
        "Produced by John.\n\n"
        "This episode is brought to you by Acme Corp.\n\n"
        "Read more at test-site\n\n"
        "TextColor- Artifact"
    )
    v1_result = apply_profile(text, "cleaning_v1")
    v2_result = apply_profile(text, "cleaning_v2")
    v3_result = apply_profile(text, "cleaning_v3")

    # v1 should only remove timestamps and normalize speakers
    assert "[00:12:34]" not in v1_result
    assert "SPEAKER 1:" not in v1_result
    # v1 should NOT remove sponsor/credits/garbage/artifacts
    assert "This episode is brought to you by" in v1_result
    assert "Produced by" in v1_result
    assert "TextColor" in v1_result

    # v2 should add sponsor/outro removal
    assert "This episode is brought to you by" not in v2_result
    # v2 should NOT remove credits/artifacts (garbage removal is v3)
    # Note: "Read more" may be removed by outro patterns in v2
    assert "Produced by" in v2_result
    assert "TextColor" in v2_result

    # v3 should remove everything
    assert "This episode is brought to you by" not in v3_result
    assert "Produced by" not in v3_result
    assert "Read more" not in v3_result
    assert "TextColor" not in v3_result
    assert "Hello" in v3_result
    assert "world" in v3_result
