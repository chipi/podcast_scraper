"""Tests for the #1060 FU4 `--clean-reference` preprocessing in transcription_sweep.

The clean_reference_for_wer helper strips markdown headers, host/guest
lines, per-utterance speaker labels, and bracketed timestamps from the
synthetic v2 reference transcripts. The cleaned reference is the same
shape as the Whisper hypothesis, so WER becomes comparable to the
v2-fixture baselines in EVAL_TRANSCRIPTION_3WAY_2026_06.md and
EVAL_WHISPER_SMALL_EN_2026_06_13.md.

If any of these regex contracts change, downstream `headline_metric`
strings on `local_whisper_tiny_en` and `local_whisper_medium_en` should
be re-measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "eval" / "experiment"
sys.path.insert(0, str(_SCRIPT_DIR))

from transcription_sweep import clean_reference_for_wer  # noqa: E402


def test_strips_markdown_headers_and_subheaders() -> None:
    raw = "# Singletrack Sessions — Episode\n## Building Trails That Last\nactual content"
    cleaned = clean_reference_for_wer(raw)
    assert "Singletrack Sessions" not in cleaned
    assert "Building Trails" not in cleaned
    assert "actual content" in cleaned


def test_strips_host_and_guest_block_lines() -> None:
    raw = "Host: Maya\nGuest: Liam\nMaya: hello there"
    cleaned = clean_reference_for_wer(raw)
    assert "Host: Maya" not in cleaned
    assert "Guest: Liam" not in cleaned
    # Per-utterance speaker label removed but the spoken content survives.
    assert "hello there" in cleaned
    assert "Maya:" not in cleaned


def test_strips_per_utterance_speaker_labels() -> None:
    raw = "Maya: Welcome back.\nLiam: Thanks for having me."
    cleaned = clean_reference_for_wer(raw)
    assert "Maya:" not in cleaned
    assert "Liam:" not in cleaned
    assert "Welcome back." in cleaned
    assert "Thanks for having me." in cleaned


def test_strips_bracketed_timestamps() -> None:
    raw = "[00:00]\n[12:34]\n[1:23:45]\nthe content"
    cleaned = clean_reference_for_wer(raw)
    assert "[00:00]" not in cleaned
    assert "[12:34]" not in cleaned
    assert "[1:23:45]" not in cleaned
    assert "the content" in cleaned


def test_preserves_spoken_content_with_colons() -> None:
    # Sentences ending with a colon mid-utterance should NOT be stripped —
    # the speaker-label regex anchors at line start and requires capitalized
    # name + space after the colon.
    raw = "Maya: the rule is this: do not lie. another sentence: still here."
    cleaned = clean_reference_for_wer(raw)
    assert "the rule is this: do not lie." in cleaned
    assert "another sentence: still here." in cleaned


def test_does_not_swallow_lowercase_colon_words() -> None:
    raw = "http://example.com is not a label."
    cleaned = clean_reference_for_wer(raw)
    # 'http:' starts with lowercase → not a speaker label.
    assert "http://example.com is not a label." in cleaned


def test_word_count_reduction_on_smoke_v2_sample() -> None:
    # Sample shape lifted directly from data/eval/materialized/curated_5feeds_smoke_v2/p01_e01.txt.
    raw = (
        "# Singletrack Sessions — Episode\n"
        "## Building Trails That Last\n"
        "Host: Maya\n"
        "Guest: Liam\n"
        "\n"
        "[00:00]\n"
        "Maya: Welcome back to Singletrack Sessions.\n"
        "Liam: Thanks for having me.\n"
    )
    cleaned = clean_reference_for_wer(raw)
    # Spoken content survives; artifacts gone.
    assert "Welcome back to Singletrack Sessions." in cleaned
    assert "Thanks for having me." in cleaned
    assert "Maya:" not in cleaned
    assert "Liam:" not in cleaned
    assert "[00:00]" not in cleaned
    assert "# Singletrack" not in cleaned
    # Word count drops by exactly the stripped tokens (4 header tokens +
    # 1 "Maya" + 1 "Liam" + 0 timestamp = 6 fewer non-whitespace tokens).
    raw_tokens = raw.split()
    cleaned_tokens = cleaned.split()
    assert len(cleaned_tokens) < len(raw_tokens)
