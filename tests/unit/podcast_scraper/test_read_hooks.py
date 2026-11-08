"""The server's read side reaches the speaker, ad and identity stages through read_hooks
(E04 S3), kept out of the registry corpus migrations reach."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import builtin_read_hooks, read_hooks
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.read_hooks import ReadHooks

pytestmark = pytest.mark.unit

_NONE = ReadHooks(name="builtin")


def test_without_any_stage_each_hook_answers_its_plain_default() -> None:
    with patch.object(builtin_read_hooks, "BUILTIN", _NONE):
        assert read_hooks.segment_speaker({"speaker": "SPEAKER_00"}) == (None, None)
        assert read_hooks.unidentified_label() == "Unidentified speaker"
        assert read_hooks.ad_reads([{"start": 0.0, "end": 1.0}]) == []
        assert read_hooks.looks_like_publisher("NPR") is False
        assert read_hooks.entity_id_map(Path("/c")) == {}
        assert read_hooks.gi_integrity(Path("/c")) is None


def test_without_the_built_in_an_extension_owns_the_hook() -> None:
    ext = Extension(name="x", read_hooks=ReadHooks(name="x", unidentified_label=lambda: "Voice"))
    with patch.object(builtin_read_hooks, "BUILTIN", _NONE), use_extensions([ext]):
        assert read_hooks.unidentified_label() == "Voice"


def test_the_built_in_answers_before_an_extension() -> None:
    ext = Extension(name="x", read_hooks=ReadHooks(name="x", unidentified_label=lambda: "Voice"))
    with use_extensions([ext]):
        assert read_hooks.unidentified_label() == "Unidentified speaker"


def test_the_publisher_reader_gets_the_language() -> None:
    with patch(
        "podcast_scraper.speaker_detectors.hosts.looks_like_publisher", return_value=True
    ) as reader:
        assert read_hooks.looks_like_publisher("Rádio Senado", "pt") is True
    reader.assert_called_once_with("Rádio Senado", "pt")
