"""One person split into two diarized voices, merged by fingerprint (``merge_split_voices``).

Synthetic fingerprints only (never-commit-real-episodes).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

HOST = [1.0, 0.0, 0.0]
HOST_AGAIN = [0.95, 0.31, 0.0]  # cos to HOST ~0.95
GUEST = [0.0, 1.0, 0.0]


def test_off_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["merge_split_voices"].default is False
