"""``speaker_naming_passes``: each naming LLM call asked k times, a name kept only when all agree.

Covers the ladder's resolution call (``_resolve_voices_k_passes``) and transcript naming
(``transcript_naming.agree``). Synthetic fixtures only.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_one_pass_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["speaker_naming_passes"].default == 1
