"""Guard: the e2e OpenAI mock reads the transcript under BOTH RFC-115 layouts.

If the mock only read the user message, the transcript-first layout (flag on, the default) would
make it "summarize" the marker instead of the transcript — the e2e tests would still pass but
silently stop exercising transcript flow. This pins the mock's layout-aware extraction so it can't
regress.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.providers.common.transcript_cache import (
    openai_style_messages,
    TRANSCRIPT_BLOCK_HEADER,
)
from tests.e2e.fixtures.e2e_http_server import _bundled_gil_json
from tests.e2e.fixtures.openai_mock import _transcript_from_messages

pytestmark = pytest.mark.e2e

TRANSCRIPT = "HOST: welcome. GUEST: durable software for small teams and long-lived systems."


def test_mock_extracts_transcript_from_transcript_first_layout() -> None:
    # Flag ON: transcript is the leading block of the system message, marker in user.
    messages = openai_style_messages(
        TRANSCRIPT, "You are a summarizer.", f"Summarize.\n{TRANSCRIPT}", enabled=True
    )
    assert messages[0]["content"].startswith(TRANSCRIPT_BLOCK_HEADER)  # precondition
    assert _transcript_from_messages(messages) == TRANSCRIPT


def test_mock_extracts_transcript_from_legacy_layout() -> None:
    # Flag OFF: transcript stays in the user message.
    messages = openai_style_messages(
        TRANSCRIPT, "You are a summarizer.", f"Summarize.\n{TRANSCRIPT}", enabled=False
    )
    assert TRANSCRIPT in _transcript_from_messages(messages)


@pytest.mark.parametrize("enabled", [True, False], ids=["transcript-first", "legacy"])
def test_stack_server_bundled_quote_grounds_under_both_layouts(enabled: bool) -> None:
    """The airgapped stack-test HTTP server returns a VERBATIM transcript snippet for bundled quote
    extraction so quotes ground (resolve_llm_quote_span). If it read the relocated marker instead of
    the transcript, every grounded quote would vanish and the GIL stage would silently degrade."""
    user = f"Transcript (excerpt):\n{TRANSCRIPT}\n\nInsights:\n0: a claim\n\nReturn JSON only."
    msgs = openai_style_messages(TRANSCRIPT, "You extract quotes.", user, enabled=enabled)
    combined = msgs[0]["content"] + "\n" + msgs[1]["content"]
    result = _bundled_gil_json(combined)
    assert result is not None
    snippet = json.loads(result)["0"][0]
    assert snippet in TRANSCRIPT, "bundled-quote snippet must be verbatim from the transcript"
