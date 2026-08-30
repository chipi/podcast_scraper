"""#1878: a summary that stops one token early must be repaired, not discarded.

The production shape (2026-08-30, runs 59322c3e / 0edf2e37): the backend emitted a complete,
coherent bullets JSON, closed the final string — and then declared ``finish_reason: stop``
without ever writing the trailing ``]`` and ``}``. Every byte of content was present; the parser
threw it away twice and the episode shipped with no summary.

These tests pin the repair against that EXACT payload shape (tail reconstructed from the
Langfuse-captured generations), plus the neighbouring shapes the closer must and must not touch.
"""

from __future__ import annotations

import pytest

from podcast_scraper.schemas.summary_schema import (
    _close_unterminated_json,
    parse_summary_output,
)

pytestmark = [pytest.mark.unit]


# The production failure, structurally: title + bullets, final bullet string properly closed,
# document missing its closing "]}" — verbatim shape of gen-1788059162 / gen-1788059166.
PROD_SHAPE = (
    '{\n  "title": "Aladdin: A generative model that predicts disease trajectories",\n'
    '  "bullets": [\n'
    '    "The model uses diagnostic codes from electronic health records.",\n'
    '    "Signature loadings are recalculated after each new diagnosis, allowing physicians '
    'to reassess one-year risk."'
)


class TestCloseUnterminatedJson:
    def test_the_production_shape_closes_and_parses(self):
        import json

        closed = _close_unterminated_json(PROD_SHAPE)
        assert closed is not None
        data = json.loads(closed)
        assert data["title"].startswith("Aladdin")
        assert len(data["bullets"]) == 2

    def test_truncation_mid_string_closes_the_string_too(self):
        import json

        closed = _close_unterminated_json('{"bullets": ["cut off mid sent')
        assert closed is not None
        assert json.loads(closed)["bullets"] == ["cut off mid sent"]

    def test_balanced_document_returns_none(self):
        # Already-valid JSON is not this failure mode — the closer must not claim it.
        assert _close_unterminated_json('{"bullets": ["done"]}') is None

    def test_mismatched_closer_returns_none(self):
        # Appending cannot fix a corrupt interior; pretending otherwise would mask real damage.
        assert _close_unterminated_json('{"bullets": ["x"}') is None

    def test_escaped_quotes_do_not_confuse_the_string_scanner(self):
        import json

        closed = _close_unterminated_json('{"bullets": ["a \\"quoted\\" phrase"')
        assert closed is not None
        assert json.loads(closed)["bullets"] == ['a "quoted" phrase']


class TestParseSummaryOutputRepairsTruncation:
    def test_production_payload_now_parses_with_repair_flag(self):
        result = parse_summary_output(PROD_SHAPE, provider=None, episode_title=None)
        assert result.success, f"expected repair to save the payload, got: {result.error}"
        assert result.repair_attempted, "must be flagged as repaired, not silently clean"
        assert result.schema is not None
        assert len(result.schema.bullets) == 2
        # Policy (#1878 review): bracket-append only = content was COMPLETE -> valid.
        assert result.schema.status == "valid"

    def test_string_closure_repair_ships_degraded(self):
        # Policy (#1878 review): terminating an OPEN string fabricated the final bullet's ending
        # — a half-bullet summary must carry degraded status, never pass as whole.
        cut_mid_bullet = (
            '{"title": "T", "bullets": ["First complete point.", "Second point cut mid sent'
        )
        result = parse_summary_output(cut_mid_bullet, provider=None, episode_title=None)
        assert result.success and result.repair_attempted
        assert result.schema is not None
        assert result.schema.status == "degraded"

    def test_fenced_and_truncated_together(self):
        result = parse_summary_output("```json\n" + PROD_SHAPE, provider=None, episode_title=None)
        assert result.success and result.schema is not None

    def test_hopeless_garbage_still_fails(self):
        # The repair must not turn the strict-contract rejection into false success.
        result = parse_summary_output(
            '{"bullets": ["x"} broken beyond append-only repair', provider=None, episode_title=None
        )
        assert not result.success
