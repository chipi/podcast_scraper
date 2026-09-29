"""A failed KG extraction must say WHY, at a level an operator actually sees.

THE INCIDENT (prod, 2026-09-29). Three episodes failed KG extraction twice in a row during the
ADR-156 repair. The only trace anywhere was the pipeline's "provider extraction produced no
topics/entities". The same three episodes, re-run in isolation against the same model and profile,
extracted cleanly 3 out of 3 — so the cause lived in whatever the prod call got back, and that was
exactly what ``extract_kg_graph`` discarded: its ``except Exception`` logged at DEBUG and returned
None, and a reply that parsed to nothing was never logged at all. The code's own comment already
admitted a dead endpoint and "the model found no topics" were indistinguishable from outside.

Behaviour is deliberately unchanged — still None, so the fallback chain and the "no topics"
provenance run exactly as before. Only the visibility changes, and these tests pin it.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

pytestmark = pytest.mark.unit

_TRANSCRIPT = ("HOST: welcome. GUEST: we talk about humanoid robots and data. " * 20).strip()
_LOGGER = "podcast_scraper.providers.openai.openai_provider"


def _provider() -> OpenAIProvider:
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="openai",
        openai_summary_model="gpt-4o-mini",
        openai_api_key="sk-test-api-key-123",
    )
    p = OpenAIProvider(cfg)
    p._summarization_initialized = True
    return p


def _reply(content: str, finish_reason: str = "stop") -> Mock:
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage = Mock(prompt_tokens=100, completion_tokens=20)
    resp.usage.prompt_tokens_details = Mock(cached_tokens=0)
    resp.model = "gpt-4o-mini"
    resp.id = "r"
    return resp


def _run(p: OpenAIProvider, *, returns: Mock | None = None, raises: Exception | None = None):
    client = Mock()
    if raises is not None:
        client.chat.completions.create.side_effect = raises
    else:
        client.chat.completions.create.return_value = returns
    p.client = client
    return p.extract_kg_graph(text=_TRANSCRIPT, episode_title="Ep")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


class TestAFailureSaysWhy:
    def test_an_api_error_is_a_warning_naming_the_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Was logger.debug — invisible at the default level, which is how the incident hid."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            out = _run(_provider(), raises=RuntimeError("upstream connection reset"))
        assert out is None, "behaviour must not change: a failure still returns None"
        msgs = _warnings(caplog)
        assert any("RuntimeError" in m and "upstream connection reset" in m for m in msgs), msgs
        assert any("<none received>" in m for m in msgs), "must say no reply ever arrived"

    def test_a_reply_that_yields_nothing_shows_what_the_model_said(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The incident shape: a normal-length reply that parsed to no topics and no entities."""
        prose = "I'm sorry, but I can't produce a knowledge graph for this transcript."
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            out = _run(_provider(), returns=_reply(prose))
        assert not (out and (out.get("topics") or out.get("entities")))
        msgs = _warnings(caplog)
        assert any("can't produce a knowledge graph" in m for m in msgs), msgs

    def test_a_truncated_reply_reports_its_finish_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """finish_reason=length is the one-word answer to "was it cut off?"."""
        cut = '{"topics": [{"label": "humanoid robots", "description": "The emerg'
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _run(_provider(), returns=_reply(cut, finish_reason="length"))
        assert any("finish_reason=length" in m for m in _warnings(caplog)), _warnings(caplog)


class TestSuccessStaysQuiet:
    def test_a_good_extraction_logs_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        good = (
            '{"topics": [{"label": "humanoid robots", "description": "d"}],'
            ' "entities": [{"name": "One X", "entity_kind": "organization", "description": "d"}]}'
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            out = _run(_provider(), returns=_reply(good))
        assert out and out.get("topics"), out
        assert _warnings(caplog) == []


class TestTheSnippetIsBounded:
    def test_a_short_reply_is_shown_whole(self) -> None:
        from podcast_scraper.providers.openai.openai_provider import _kg_reply_snippet

        assert _kg_reply_snippet("tiny") == "'tiny'"

    def test_a_long_reply_is_head_and_tail_only(self) -> None:
        from podcast_scraper.providers.openai.openai_provider import _kg_reply_snippet

        raw = "H" * 400 + "M" * 5000 + "T" * 300
        out = _kg_reply_snippet(raw)
        assert "5200 chars" in out, out
        assert len(out) < 700, "a failure line must not carry a multi-kilobyte reply"
        assert out.startswith("'HHH") and out.endswith("TTT'")
