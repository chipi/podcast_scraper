"""GlitchTip groups the same error into ONE issue, whatever numbers or ids it carries.

2026-10-15: podcast-pipeline held 424 unresolved issues that were 72 errors. One error,
"context limit 65536 cannot fit this request: the prompt alone is ~70118 tokens, leaving -4582
for the output", was 194 issues, one per distinct token count, because log-message events
group by their final text. before_send now sets a fingerprint from the logger and the message
with its variable parts (numbers, money, ids, quoted values, URLs) replaced.
"""

from __future__ import annotations

import pytest

from podcast_scraper.utils.sentry_init import _before_send, _normalize_message

pytestmark = pytest.mark.unit


def _log_event(msg: str, logger: str = "podcast_scraper.providers.vllm") -> dict:
    return {"logger": logger, "logentry": {"formatted": msg}, "level": "error"}


def test_same_error_different_numbers_same_fingerprint() -> None:
    a = _before_send(
        _log_event(
            "context limit 65536 cannot fit this request: the prompt alone is ~70118 tokens, "
            "leaving -4582 for the output"
        ),
        None,
    )
    b = _before_send(
        _log_event(
            "context limit 32768 cannot fit this request: the prompt alone is ~41002 tokens, "
            "leaving -8234 for the output"
        ),
        None,
    )
    assert a["fingerprint"] == b["fingerprint"]


def test_money_ids_quotes_and_urls_do_not_split_issues() -> None:
    a = _before_send(
        _log_event(
            "[ep-1] cost soft cap exceeded: $3.21 > $3.00 run=run-20261015T094154.875903Z "
            "episode 'b838e720-7f99-11f1-b659-cffa64e5d561' https://x/feed.xml"
        ),
        None,
    )
    b = _before_send(
        _log_event(
            "[ep-9] cost soft cap exceeded: $12.04 > $10.00 run=run-20261016T010000.000001Z "
            "episode 'aaaaaaaa-1111-2222-3333-444444444444' https://y/other.xml"
        ),
        None,
    )
    assert a["fingerprint"] == b["fingerprint"]


def test_different_errors_stay_different() -> None:
    a = _before_send(_log_event("cost soft cap exceeded: $3.21 > $3.00"), None)
    b = _before_send(
        _log_event(
            "DEADLINE EXCEEDED: summarization for episode 7 has been running longer than 600s"
        ),
        None,
    )
    c = _before_send(
        _log_event("cost soft cap exceeded: $3.21 > $3.00", logger="podcast_scraper.workflow"), None
    )
    assert len({tuple(a["fingerprint"]), tuple(b["fingerprint"]), tuple(c["fingerprint"])}) == 3


def test_the_logging_template_is_preferred_when_present() -> None:
    e = {
        "logger": "l",
        "logentry": {
            "message": "deadline %ds for %s",
            "formatted": "deadline 600s for ep-3",
            "params": [600, "ep-3"],
        },
    }
    assert _before_send(e, None)["fingerprint"] == ["l", "deadline %ds for %s"]


def test_exceptions_group_by_type_and_normalized_value() -> None:
    def exc(v: str) -> dict:
        return {"exception": {"values": [{"type": "CostCapExceeded", "value": v}]}}

    a = _before_send(exc("cost soft cap exceeded: $3.21 > $3.00"), None)
    b = _before_send(exc("cost soft cap exceeded: $9.99 > $5.00"), None)
    assert a["fingerprint"] == b["fingerprint"]
    assert a["fingerprint"][0] == "CostCapExceeded"


def test_an_explicit_fingerprint_is_kept() -> None:
    e = _log_event("x 1")
    e["fingerprint"] = ["mine"]
    assert _before_send(e, None)["fingerprint"] == ["mine"]


def test_an_event_without_message_or_exception_keeps_default_grouping() -> None:
    assert "fingerprint" not in _before_send({"level": "error"}, None)


def test_normalize_is_stable_and_bounded() -> None:
    assert _normalize_message("took 12.5s, 3 retries, id 0xdeadbeef01") == _normalize_message(
        "took 9s, 14 retries, id 0xcafebabe02"
    )
    assert len(_normalize_message("a " * 1000)) <= 200
