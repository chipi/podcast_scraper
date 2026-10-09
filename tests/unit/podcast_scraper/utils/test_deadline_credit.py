"""Translation's wall time is credited back to the metadata deadline (#2230, RFC-124).

Found on El Hilo (es, 2026-10-09): with summaries, GI and KG all OFF, a 41-minute episode logged
"DEADLINE EXCEEDED: metadata generation (summary+GI+KG)" at 1200 s — the whole overrun was
translation, which runs inside that block. The design credited translation's time back so the
alert keeps meaning "summary + GI + KG were slow"; the credit was never built.
"""

from __future__ import annotations

import logging
import time

import pytest

from podcast_scraper.utils.timeout import credit_deadline, timeout_context, TimeoutError


def test_credited_time_does_not_count_toward_the_deadline(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with timeout_context(1, "metadata generation"):
            time.sleep(0.6)
            credit_deadline(0.6)  # the translation that just ran
            time.sleep(0.6)
    assert "DEADLINE EXCEEDED" not in caplog.text


def test_uncredited_overrun_still_alerts_and_raises(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TimeoutError):
            with timeout_context(1, "metadata generation"):
                time.sleep(1.3)
    assert "DEADLINE EXCEEDED" in caplog.text


def test_credit_only_extends_by_what_was_credited(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TimeoutError):
            with timeout_context(1, "metadata generation"):
                credit_deadline(0.3)
                time.sleep(1.6)
    assert "DEADLINE EXCEEDED" in caplog.text


def test_a_credit_outside_any_deadline_is_a_no_op() -> None:
    credit_deadline(5.0)


def test_the_translation_stage_credits_its_wall_time(monkeypatch) -> None:
    from podcast_scraper.workflow import translation_stage as ts

    credited: list[float] = []
    monkeypatch.setattr(ts, "credit_deadline", credited.append)
    monkeypatch.setattr(
        ts,
        "decide_translation",
        lambda cfg, **kw: ts.TranslationOutcome(status=ts.STATUS_SKIPPED, reason="x"),
    )
    ts.run_translation_stage(object())
    assert len(credited) == 1 and credited[0] >= 0
