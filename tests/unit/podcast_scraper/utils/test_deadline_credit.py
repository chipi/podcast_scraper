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

from podcast_scraper.utils.timeout import timeout_context, TimeoutError


def test_uncredited_overrun_still_alerts_and_raises(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TimeoutError):
            with timeout_context(1, "metadata generation"):
                time.sleep(1.3)
    assert "DEADLINE EXCEEDED" in caplog.text


# The credit above arrives only when translation ENDS. A translation longer than the deadline
# itself therefore still fired the alarm mid-way: on 2026-10-12 every El Hilo, Radio Ambulante,
# SWR, Rádio Novelo and Camera episode logged "DEADLINE EXCEEDED: metadata generation" with
# summary, GI and KG all off. The exclusion counts while the excluded block is still running.


@pytest.fixture
def short_rearm(monkeypatch):
    from podcast_scraper.utils import timeout as t

    monkeypatch.setattr(t, "EXCLUSION_REARM_FLOOR_S", 0.1)


def test_an_exclusion_still_running_at_the_deadline_does_not_alert(caplog, short_rearm) -> None:
    from podcast_scraper.utils.timeout import excluded_from_deadline

    with caplog.at_level(logging.ERROR):
        with timeout_context(1, "metadata generation"):
            with excluded_from_deadline():
                time.sleep(1.5)  # the translation, longer than the whole deadline
            time.sleep(0.3)
    assert "DEADLINE EXCEEDED" not in caplog.text


def test_time_outside_the_exclusion_still_counts(caplog, short_rearm) -> None:
    from podcast_scraper.utils.timeout import excluded_from_deadline

    with caplog.at_level(logging.ERROR):
        with pytest.raises(TimeoutError):
            with timeout_context(1, "metadata generation"):
                time.sleep(0.6)
                with excluded_from_deadline():
                    time.sleep(1.2)
                time.sleep(0.6)
    assert "DEADLINE EXCEEDED" in caplog.text


def test_an_exclusion_outside_any_deadline_is_a_no_op() -> None:
    from podcast_scraper.utils.timeout import excluded_from_deadline

    with excluded_from_deadline():
        pass


def test_the_translation_stage_runs_inside_an_exclusion(monkeypatch) -> None:
    from contextlib import contextmanager

    from podcast_scraper.workflow import translation_stage as ts

    entered: list[bool] = []

    @contextmanager
    def recording():
        entered.append(True)
        yield

    monkeypatch.setattr(ts, "excluded_from_deadline", recording)
    monkeypatch.setattr(
        ts,
        "decide_translation",
        lambda cfg, **kw: ts.TranslationOutcome(status=ts.STATUS_SKIPPED, reason="x"),
    )
    ts.run_translation_stage(object())
    assert entered == [True]
