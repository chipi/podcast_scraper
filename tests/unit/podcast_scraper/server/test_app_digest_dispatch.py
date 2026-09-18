"""The shared digest dispatcher — the single enqueuer list both schedulers drive (#2119).

These lock the properties that make the duplication-bug unrepeatable:
  * every declared enqueuer is actually called, with the right arguments;
  * one failing enqueuer does not stop the others (independent cadences);
  * results are reported PER ENQUEUER, never as a bare total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_digest_dispatch

pytestmark = pytest.mark.unit


def _stub(monkeypatch, **returns):
    """Stub every enqueuer; a value may be a list of ids or an Exception to raise."""
    import importlib

    calls: list[tuple[str, Path, Path]] = []
    for label, module_name, func_name in app_digest_dispatch.ENQUEUERS:
        module = importlib.import_module(f"podcast_scraper.server.{module_name}")
        value = returns.get(label, [])

        def _fn(root, data, _v=value, _l=label):
            calls.append((_l, root, data))
            if isinstance(_v, Exception):
                raise _v
            return _v

        monkeypatch.setattr(module, func_name, _fn)
    return calls


def test_every_declared_enqueuer_is_called(monkeypatch, tmp_path) -> None:
    calls = _stub(monkeypatch)
    app_digest_dispatch.enqueue_all_due(tmp_path / "corpus", tmp_path / "data")
    called = [c[0] for c in calls]
    assert called == [label for label, _, _ in app_digest_dispatch.ENQUEUERS]


def test_enqueuers_receive_corpus_root_and_data_dir(monkeypatch, tmp_path) -> None:
    calls = _stub(monkeypatch)
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    app_digest_dispatch.enqueue_all_due(corpus, data)
    for _, root, dd in calls:
        assert root == corpus
        assert dd == data


def test_results_are_per_enqueuer_not_a_bare_total(monkeypatch, tmp_path) -> None:
    """A single number across several enqueuers cannot distinguish 'nobody due' from 'never
    called' — which is the ambiguity that hid #2119 for months."""
    _stub(monkeypatch, weekly=["a", "b"], daily_recap=["c"])
    res = app_digest_dispatch.enqueue_all_due(tmp_path / "c", tmp_path / "d")
    assert res.total == 3
    assert res.ids["weekly"] == ["a", "b"]
    assert res.ids["daily_recap"] == ["c"]
    assert res.ids["recommendations"] == []
    assert "weekly=2" in res.summary()
    assert "recommendations=0" in res.summary()
    assert "daily_recap=1" in res.summary()


def test_one_failing_enqueuer_does_not_mute_the_others(monkeypatch, tmp_path) -> None:
    """Cadences are independent. Coupling them would let one broken assembler silence every
    notification the product has."""
    _stub(monkeypatch, weekly=RuntimeError("outbox unreachable"), daily_recap=["survivor"])
    res = app_digest_dispatch.enqueue_all_due(tmp_path / "c", tmp_path / "d")
    assert "weekly" in res.errors
    assert "outbox unreachable" in res.errors["weekly"]
    assert res.ids["daily_recap"] == ["survivor"]  # healthy one still ran
    assert res.all_ids == ["survivor"]
    assert "weekly=ERR" in res.summary()


def test_dispatch_never_raises_for_an_enqueuer_failure(monkeypatch, tmp_path) -> None:
    """The callers rely on this: the sidecar loop and the scheduler branch must not die."""
    _stub(
        monkeypatch,
        weekly=RuntimeError("a"),
        recommendations=ValueError("b"),
        daily_recap=KeyError("c"),
    )
    res = app_digest_dispatch.enqueue_all_due(tmp_path / "c", tmp_path / "d")  # must not raise
    assert set(res.errors) == {"weekly", "recommendations", "daily_recap"}
    assert res.total == 0


def test_enqueuer_targets_all_resolve(tmp_path) -> None:
    """Each declared (module, function) must actually exist — a typo here disables a whole
    cadence in BOTH schedulers at once, which is the cost of having one list."""
    import importlib

    for label, module_name, func_name in app_digest_dispatch.ENQUEUERS:
        module = importlib.import_module(f"podcast_scraper.server.{module_name}")
        func = getattr(module, func_name, None)
        assert callable(func), f"{label}: {module_name}.{func_name} is missing or not callable"


def test_every_comms_type_with_a_schedule_has_an_enqueuer() -> None:
    """Product-level guard: a notification type the UI offers must have something that can
    produce it. ``daily_recap`` was toggleable in the profile UI while nothing in prod could
    ever emit it (#2119).

    ``new_episodes`` is intentionally exempt — it is push-nudged from the weekly enqueuer rather
    than having a cadence slot of its own. ``product`` is announcement-driven, not scheduled.
    """
    from podcast_scraper.server import app_comms_store

    scheduled_types = {"digest", "daily_recap"}
    assert scheduled_types <= set(app_comms_store.TYPES)

    labels = {label for label, _, _ in app_digest_dispatch.ENQUEUERS}
    # "digest" is the weekly cadence; its enqueuer label is "weekly".
    assert "weekly" in labels, "no enqueuer for the weekly 'digest' comms type"
    assert "daily_recap" in labels, "no enqueuer for the 'daily_recap' comms type"
