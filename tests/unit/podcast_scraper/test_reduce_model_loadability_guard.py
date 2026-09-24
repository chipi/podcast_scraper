"""A reduce model this runtime cannot load must be substituted, not crashed on.

``PICKLE_ONLY_CHECKPOINTS`` sat in the manifest as data that nothing at runtime consulted:
it decided what to PRELOAD, while the model selectors went on naming those same checkpoints
as defaults. A model excluded from one path and still chosen by another is excluded from
neither — on x86_64 macOS (torch caps at 2.2.2, and transformers >= 4.56 refuses
``torch.load`` below 2.6) that surfaced as a ValueError from inside ``from_pretrained``,
taking out 38 e2e tests.

These tests pin BOTH halves: the substitution happens when the runtime cannot load the
checkpoint, and does NOT happen when it can — a guard that always fires is just a rename.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml import model_manifest as mm
from podcast_scraper.providers.ml.summarizer import _loadable_or_map_model

pytestmark = pytest.mark.unit

_PICKLE_ONLY = "allenai/led-base-16384"
_SAFETENSORS = "facebook/bart-base"


def test_the_pickle_only_set_is_not_empty() -> None:
    """Guard the guard: an empty set would make every test below vacuously pass."""
    assert _PICKLE_ONLY in mm.PICKLE_ONLY_CHECKPOINTS
    assert _SAFETENSORS not in mm.PICKLE_ONLY_CHECKPOINTS


def test_unloadable_reduce_model_falls_back_to_the_map_model(monkeypatch) -> None:
    monkeypatch.setattr(mm, "torch_refuses_pickle_weights", lambda: True)
    assert _loadable_or_map_model(_PICKLE_ONLY, _SAFETENSORS) == _SAFETENSORS


def test_a_loadable_reduce_model_is_left_alone(monkeypatch) -> None:
    """The substitution must be conditional, or it silently downgrades every summary."""
    monkeypatch.setattr(mm, "torch_refuses_pickle_weights", lambda: False)
    assert _loadable_or_map_model(_PICKLE_ONLY, _SAFETENSORS) == _PICKLE_ONLY


def test_a_safetensors_model_is_never_substituted(monkeypatch) -> None:
    monkeypatch.setattr(mm, "torch_refuses_pickle_weights", lambda: True)
    assert _loadable_or_map_model(_SAFETENSORS, "other") == _SAFETENSORS


def test_the_substitution_is_logged_with_its_cause(monkeypatch, caplog) -> None:
    """A silent swap makes a shorter-context summary look like a model regression."""
    import logging

    monkeypatch.setattr(mm, "torch_refuses_pickle_weights", lambda: True)
    with caplog.at_level(logging.WARNING):
        _loadable_or_map_model(_PICKLE_ONLY, _SAFETENSORS)
    messages = [r.getMessage() for r in caplog.records]
    assert any("CVE-2025-32434" in m for m in messages), messages
    assert any(_PICKLE_ONLY in m and _SAFETENSORS in m for m in messages), messages


@pytest.mark.parametrize(
    "version,refuses",
    [("2.2.2", True), ("2.5.1", True), ("2.6.0", False), ("2.9.0", False), ("3.0.0", False)],
)
def test_the_torch_version_boundary_is_2_6(monkeypatch, version, refuses) -> None:
    """2.6 is the version transformers names; off-by-one here re-breaks or over-degrades."""

    class _FakeTorch:
        __version__ = version

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch())
    assert mm.torch_refuses_pickle_weights() is refuses


def test_an_unparseable_torch_version_does_not_degrade(monkeypatch) -> None:
    """Unknown is not 'broken' — assume capable rather than silently downgrading output."""

    class _FakeTorch:
        __version__ = "not-a-version"

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch())
    assert mm.torch_refuses_pickle_weights() is False
