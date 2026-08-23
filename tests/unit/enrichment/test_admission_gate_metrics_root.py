"""E2 — the shipped gate_metrics resolver logs + degrades instead of masking a broken wheel.

If the packaged ``gate_metrics`` sub-package can't be resolved (a broken package_data glob in
a bad build), ``_shipped_gate_metrics_root`` must return None AND log a WARNING, so the silent
fallback to the (absent-in-image) repo data/eval path — the exact E2 self-rejection bug — is
visible in logs rather than re-hidden. (#1811 E2 review-fix)
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from podcast_scraper.enrichment.eval import admission


def test_shipped_gate_metrics_root_warns_and_returns_none_when_unresolvable(
    monkeypatch, caplog
) -> None:
    def _boom(_name: str):
        raise ModuleNotFoundError("no such package")

    monkeypatch.setattr(importlib.resources, "files", _boom)

    with caplog.at_level(logging.WARNING):
        result = admission._shipped_gate_metrics_root()

    assert result is None
    assert any(
        "gate_metrics package not resolvable" in r.getMessage() for r in caplog.records
    ), "a broken package resource must WARN, not silently fall back (#1811 E2)"


def test_shipped_gate_metrics_root_resolves_in_a_normal_install() -> None:
    # Sanity: in a correct editable/source install the resolver returns a real dir.
    root = admission._shipped_gate_metrics_root()
    assert root is not None
    assert isinstance(root, Path)
    assert root.is_dir()
