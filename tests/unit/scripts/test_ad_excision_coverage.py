"""The ad-excision coverage audit must count what is on disk, and recompute what is not (#1976).

Per-episode excision data already exists — ``<base>.adfree.admap.json``, written by default and
written even when nothing was cut, so denominators exist. The gap #1976 identified was that
nothing reads them corpus-wide. This audit does, and dry-runs the detector for episodes predating
the sidecar so coverage is retroactive.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "audit" / "ad_excision_coverage.py"


def _load():
    spec = importlib.util.spec_from_file_location("ad_excision_coverage", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(corpus: Path, feed: str, name: str, *, admap: dict | None, text: str = "") -> None:
    d = corpus / "feeds" / feed / "run_1" / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.txt").write_text(text or "Some ordinary episode content.", encoding="utf-8")
    if admap is not None:
        (d / f"{name}.adfree.admap.json").write_text(json.dumps(admap), encoding="utf-8")


def test_counts_cuts_and_no_ops_separately(tmp_path: Path) -> None:
    """An identity record is a denominator, not a cut — conflating them inflates coverage."""
    mod = _load()
    corpus = tmp_path / "corpus"
    _episode(corpus, "showA", "e1", admap={"chars_removed": 1200, "source_length": 40000})
    _episode(corpus, "showA", "e2", admap={"chars_removed": 0, "source_length": 40000})
    _episode(corpus, "showA", "e3", admap={"chars_removed": 800, "source_length": 40000})

    stats = mod.collect(corpus)
    assert stats["showA"]["episodes"] == 3
    assert stats["showA"]["cut"] == 2
    assert stats["showA"]["chars_removed"] == 2000
    assert stats["showA"]["from_sidecar"] == 3
    assert stats["showA"]["recomputed"] == 0


def test_episodes_without_a_sidecar_are_recomputed_not_skipped(tmp_path: Path) -> None:
    """Retroactive coverage is the point — skipping them would understate the denominator."""
    mod = _load()
    corpus = tmp_path / "corpus"
    _episode(corpus, "showB", "e1", admap=None, text="An episode with no ad map on disk. " * 50)

    stats = mod.collect(corpus)
    assert stats["showB"]["episodes"] == 1
    assert stats["showB"]["recomputed"] == 1
    assert stats["showB"]["from_sidecar"] == 0


def test_derived_artifacts_are_not_counted_as_episodes(tmp_path: Path) -> None:
    """``.adfree.txt`` / ``.segments`` siblings must not double-count an episode."""
    mod = _load()
    corpus = tmp_path / "corpus"
    _episode(corpus, "showC", "e1", admap={"chars_removed": 10, "source_length": 100})
    d = corpus / "feeds" / "showC" / "run_1" / "transcripts"
    (d / "e1.adfree.txt").write_text("cleaned", encoding="utf-8")
    (d / "e1.segments.txt").write_text("segments", encoding="utf-8")

    assert mod.collect(corpus)["showC"]["episodes"] == 1


def test_a_feed_far_below_the_median_is_surfaced(tmp_path: Path) -> None:
    """The actionable signal: one feed the detector is missing, against healthy peers."""
    mod = _load()
    corpus = tmp_path / "corpus"
    for feed in ("healthy1", "healthy2"):
        for i in range(4):
            _episode(corpus, feed, f"e{i}", admap={"chars_removed": 900, "source_length": 40000})
    for i in range(4):
        _episode(corpus, "silent", f"e{i}", admap={"chars_removed": 0, "source_length": 40000})

    text = "\n".join(mod.report(mod.collect(corpus)))
    assert "FAR BELOW" in text
    assert "silent" in text
    assert "healthy1" not in text.split("FAR BELOW")[1]


def test_a_uniformly_healthy_corpus_reports_no_suspects(tmp_path: Path) -> None:
    mod = _load()
    corpus = tmp_path / "corpus"
    for feed in ("a", "b"):
        for i in range(4):
            _episode(corpus, feed, f"e{i}", admap={"chars_removed": 700, "source_length": 40000})

    assert "No feed sits far below" in "\n".join(mod.report(mod.collect(corpus)))


def test_a_corrupt_sidecar_does_not_take_the_audit_down(tmp_path: Path) -> None:
    """An audit that dies on one bad file cannot be run against a real corpus."""
    mod = _load()
    corpus = tmp_path / "corpus"
    _episode(corpus, "showD", "e1", admap={"chars_removed": 5, "source_length": 100})
    d = corpus / "feeds" / "showD" / "run_1" / "transcripts"
    (d / "e2.adfree.admap.json").write_text("{not json", encoding="utf-8")

    assert mod.collect(corpus)["showD"]["episodes"] == 1


def test_missing_corpus_root_exits_two(tmp_path: Path) -> None:
    mod = _load()
    assert mod.main([str(tmp_path / "nope")]) == 2
