"""The identity ledger (ADR-166): what it records, how it is read; a failure means no joins."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.identity import ledger as L
from podcast_scraper.identity.ledger_cli import parse_identity_ledger_argv, run_identity_ledger_cli
from podcast_scraper.kg import entity_clusters


def _kg_episode(root: Path, feed: str, run: str, ep: str, persons: Dict[str, str]) -> None:
    meta = root / "feeds" / feed / run / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{ep}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": ep}, "feed": {"title": feed}}), encoding="utf-8"
    )
    nodes = [{"id": f"episode:{ep}", "type": "Episode", "properties": {"podcast_id": feed}}]
    nodes += [
        {"id": pid, "type": "Person", "properties": {"name": n}} for pid, n in persons.items()
    ]
    (meta / f"{ep}.kg.json").write_text(
        json.dumps({"episode_id": ep, "nodes": nodes, "edges": []}), encoding="utf-8"
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    # Synthetic names: one person spelled two ways on one show (joined), and a stranger.
    _kg_episode(root, "show_a", "run_x_20261001-000000", "e1", {"person:korin-vael": "Korin Vael"})
    _kg_episode(root, "show_a", "run_y_20261002-000000", "e2", {"person:korin-vaal": "Korin Vaal"})
    _kg_episode(root, "show_a", "run_z_20261003-000000", "e3", {"person:korin-vael": "Korin Vael"})
    _kg_episode(
        root, "show_b", "run_w_20261004-000000", "e4", {"person:odile-brand": "Odile Brand"}
    )
    entity_clusters._ID_MAP_NS  # noqa: B018 - the cache namespace exists
    return root


def test_the_ledger_records_exactly_todays_map(corpus: Path) -> None:
    want = entity_clusters.build_entity_id_map(corpus)
    assert want == {"person:korin-vaal": "person:korin-vael"}
    ledger = L.build_ledger(corpus)
    assert ledger["schema_version"] == L.LEDGER_SCHEMA_VERSION
    assert ledger["rule_version"] == L.RULE_VERSION
    assert L.ledger_id_map(ledger) == want
    L.write_ledger(corpus, ledger)
    assert L.served_id_map(corpus) == want


@pytest.mark.parametrize(
    "content,state",
    [
        (None, L.STATE_MISSING),
        ("{not json", L.STATE_UNREADABLE),
        ('["a list"]', L.STATE_UNREADABLE),
        (
            json.dumps({"schema_version": "identity-ledger/0", "rule_version": L.RULE_VERSION}),
            L.STATE_VERSION_MISMATCH,
        ),
        (
            json.dumps({"schema_version": L.LEDGER_SCHEMA_VERSION, "rule_version": "old/0"}),
            L.STATE_VERSION_MISMATCH,
        ),
    ],
)
def test_a_missing_or_bad_ledger_means_no_joins(corpus: Path, content, state) -> None:
    if content is not None:
        (corpus / L.LEDGER_NAME).write_text(content, encoding="utf-8")
    assert L.read_ledger(corpus) == (None, state)
    # Never the unguarded in-memory build, which would join here.
    assert entity_clusters.build_entity_id_map(corpus)
    assert L.served_id_map(corpus) == {}


def test_the_write_is_atomic_and_the_change_log_records_each_build(corpus: Path) -> None:
    first = L.write_ledger(corpus, L.build_ledger(corpus))
    assert (first["joins"], first["joins_added"], first["joins_removed"]) == (1, 1, 0)
    assert first["added"] == [["person:korin-vaal", None, "person:korin-vael"]]
    # A second spelling of the stranger's show changes nothing about the first join.
    second = L.write_ledger(corpus, L.build_ledger(corpus))
    assert (second["joins_added"], second["joins_removed"], second["canonical_changed"]) == (
        0,
        0,
        0,
    )
    lines = (corpus / L.CHANGES_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert not [p for p in corpus.iterdir() if p.name.startswith(".identity_ledger.")]


def test_the_ledger_moves_the_projection_token_but_not_its_own(corpus: Path) -> None:
    (corpus / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    old = time.time() - 100
    os.utime(corpus / "corpus_manifest.json", (old, old))
    content_before = perf_cache.corpus_content_mtime(corpus)
    projections_before = perf_cache.corpus_mtime(corpus)
    L.write_ledger(corpus, L.build_ledger(corpus))
    assert perf_cache.corpus_content_mtime(corpus) == content_before
    assert perf_cache.corpus_mtime(corpus) > projections_before


def test_status_reports_a_corpus_newer_than_its_ledger(corpus: Path) -> None:
    assert L.ledger_status(corpus) == {"state": L.STATE_MISSING}
    L.write_ledger(corpus, L.build_ledger(corpus))
    status = L.ledger_status(corpus)
    assert status["state"] == L.STATE_OK and status["corpus_newer_than_ledger"] is False
    _kg_episode(corpus, "show_a", "run_v_20261005-000000", "e5", {"person:x-y": "Xan Yull"})
    (corpus / "corpus_run_summary.json").write_text("{}", encoding="utf-8")
    later = time.time() + 5
    os.utime(corpus / "corpus_run_summary.json", (later, later))
    status = L.ledger_status(corpus)
    assert status["corpus_newer_than_ledger"] is True
    assert status["kg_episodes_since_build"] == 1


def test_the_server_serves_the_ledger_and_picks_up_a_rebuild(corpus: Path) -> None:
    assert entity_clusters.cached_entity_id_map(corpus) == {}  # no ledger yet: no joins
    L.write_ledger(corpus, L.build_ledger(corpus))
    assert entity_clusters.cached_entity_id_map(corpus) == {
        "person:korin-vaal": "person:korin-vael"
    }


def test_cli_dry_run_writes_nothing_and_status_reads(corpus: Path, capsys) -> None:
    import logging

    log = logging.getLogger("t")
    assert (
        run_identity_ledger_cli(
            parse_identity_ledger_argv(["--output-dir", str(corpus), "--dry-run"]), log
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["joins_added"] == 1
    assert not (corpus / L.LEDGER_NAME).exists()
    assert (
        run_identity_ledger_cli(parse_identity_ledger_argv(["--output-dir", str(corpus)]), log) == 0
    )
    capsys.readouterr()
    assert (corpus / L.LEDGER_NAME).exists()
    assert (
        run_identity_ledger_cli(
            parse_identity_ledger_argv(["--output-dir", str(corpus), "--status"]), log
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["state"] == L.STATE_OK
    assert (
        run_identity_ledger_cli(
            parse_identity_ledger_argv(["--output-dir", str(corpus / "missing")]), log
        )
        == 2
    )
