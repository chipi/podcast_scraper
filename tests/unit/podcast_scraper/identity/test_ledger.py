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
        # An English show, as every prod feed declares one (S5b: untagged joins by equality only).
        json.dumps({"episode": {"episode_id": ep}, "feed": {"title": feed, "language": "en"}}),
        encoding="utf-8",
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


def test_where_nothing_refuses_the_ledger_records_todays_map(corpus: Path) -> None:
    """S4 changes joins only where its rule or a guard refuses; this corpus has neither."""
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


def test_a_write_keeps_the_previous_ledger_and_restore_puts_it_back(corpus: Path) -> None:
    first = L.build_ledger(corpus)
    L.write_ledger(corpus, first)
    assert not (corpus / L.PREVIOUS_NAME).exists()  # nothing valid to keep yet
    second = dict(first, clusters_payload={"clusters": []}, built_at="later")
    L.write_ledger(corpus, second)
    kept = json.loads((corpus / L.PREVIOUS_NAME).read_text(encoding="utf-8"))
    assert kept["built_at"] == first["built_at"]
    assert L.served_id_map(corpus) == {}
    change = L.restore_previous(corpus)
    assert change["restored_from"] == L.PREVIOUS_NAME and change["joins_added"] == 1
    assert L.served_id_map(corpus) == L.ledger_id_map(first)
    assert len((corpus / L.CHANGES_NAME).read_text(encoding="utf-8").splitlines()) == 3


def test_restore_refuses_without_a_valid_previous_ledger(corpus: Path, capsys) -> None:
    import logging

    L.write_ledger(corpus, L.build_ledger(corpus))
    with pytest.raises(FileNotFoundError):
        L.restore_previous(corpus)
    (corpus / L.PREVIOUS_NAME).write_text('{"schema_version": "other"}', encoding="utf-8")
    with pytest.raises(ValueError):
        L.restore_previous(corpus)
    args = parse_identity_ledger_argv(["--output-dir", str(corpus), "--restore-previous"])
    assert run_identity_ledger_cli(args, logging.getLogger("t")) == 1
    assert L.read_ledger(corpus)[1] == L.STATE_OK  # the current ledger is untouched


# ---- S5a: a ledger of the previous rule is still served, and says it needs a rebuild -------------


def _as_previous_rule(corpus: Path) -> dict:
    ledger = dict(L.build_ledger(corpus), rule_version=L.PREVIOUS_RULE_VERSION)
    (corpus / L.LEDGER_NAME).write_text(json.dumps(ledger), encoding="utf-8")
    return ledger


def test_a_previous_rule_ledger_is_served_until_the_rebuild(corpus: Path) -> None:
    """D2's deploy: the new image reads D1's ledger and serves its joins, not none."""
    assert L.PREVIOUS_RULE_VERSION != L.RULE_VERSION
    old = _as_previous_rule(corpus)
    ledger, state = L.read_ledger(corpus)
    assert state == L.STATE_OK_PREVIOUS_RULE and ledger is not None
    assert (
        L.served_id_map(corpus)
        == L.ledger_id_map(old)
        == {"person:korin-vaal": "person:korin-vael"}
    )
    status = L.ledger_status(corpus)
    assert status["state"] == L.STATE_OK_PREVIOUS_RULE and status["rebuild_needed"] is True
    assert L.ledger_status(corpus.parent / "none")["state"] == L.STATE_MISSING


def test_the_rebuild_keeps_the_previous_rule_ledger_for_the_undo(corpus: Path) -> None:
    old = _as_previous_rule(corpus)
    change = L.write_ledger(corpus, L.build_ledger(corpus))
    assert (change["joins_added"], change["joins_removed"], change["canonical_changed"]) == (
        0,
        0,
        0,
    )
    kept = json.loads((corpus / L.PREVIOUS_NAME).read_text(encoding="utf-8"))
    assert kept["rule_version"] == L.PREVIOUS_RULE_VERSION == old["rule_version"]
    assert L.read_ledger(corpus)[1] == L.STATE_OK
    assert L.ledger_status(corpus)["rebuild_needed"] is False
    L.restore_previous(corpus)
    assert L.read_ledger(corpus)[1] == L.STATE_OK_PREVIOUS_RULE


def test_an_older_rule_ledger_still_means_no_joins(corpus: Path) -> None:
    ledger = dict(L.build_ledger(corpus), rule_version="cluster-person-no-bridge-guarded/1")
    (corpus / L.LEDGER_NAME).write_text(json.dumps(ledger), encoding="utf-8")
    assert L.read_ledger(corpus) == (None, L.STATE_VERSION_MISMATCH)
    assert L.served_id_map(corpus) == {}


def test_the_state_gauge_knows_every_state() -> None:
    assert set(L.LEDGER_STATES) == {
        L.STATE_OK,
        L.STATE_OK_PREVIOUS_RULE,
        L.STATE_MISSING,
        L.STATE_UNREADABLE,
        L.STATE_VERSION_MISMATCH,
    }


def test_cli_dry_run_names_a_previous_rule_ledger(corpus: Path, capsys) -> None:
    import logging

    _as_previous_rule(corpus)
    args = parse_identity_ledger_argv(["--output-dir", str(corpus), "--dry-run"])
    assert run_identity_ledger_cli(args, logging.getLogger("t")) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["previous_state"] == L.STATE_OK_PREVIOUS_RULE


# ---- S5b: candidates carry their episodes' languages ---------------------------------------------


def _lang_episode(root: Path, feed: str, ep: str, persons: Dict[str, str], language) -> None:
    _kg_episode(root, feed, f"run_{ep}_20261001-000000", ep, persons)
    meta = next((root / "feeds" / feed).glob(f"run_{ep}_*/metadata/{ep}.metadata.json"))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    if language is None:
        doc["feed"].pop("language", None)
    else:
        doc["feed"]["language"] = language
    meta.write_text(json.dumps(doc), encoding="utf-8")


def test_candidates_carry_the_episode_language(tmp_path: Path) -> None:
    root = tmp_path / "c"
    _lang_episode(root, "show_de", "d1", {"person:joern-bruell": "Joern Bruell"}, "de-DE")
    _lang_episode(root, "show_de", "d2", {"person:jorn-brull": "Jörn Brüll"}, "de")
    _lang_episode(root, "show_x", "x1", {"person:korin-vael": "Korin Vael"}, None)
    _lang_episode(root, "show_x", "x2", {"person:korin-vaal": "Korin Vaal"}, None)
    cands = entity_clusters.collect_entity_candidates(root)
    assert cands["person:joern-bruell"].languages == {"de"}
    assert cands["person:korin-vael"].languages == {"und"}
    joins = L.ledger_id_map(L.build_ledger(root))
    # de: the fold joins the two spellings; und: two different spellings stay apart
    assert joins.get("person:jorn-brull", joins.get("person:joern-bruell")) in {
        "person:joern-bruell",
        "person:jorn-brull",
    }
    assert len([k for k in joins if k in {"person:jorn-brull", "person:joern-bruell"}]) == 1
    assert "person:korin-vaal" not in joins and "person:korin-vael" not in joins


def test_an_org_of_unknown_language_still_joins(tmp_path: Path) -> None:
    root = tmp_path / "c"
    _lang_episode(root, "show_x", "x1", {}, None)
    meta = root / "feeds" / "show_x" / "run_x1_20261001-000000" / "metadata" / "x1.kg.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["nodes"] += [
        {"id": "org:halloway-labs", "type": "Organization", "properties": {"name": "Halloway Labs"}}
    ]
    meta.write_text(json.dumps(doc), encoding="utf-8")
    _lang_episode(root, "show_x", "x2", {}, None)
    meta2 = root / "feeds" / "show_x" / "run_x2_20261001-000000" / "metadata" / "x2.kg.json"
    doc2 = json.loads(meta2.read_text(encoding="utf-8"))
    doc2["nodes"] += [
        {"id": "org:haloway-labs", "type": "Organization", "properties": {"name": "Haloway Labs"}}
    ]
    meta2.write_text(json.dumps(doc2), encoding="utf-8")
    assert entity_clusters.collect_entity_candidates(root)["org:halloway-labs"].languages == {"und"}
    assert L.ledger_id_map(L.build_ledger(root)) in (
        {"org:haloway-labs": "org:halloway-labs"},
        {"org:halloway-labs": "org:haloway-labs"},
    )
