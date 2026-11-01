"""The ledger step respells the episode files to each build's group spellings (RFC-131 0041).

A three-episode corpus built with m0025's fixture: one person spelled two ways on one show, the
show's text naming one spelling. Invented names only.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from podcast_scraper.identity import ledger as L, respell as R
from podcast_scraper.identity.ledger_cli import parse_identity_ledger_argv, run_identity_ledger_cli

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture",
    Path(__file__).parents[1] / "upgrade" / "test_m0025_one_person_one_entry.py",
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def _episode(root: Path, n: int, host: str) -> dict:
    tmp = root.parent / f"build_{n}"
    p = fx._corpus(
        tmp,
        speakers=[
            fx._speaker(host, "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Odile Brand", "guest", ["SPEAKER_01"], "llm", "guest"),
        ],
        turns=[
            ("SPEAKER_00", host, "host", "Welcome to the show."),
            ("SPEAKER_01", "Odile Brand", "guest", "Thanks for having me."),
        ],
        known_hosts=["Korin Vael"],
    )
    run = root / "feeds" / "show_a" / f"run_{n}_2026100{n}-000000"
    run.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p["meta"].parent.parent), str(run))
    shutil.rmtree(tmp)
    out = {k: run / v.relative_to(p["meta"].parent.parent) for k, v in p.items()}
    meta = fx._r(out["meta"])
    meta["episode"]["episode_id"] = f"e{n}"
    meta["feed"] = {"title": "show_a", "language": "en"}
    fx._w(out["meta"], meta)
    kg = fx._r(out["kg"])
    kg["episode_id"] = f"e{n}"
    kg["nodes"].append(
        {"id": f"episode:e{n}", "type": "Episode", "properties": {"podcast_id": "show_a"}}
    )
    fx._w(out["kg"], kg)
    return out


@pytest.fixture()
def corpus(tmp_path: Path):
    root = tmp_path / "corpus"
    eps = {
        1: _episode(root, 1, "Korin Vael"),
        2: _episode(root, 2, "Korin Vaal"),
        3: _episode(root, 3, "Korin Vael"),
    }
    return root, eps


def _changes(root: Path) -> list:
    return [json.loads(x) for x in (root / L.CHANGES_NAME).read_text().splitlines()]


def test_a_build_respells_the_files_and_a_second_build_writes_nothing(corpus) -> None:
    root, eps = corpus
    text_before = eps[2]["txt"].read_text(encoding="utf-8")
    assert L.refresh_ledger(root, after="test")
    assert L.served_id_map(root) == {"person:korin-vaal": "person:korin-vael"}
    assert eps[2]["txt"].read_text(encoding="utf-8").startswith("Korin Vael: Welcome")
    assert fx._r(eps[2]["seg"])[0]["speaker_label"] == "Korin Vael"
    kg = {n["id"]: n["properties"].get("name") for n in fx._r(eps[2]["kg"])["nodes"]}
    assert kg["person:korin-vaal"] == "Korin Vael"  # the id stays
    respell = _changes(root)[-1]["respell"]
    assert (respell["written"], respell["refused"], respell["deferred"]) == (1, 0, False)
    assert eps[1]["txt"].read_text(encoding="utf-8").startswith("Korin Vael: Welcome")
    assert L.refresh_ledger(root, after="test")
    assert _changes(root)[-1]["respell"]["written"] == 0
    assert L.served_id_map(root) == {"person:korin-vaal": "person:korin-vael"}  # the fixpoint
    restored, refused = R.undo_respell(root, respell["stamp"])
    assert restored and not refused
    assert eps[2]["txt"].read_text(encoding="utf-8") == text_before


def test_a_held_lock_defers_the_respelling_and_the_next_build_catches_up(
    corpus, monkeypatch
) -> None:
    root, eps = corpus

    @contextlib.contextmanager
    def held(*_a, **_k):
        raise RuntimeError("held by pid 1")
        yield  # pragma: no cover

    monkeypatch.setattr("podcast_scraper.utils.corpus_lock.corpus_parent_lock", held)
    assert L.refresh_ledger(root, after="test")  # the ledger is still written
    assert _changes(root)[-1]["respell"]["deferred"] is True
    assert fx._r(eps[2]["seg"])[0]["speaker_label"] == "Korin Vaal"
    monkeypatch.undo()
    assert L.refresh_ledger(root, after="test")
    assert _changes(root)[-1]["respell"]["written"] == 1
    assert fx._r(eps[2]["seg"])[0]["speaker_label"] == "Korin Vael"


def test_a_failing_respelling_never_blocks_the_ledger(corpus, monkeypatch) -> None:
    root, eps = corpus

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(R, "_respell_locked", boom)
    assert L.refresh_ledger(root, after="test")
    assert "OSError" in _changes(root)[-1]["respell"]["failed"]
    assert L.served_id_map(root) == {"person:korin-vaal": "person:korin-vael"}


def test_the_cli_respells_unless_told_not_to(corpus, capsys) -> None:
    root, eps = corpus
    args = parse_identity_ledger_argv(["--output-dir", str(root), "--no-respell"])
    assert run_identity_ledger_cli(args, __import__("logging").getLogger("t")) == 0
    assert fx._r(eps[2]["seg"])[0]["speaker_label"] == "Korin Vaal"
    assert "respell" not in json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    args = parse_identity_ledger_argv(["--output-dir", str(root)])
    assert run_identity_ledger_cli(args, __import__("logging").getLogger("t")) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["respell"]["written"] == 1
    assert fx._r(eps[2]["seg"])[0]["speaker_label"] == "Korin Vael"


def test_the_ledger_reads_the_episodes_own_spelling_after_a_respell(corpus) -> None:
    """Evidence and presentation stay apart (record 0041, 4g): the files show the group spelling,
    the ledger reads what each episode said, so a rebuild after a respell is the same ledger."""
    from podcast_scraper.identity.guards import collect_episode_evidence
    from podcast_scraper.kg.entity_clusters import collect_entity_candidates

    root, eps = corpus
    names_before = {
        k: (c.name, dict(c._name_counts)) for k, c in collect_entity_candidates(root).items()
    }
    evidence_before = collect_episode_evidence(root)
    assert L.refresh_ledger(root, after="test")
    map_before = L.served_id_map(root)
    kg = {n["id"]: n["properties"] for n in fx._r(eps[2]["kg"])["nodes"] if n["type"] == "Person"}
    assert kg["person:korin-vaal"]["name"] == "Korin Vael"  # shown respelled
    assert kg["person:korin-vaal"]["respelled_from"] == "Korin Vaal"  # the episode's own kept
    roster = fx._r(eps[2]["meta"])["content"]["speakers"]
    assert {(s["name"], s.get("respelled_from")) for s in roster} >= {("Korin Vael", "Korin Vaal")}
    names_after = {
        k: (c.name, dict(c._name_counts)) for k, c in collect_entity_candidates(root).items()
    }
    assert names_after == names_before
    assert collect_episode_evidence(root) == evidence_before
    assert L.ledger_id_map(L.build_ledger(root)) == map_before
