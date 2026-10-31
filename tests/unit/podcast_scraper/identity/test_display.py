"""One spelling per person on the insight surfaces, from the ledger (RFC-131 S6a).

Invented names only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from podcast_scraper.identity import display as D, ledger as L
from podcast_scraper.server.app_gi_view import insights_from_gi

pytestmark = pytest.mark.unit


def _episode(root: Path, ep: str, person: str, name: str, publisher: List[str]) -> Dict:
    meta = root / "feeds" / "show_a" / f"run_{ep}_20261001-000000" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{ep}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": ep, "title": ep},
                "feed": {"title": "show_a", "language": "en"},
                "content": {"speakers": [], "transcript_file_path": f"transcripts/{ep}.txt"},
            }
        ),
        encoding="utf-8",
    )
    if publisher:
        tr = meta.parent / "transcripts"
        tr.mkdir(exist_ok=True)
        (tr / f"{ep}.speakers.diagnostics.json").write_text(
            json.dumps({"tried": {"known_hosts": publisher}}), encoding="utf-8"
        )
    episode = {"id": f"episode:{ep}", "type": "Episode", "properties": {"podcast_id": "show_a"}}
    pnode = {"id": person, "type": "Person", "properties": {"name": name}}
    (meta / f"{ep}.kg.json").write_text(
        json.dumps({"episode_id": ep, "nodes": [episode, pnode], "edges": []}), encoding="utf-8"
    )
    gi = {
        "nodes": [
            pnode,
            {
                "id": f"insight:{ep}",
                "type": "Insight",
                "properties": {"text": "A point.", "tier": 3},
            },
            {
                "id": f"quote:{ep}",
                "type": "Quote",
                "properties": {"text": "Said so.", "speaker_id": person},
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": f"insight:{ep}", "to": f"quote:{ep}"}],
    }
    (meta / f"{ep}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    return gi


@pytest.fixture()
def corpus(tmp_path: Path):
    root = tmp_path / "c"
    # One person: spelled "Korin Vaal" in two episodes, "Korin Vael" in one; the show's text says
    # "Korin Vael". Frequency would show "Korin Vaal"; publisher text first shows "Korin Vael".
    gis = {
        "e1": _episode(root, "e1", "person:korin-vaal", "Korin Vaal", []),
        "e2": _episode(root, "e2", "person:korin-vaal", "Korin Vaal", []),
        "e3": _episode(root, "e3", "person:korin-vael", "Korin Vael", ["Korin Vael"]),
    }
    return root, gis


def _speakers(gi, people=None):
    return [q.speaker for i in insights_from_gi(gi, people=people) for q in i.quotes]


def test_the_ledger_records_the_group_spelling_and_its_evidence(corpus) -> None:
    root, _ = corpus
    ledger = L.build_ledger(root)
    (cluster,) = [
        c for c in ledger["clusters_payload"]["clusters"] if c["canonical_id"].startswith("person:")
    ]
    assert cluster["display_name"] == "Korin Vael"
    assert cluster["display_evidence"] == {"rule": "publisher", "episodes": 1}


def test_every_episode_shows_the_group_spelling(corpus) -> None:
    root, gis = corpus
    L.write_ledger(root, L.build_ledger(root))
    people = D.person_display(root)
    for gi in gis.values():
        assert _speakers(gi, people) == ["Korin Vael"]
    assert _speakers(gis["e1"]) == ["Korin Vaal"]  # without the view: today's name


def test_no_ledger_shows_each_episodes_own_name(corpus) -> None:
    root, gis = corpus
    people = D.person_display(root)
    assert _speakers(gis["e1"], people) == ["Korin Vaal"]
    assert _speakers(gis["e3"], people) == ["Korin Vael"]


def test_a_previous_rule_ledger_without_spellings_changes_nothing(corpus) -> None:
    root, gis = corpus
    ledger = L.build_ledger(root)
    for c in ledger["clusters_payload"]["clusters"]:
        c.pop("display_name", None)
        c.pop("display_evidence", None)
    ledger["rule_version"] = L.PREVIOUS_RULE_VERSION
    (root / L.LEDGER_NAME).write_text(json.dumps(ledger), encoding="utf-8")
    people = D.person_display(root)
    assert [_speakers(g, people) for g in gis.values()] == [_speakers(g) for g in gis.values()]


def test_a_rebuilt_ledger_is_picked_up_without_a_restart(corpus) -> None:
    root, gis = corpus
    assert _speakers(gis["e1"], D.person_display(root)) == ["Korin Vaal"]
    L.write_ledger(root, L.build_ledger(root))
    assert _speakers(gis["e1"], D.person_display(root)) == ["Korin Vael"]


def test_an_unjoined_person_keeps_its_own_name(corpus) -> None:
    root, _ = corpus
    L.write_ledger(root, L.build_ledger(root))
    assert D.person_display(root).shown("person:odile-brand", "Odile Brand") == "Odile Brand"
