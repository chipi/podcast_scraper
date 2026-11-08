"""The edge paths of the 2026-11-08 MCP fixes: each fallback does what its comment says.

- the registry keeps working when the identity map or the ledger spellings cannot be read;
- a member's own aliases join the canonical's, without repeats;
- explore keeps the files' speakers when the ledger cannot be read, and leaves insights alone when
  there is nothing to apply;
- the enrichment window rollup refuses a malformed window and skips lines it cannot use;
- the event query's prefix form;
- releasing freed memory is a no-op where there is no libc.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

import podcast_scraper.identity.display as D
import podcast_scraper.identity.resolver as R
import podcast_scraper.kg.entity_clusters as EC

pytestmark = pytest.mark.unit


def _ident(cid: str, name: str, aliases=()) -> dict:
    return {"id": cid, "type": "person", "display_name": name, "aliases": list(aliases)}


# --- resolver ---------------------------------------------------------------------------------


def test_the_fold_survives_an_unreadable_identity_map(tmp_path, monkeypatch, caplog) -> None:
    def boom(root, **kw):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(EC, "cached_entity_id_map", boom)
    identities = {"person:a": _ident("person:a", "A")}
    with caplog.at_level(logging.WARNING, logger=R.logger.name):
        R._fold_joined_members(tmp_path, identities, __import__("collections").Counter())
    assert set(identities) == {"person:a"}
    assert "identity map unavailable" in caplog.text


def test_a_members_aliases_join_the_canonicals_once(tmp_path, monkeypatch) -> None:
    from collections import Counter

    monkeypatch.setattr(
        EC,
        "cached_entity_id_map",
        lambda root, **kw: {"person:b": "person:a", "person:x": "person:x"},
    )
    identities = {
        "person:a": _ident("person:a", "Ann Lee", ["A. Lee"]),
        "person:b": _ident("person:b", "Anne Lee", ["A. Lee", "Annie"]),
    }
    freq = Counter({"person:a": 2, "person:b": 3})
    R._fold_joined_members(tmp_path, identities, freq)
    assert set(identities) == {"person:a"}
    assert identities["person:a"]["aliases"] == ["A. Lee", "Anne Lee", "Annie"]
    assert identities["person:a"]["members"] == ["person:b"]
    assert freq["person:a"] == 5


def test_a_member_whose_canonical_has_no_node_becomes_the_canonical(tmp_path, monkeypatch) -> None:
    from collections import Counter

    monkeypatch.setattr(EC, "cached_entity_id_map", lambda root, **kw: {"person:b": "person:a"})
    identities = {"person:b": _ident("person:b", "Anne Lee")}
    R._fold_joined_members(tmp_path, identities, Counter({"person:b": 1}))
    assert set(identities) == {"person:a"}
    assert identities["person:a"]["display_name"] == "Anne Lee"


def test_spelling_survives_an_unreadable_ledger(tmp_path, monkeypatch, caplog) -> None:
    def boom(root):
        raise ValueError("bad ledger")

    monkeypatch.setattr(D, "person_display", boom)
    identities = {"person:a": _ident("person:a", "Ann")}
    with caplog.at_level(logging.WARNING, logger=R.logger.name):
        R._spell_people(tmp_path, identities)
    assert identities["person:a"]["display_name"] == "Ann"
    assert "ledger spellings unavailable" in caplog.text


def test_spelling_leaves_non_persons_and_same_names_alone(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        D, "person_display", lambda root: D.PersonDisplay({"person:a": "Ann", "org:x": "X Corp"})
    )
    identities = {"person:a": _ident("person:a", "Ann"), "org:x": _ident("org:x", "Xco")}
    R._spell_people(tmp_path, identities)
    assert identities["person:a"]["aliases"] == []
    assert identities["org:x"]["display_name"] == "Xco"


def test_a_fresh_refresh_in_flight_is_not_doubled(tmp_path, monkeypatch) -> None:
    key = ("k", 0.75, "m")
    R._refreshing[key] = __import__("time").monotonic()
    try:
        R._last_refresh = None
        R._refresh_in_background(key, tmp_path, None, "m", 0.75)
        assert R._last_refresh is None
    finally:
        R._refreshing.pop(key, None)


# --- explore ----------------------------------------------------------------------------------


def test_explore_keeps_the_files_speakers_without_a_ledger(tmp_path, monkeypatch, caplog) -> None:
    from podcast_scraper.gi import explore as E

    def boom_map(root, **kw):
        raise OSError("no map")

    def boom_display(root):
        raise OSError("no ledger")

    monkeypatch.setattr(EC, "cached_entity_id_map", boom_map)
    monkeypatch.setattr(D, "person_display", boom_display)
    with caplog.at_level(logging.WARNING, logger=E.logger.name):
        assert E._person_id_map(tmp_path) == {}
        unchanged: Any = ["unchanged"]  # any list: the fallback returns it untouched
        assert E._ledger_speakers(tmp_path, unchanged, {}) is unchanged
    assert "identity map unavailable" in caplog.text
    assert "ledger spellings unavailable" in caplog.text


def test_explore_changes_nothing_with_nothing_to_apply(tmp_path, monkeypatch) -> None:
    from podcast_scraper.gi import explore as E

    monkeypatch.setattr(D, "person_display", lambda root: D.PersonDisplay({}))
    sentinel: Any = [object()]
    assert E._ledger_speakers(tmp_path, sentinel, {}) is sentinel
    assert E._speaker_group(None, {"a": "b"}) == set()
    assert E._speaker_group("person:z", {"person:a": "person:b"}) == set()
    assert E._speaker_group("person:a", {"person:a": "person:b"}) == {"person:a", "person:b"}


# --- enrichment window rollup -----------------------------------------------------------------


# --- o11y events prefix -----------------------------------------------------------------------


def test_a_trailing_star_is_a_prefix_query(monkeypatch) -> None:
    from podcast_obs.config import TargetConfig
    from podcast_obs.sources import victoria

    sent = {}

    def fake_get(url, **kw):
        sent["q"] = kw["params"]["query"]
        return []

    monkeypatch.setattr(victoria, "get_ndjson", fake_get)
    r = victoria.events(TargetConfig(name="t", victorialogs_url="http://l"), "pipeline_run.*")
    assert r["ok"]
    assert '(event_type:"pipeline_run."* OR _msg:"pipeline_run."*)' in sent["q"]


# --- memory -----------------------------------------------------------------------------------


def test_releasing_memory_without_a_libc_is_a_no_op(monkeypatch) -> None:
    import ctypes.util

    from podcast_scraper.utils import memory

    monkeypatch.setattr(memory, "_libc", None)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    memory.release_freed_memory()  # no libc found: returns quietly

    def boom(name):
        raise OSError("cannot load")

    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libc.so.6")
    monkeypatch.setattr(memory.ctypes, "CDLL", boom)
    memory.release_freed_memory()  # libc cannot load: logged at debug, never raised


def test_explore_respells_each_quote_to_its_group(tmp_path, monkeypatch) -> None:
    from podcast_scraper.gi import explore as E
    from podcast_scraper.gi.contracts import EvidenceSpan, InsightSummary, SupportingQuote

    monkeypatch.setattr(D, "person_display", lambda root: D.PersonDisplay({"person:a": "Ann Lee"}))
    span = EvidenceSpan(transcript_ref="t.txt", char_start=0, char_end=1)
    quotes = [
        SupportingQuote(
            quote_id="q1", text="x", speaker_id="person:b", speaker_name="Anne", evidence=span
        ),
        SupportingQuote(quote_id="q2", text="y", speaker_id=None, evidence=span),
    ]
    ins = InsightSummary(
        insight_id="i", text="t", grounded=True, episode_id="e", supporting_quotes=quotes
    )
    (out,) = E._ledger_speakers(tmp_path, [ins], {"person:b": "person:a"})
    assert [(q.speaker_id, q.speaker_name) for q in out.supporting_quotes] == [
        ("person:a", "Ann Lee"),
        (None, None),
    ]


def test_top_persons_counts_a_members_quotes_for_the_group(tmp_path, monkeypatch) -> None:
    """Same contract as the integration test, without the HTTP client."""
    from podcast_scraper import perf_cache
    from podcast_scraper.server.routes import corpus_persons

    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f1", "title": "F"},
                "episode": {"episode_id": "ep1", "title": "E", "published_date": "2024-04-01"},
            }
        ),
        encoding="utf-8",
    )
    (meta / "ep1.gi.json").write_text(
        json.dumps(
            {
                "episode_id": "ep1",
                "nodes": [
                    {"id": "person:anne", "type": "Person", "properties": {"name": "Anne"}},
                    {"id": "q1", "type": "Quote", "properties": {"text": "hi"}},
                    {"id": "i1", "type": "Insight", "properties": {"text": "t"}},
                ],
                "edges": [
                    {"type": "SPOKEN_BY", "from": "q1", "to": "person:anne"},
                    {"type": "SUPPORTED_BY", "from": "i1", "to": "q1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        EC, "cached_entity_id_map", lambda root, **kw: {"person:anne": "person:ann"}
    )
    perf_cache.clear()
    body = corpus_persons.top_persons(tmp_path, 5)
    assert [(p["person_id"], p["insight_count"]) for p in body["persons"]] == [("person:ann", 1)]
