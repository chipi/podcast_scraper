"""m0031: 0029's voices through the fixed rewrite — named where a reprocess unnamed them again,
shown where 0029 left them hidden under their name (#2380).

Synthetic episodes built with m0025's fixture; the prod list is replaced per test with monkeypatch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import (
    m0029_naming_arc_voices_repaired as m29,
    m0031_named_voices_shown as m31,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

ENTRY = {("ep", "SPEAKER_01"): (None, "unknown", "Steve McConnell", "guest", "the guest")}


def _episode(tmp_path: Path, label: str, role: str, typed: bool):
    """Host + guest; the guest's voice carries ``label``/``role``, typed and hidden if ``typed``."""
    speakers = [fx._speaker("Gergely Orosz", "host", ["SPEAKER_00"], "known_hosts", "host")]
    if not label.startswith("SPEAKER_"):  # the record 0029 wrote carries the guest's entry
        speakers.append(fx._speaker(label, role, ["SPEAKER_01"], "roster", "guest"))
    p = fx._corpus(
        tmp_path,
        speakers=speakers,
        turns=[
            ("SPEAKER_00", "Gergely Orosz", "host", "So, Steve, welcome to the podcast."),
            ("SPEAKER_01", label, role, "Thanks for having me. I wrote the book."),
            ("SPEAKER_00", "Gergely Orosz", "host", "Tell me about the book."),
        ],
        known_hosts=["Gergely Orosz"],
    )
    rows = fx._r(p["seg"])
    gi = fx._r(p["gi"])
    hidden = {"speaker_voice_type": "unknown", "surfaceable": False} if typed else {}
    if typed:
        for r in rows:
            if r["speaker"] == "SPEAKER_01":
                r["voice_type"] = "unknown"
        for n in gi["nodes"]:
            if n["id"] == "quote:1":
                n["properties"]["speaker_voice_type"] = "unknown"
    gi["nodes"].append(
        {
            "id": "insight:guest",
            "type": "Insight",
            "properties": {
                "text": "The book matters.",
                "speaker": label,
                "grounded": True,
                "tier": 2,
                "routing_tag": "connect" if typed else "surface",
                "salience": 0.6667 if typed else 0.7667,
                **hidden,
            },
        }
    )
    gi["edges"].append({"type": "SUPPORTED_BY", "from": "insight:guest", "to": "quote:1"})
    fx._w(p["seg"], rows)
    fx._w(p["gi"], gi)
    return p


def _review(monkeypatch, tmp_path: Path, default: str = "CORRECT", **by_voice: str) -> Path:
    """Write a real verdicts file for the CURRENT list (one row per naming entry, ``default``
    unless ``by_voice`` says otherwise; ``SKIP`` leaves the entry unreviewed) and point 0031 at it.
    """
    d = tmp_path / "_verdicts"
    d.mkdir(exist_ok=True)
    names = m31._names(m29.NAMING_ARC_VOICES)
    rows = [
        "\t".join((eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read"))
        for (eid, voice), name in sorted(names.items())
        if (verdict := by_voice.get(voice, default)) != "SKIP"
    ]
    head = (
        f"# migration: {m31.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(names)}\n# reviewer: test\n# date: 2026-10-18\n"
    )
    path = d / f"{m31.MIGRATION_ID}.tsv"
    path.write_text(head + "\t".join(rv.COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(m31.NamedVoicesShownMigration, "_verdicts_dir", d)
    return path


def _mig() -> m31.NamedVoicesShownMigration:
    return m31.NamedVoicesShownMigration()


def _apply(root: Path):
    return _mig().apply(MigrationContext(corpus_root=root))


def _guest(p) -> dict:
    props: dict = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == "insight:guest")[
        "properties"
    ]
    return props


def test_registered_last_right_after_0030() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids[-1] == m31.MIGRATION_ID
    assert ids.index(m31.MIGRATION_ID) == ids.index("0030_adfree_cut_resynced") + 1
    versions = [m.to_version for m in get_migrations()][-2:]
    assert versions == ["2.7.24", "2.7.25"]
    assert m31.BACKUP_TAG != m29.BACKUP_TAG and m31.RECEIPTS_FILE != m29.RECEIPTS_FILE


def test_a_voice_0029_named_but_left_hidden_is_shown_and_keeps_every_label(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    text_before = p["txt"].read_text(encoding="utf-8")
    meta_before = fx._r(p["meta"])["content"]["speakers"]
    assert not _mig().verify(MigrationContext(corpus_root=tmp_path))[0]
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert res.details["totals"]["voices_shown"] == 1
    assert all("voice_type" not in r for r in fx._r(p["seg"]))
    assert {r["speaker_label"] for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01"} == {
        "Steve McConnell"
    }
    assert p["txt"].read_text(encoding="utf-8") == text_before
    assert fx._r(p["meta"])["content"]["speakers"] == meta_before
    guest = _guest(p)
    assert guest["speaker"] == "Steve McConnell"
    assert "surfaceable" not in guest and "speaker_voice_type" not in guest
    assert guest["routing_tag"] == "surface"
    quote = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == "quote:1")["properties"]
    assert "speaker_voice_type" not in quote
    fx._quotes_hold(p)
    assert _mig().verify(MigrationContext(corpus_root=tmp_path))[0]
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m31.undo(tmp_path)
    assert restored and not refused
    assert _guest(p)["surfaceable"] is False


def test_a_voice_a_reprocess_unnamed_again_is_named_and_shown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and res.details["totals"]["voices_named"] == 1
    assert "\nSteve McConnell: Thanks for having me." in p["txt"].read_text(encoding="utf-8")
    guest = _guest(p)
    assert (guest["speaker"], guest["routing_tag"]) == ("Steve McConnell", "surface")
    assert "surfaceable" not in guest


def test_a_named_untyped_voice_is_done_and_nothing_is_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=False)
    before = {k: p[k].read_bytes() for k in ("seg", "gi", "meta", "txt")}
    assert _apply(tmp_path).details["episodes"] == []
    assert {k: p[k].read_bytes() for k in before} == before
    assert _mig().verify(MigrationContext(corpus_root=tmp_path))[0]


def test_an_unnaming_entry_is_not_this_migrations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): ("Steve McConnell", "guest", None, None, "unnamed")},
    )
    _review(monkeypatch, tmp_path)
    _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    assert _apply(tmp_path).details["episodes"] == []


def test_a_record_that_moved_on_is_refused_untouched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Somebody Else", "guest", typed=True)
    before = p["seg"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "the replay read" in res.details["refused"][0]["why"]
    assert p["seg"].read_bytes() == before
    assert not _mig().verify(MigrationContext(corpus_root=tmp_path))[0]


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def _add_insights(p, specs) -> None:
    """``specs``: (insight id, [quote ids in SUPPORTED_BY order], voice type or None)."""
    gi = fx._r(p["gi"])
    for iid, quotes, vt in specs:
        props = {
            "text": iid,
            "grounded": True,
            "tier": 2,
            "routing_tag": "connect",
            "salience": 0.6667,
        }
        if vt:
            props.update(speaker_voice_type=vt, surfaceable=False)
        gi["nodes"].append({"id": iid, "type": "Insight", "properties": props})
        gi["edges"] += [{"type": "SUPPORTED_BY", "from": iid, "to": q} for q in quotes]
    fx._w(p["gi"], gi)


def _props(p, iid: str) -> dict:
    out: dict = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == iid)["properties"]
    return out


def test_a_wrong_person_verdict_takes_the_name_off_every_surface(
    tmp_path: Path, monkeypatch
) -> None:
    """The review read the stored name as someone else's: m0025's unnaming path takes it off the
    voice (segments, transcript prefix, quotes, insights, diagnostics); its insights stay hidden."""
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and res.details["totals"]["unnamed_by_review"] == 1
    assert all("speaker_label" not in r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    text = p["txt"].read_text(encoding="utf-8")
    assert "Steve McConnell:" not in text and "\nSPEAKER_01: Thanks for having me." in text
    quote = _props(p, "quote:1")
    assert quote.get("speaker_id") is None and quote.get("speaker_voice_type") == "unknown"
    guest = _guest(p)
    assert "speaker" not in guest  # the name is gone from the insight too
    assert (guest["surfaceable"], guest["speaker_voice_type"]) == (False, "unknown")  # still hidden
    assert [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]] == ["Gergely Orosz"]
    fx._quotes_hold(p)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert ok and "unnamed_checked=1" in msg
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m31.undo(tmp_path)
    assert restored and not refused and "Steve McConnell:" in p["txt"].read_text(encoding="utf-8")


def test_a_wrong_person_verdict_on_a_voice_never_named_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    before = p["seg"].read_bytes()
    assert _apply(tmp_path).details["episodes"] == [] and p["seg"].read_bytes() == before


def test_an_entry_it_would_change_without_a_verdict_is_refused_as_unreviewed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="SKIP")
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    before = p["seg"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "unreviewed" in res.details["refused"][0]["why"]
    assert p["seg"].read_bytes() == before
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "refused" in msg


def test_an_entry_it_leaves_alone_needs_no_verdict(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="SKIP")
    _episode(tmp_path, "Steve McConnell", "guest", typed=False)
    assert _apply(tmp_path).details["episodes"] == []
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert ok and "unreviewed_unchanged=1" in msg


def test_a_misspelled_name_is_applied_as_listed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="CORRECT_MISSPELLED")
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _apply(tmp_path)
    assert _guest(p)["routing_tag"] == "surface" and "surfaceable" not in _guest(p)


def test_no_verdicts_file_refuses_the_plan_and_fails_verify(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    monkeypatch.setattr(m31.NamedVoicesShownMigration, "_verdicts_dir", tmp_path / "none")
    _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    with pytest.raises(rv.VerdictError, match="no verdicts file"):
        _mig().plan(_ctx(tmp_path))
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "verdicts file unusable" in msg


def test_verify_reads_the_outcome_and_can_fail(tmp_path: Path, monkeypatch) -> None:
    """Verify checks what is served, not only the labels: a shown voice's insight hidden again, or
    the name written back onto an unnamed voice, fails it; a commercial insight does not."""
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _add_insights(p, [("insight:ad", ["quote:1"], "commercial")])
    _apply(tmp_path)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert ok and "insights_checked=2" in msg
    gi = fx._r(p["gi"])
    for n in gi["nodes"]:
        if n["id"] == "insight:guest":
            n["properties"].update(surfaceable=False, speaker_voice_type="unknown")
    fx._w(p["gi"], gi)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "insight:guest still hidden" in msg


def test_verify_fails_when_an_unnamed_voice_carries_the_name_again(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _apply(tmp_path)
    assert _mig().verify(_ctx(tmp_path))[0]
    rows = fx._r(p["seg"])
    rows[1]["speaker_label"] = "Steve McConnell"
    fx._w(p["seg"], rows)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "still labels the voice" in msg


def test_only_the_first_quote_decides_and_only_unknown_or_unidentified_is_lifted(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards the reviewer's surviving mutations: the first SUPPORTED_BY quote decides (not the
    last, not any), an unidentified insight is lifted as an unknown one is, a commercial one never.
    """
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _add_insights(
        p,
        [
            ("insight:guest_first", ["quote:1", "quote:2"], "unknown"),  # guest first, host last
            ("insight:host_first", ["quote:0", "quote:1"], "unknown"),  # host first, guest second
            ("insight:unidentified", ["quote:1"], "unidentified"),
            ("insight:ad", ["quote:1"], "commercial"),
        ],
    )
    _apply(tmp_path)
    assert "surfaceable" not in _props(p, "insight:guest_first")
    assert _props(p, "insight:host_first")["surfaceable"] is False
    assert "surfaceable" not in _props(p, "insight:unidentified")
    assert _props(p, "insight:ad")["speaker_voice_type"] == "commercial"
    assert _props(p, "insight:ad")["surfaceable"] is False


def test_a_voice_typed_unidentified_under_its_name_is_shown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    rows = fx._r(p["seg"])
    for r in rows:
        if r["speaker"] == "SPEAKER_01":
            r["voice_type"] = "unidentified"
    fx._w(p["seg"], rows)
    res = _apply(tmp_path)
    assert res.details["totals"]["voices_shown"] == 1
    assert all("voice_type" not in r for r in fx._r(p["seg"]))


def test_the_right_name_in_the_wrong_role_is_not_done(tmp_path: Path, monkeypatch) -> None:
    """The done test compares the role: "Steve McConnell" stored as host is not the listed guest."""
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    _episode(tmp_path, "Steve McConnell", "host", typed=False)
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "the replay read" in res.details["refused"][0]["why"]


def test_a_newly_named_voice_takes_the_listed_role(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    _apply(tmp_path)
    assert {r["speaker_role"] for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01"} == {"guest"}


def test_the_receipt_and_the_result_name_the_verdicts_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    path = _review(monkeypatch, tmp_path)
    _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    res = _apply(tmp_path)
    import hashlib
    import json

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert res.details["verdicts_sha256"] == sha
    header = json.loads((tmp_path / m31.RECEIPTS_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert header["verdicts_sha256"] == sha


def test_a_verdict_belongs_to_its_episode_and_voice_not_to_the_name(
    tmp_path: Path, monkeypatch
) -> None:
    """One person's name on two entries (eight entries share "Justin Norman" on prod): the review
    of one episode must not decide the other."""
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {
            **ENTRY,
            ("aaa", "SPEAKER_01"): (None, "unknown", "Steve McConnell", "guest", "elsewhere"),
        },
    )
    path = _review(monkeypatch, tmp_path, SPEAKER_01="CORRECT")
    text = path.read_text(encoding="utf-8").replace(
        "ep\tfeeds/f/run_1/metadata/ep.metadata.json\tSPEAKER_01\tSteve McConnell\tCORRECT",
        "ep\tfeeds/f/run_1/metadata/ep.metadata.json\tSPEAKER_01\tSteve McConnell\tWRONG_PERSON",
    )
    path.write_text(text, encoding="utf-8")
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    res = _apply(tmp_path)
    assert res.details["totals"]["unnamed_by_review"] == 1
    assert all("speaker_label" not in r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")


def test_verify_fails_on_an_unidentified_insight_left_hidden(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _apply(tmp_path)
    gi = fx._r(p["gi"])
    for n in gi["nodes"]:
        if n["id"] == "insight:guest":
            n["properties"].update(surfaceable=False, speaker_voice_type="unidentified")
    fx._w(p["gi"], gi)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "still hidden (unidentified)" in msg


def test_verify_that_checked_nothing_does_not_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=False)
    gi = fx._r(p["gi"])
    gi["nodes"] = [n for n in gi["nodes"] if n["type"] != "Insight"]
    gi["edges"] = [e for e in gi["edges"] if e.get("type") != "SUPPORTED_BY"]
    fx._w(p["gi"], gi)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "verify checked nothing" in msg


def _with_admap_and_counts(p) -> str:
    """An ad map cutting the host's last line, and stale metadata counts; returns the cut text."""
    text = p["txt"].read_text(encoding="utf-8")
    cut = "Tell me about the book."
    a = text.index(cut)
    admap = {
        "source_length": len(text),
        "excised_ranges": [[a, a + len(cut)]],
        "excised_texts": [cut],
        "chars_removed": len(cut),
    }
    base = str(p["txt"])[: -len(".txt")]
    Path(base + ".adfree.admap.json").write_text(json.dumps(admap), encoding="utf-8")
    meta = fx._r(p["meta"])
    meta["grounded_insights"] = {"insight_count": 99}
    meta["knowledge_graph"] = {"node_count": 99, "edge_count": 99}
    fx._w(p["meta"], meta)
    return cut


def test_the_ad_map_moves_with_the_renamed_text(tmp_path: Path, monkeypatch) -> None:
    """A name changes the speaker prefixes, so every ad-map offset after it moves; the cut text
    must still be exactly what the range points at (the player cuts by these offsets)."""
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    cut = _with_admap_and_counts(p)
    res = _apply(tmp_path)
    assert res.details["totals"]["admaps_carried"] == 1
    text = p["txt"].read_text(encoding="utf-8")
    admap = json.loads(Path(str(p["txt"])[: -len(".txt")] + ".adfree.admap.json").read_text())
    [[a, b]] = admap["excised_ranges"]
    assert admap["source_length"] == len(text)
    assert text[a:b] == cut == admap["excised_texts"][0]
    assert admap["chars_removed"] == len(cut)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert ok and "admaps_checked=1" in msg


def test_the_metadata_counts_follow_the_rewritten_gi_and_kg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    _with_admap_and_counts(p)
    res = _apply(tmp_path)
    assert res.details["totals"]["metadata_counts_synced"] == 1
    meta, gi, kg = fx._r(p["meta"]), fx._r(p["gi"]), fx._r(p["kg"])
    assert meta["grounded_insights"]["insight_count"] == sum(
        1 for n in gi["nodes"] if n["type"] == "Insight"
    )
    assert meta["knowledge_graph"]["node_count"] == len(kg["nodes"])
    assert meta["knowledge_graph"]["edge_count"] == len(kg["edges"])


@pytest.mark.parametrize("damage", ["range_off_by_one", "old_length", "edited_text"])
def test_verify_fails_on_an_ad_map_that_does_not_match(tmp_path: Path, monkeypatch, damage) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    _with_admap_and_counts(p)
    _apply(tmp_path)
    path = Path(str(p["txt"])[: -len(".txt")] + ".adfree.admap.json")
    admap = json.loads(path.read_text())
    if damage == "range_off_by_one":
        admap["excised_ranges"][0][0] += 1
    elif damage == "old_length":
        admap["source_length"] -= 5
    else:
        admap["excised_texts"][0] = "something else"
    path.write_text(json.dumps(admap))
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "ad map does not match" in msg


def _diag_role(p, voice: str):
    return next(v for v in fx._r(p["diag"])["voices"] if v["voice"] == voice).get("role")


def test_a_wrong_person_verdict_also_restores_the_role_the_voice_had(
    tmp_path: Path, monkeypatch
) -> None:
    """The wrong name came with a role ("host" on a guest): unnaming puts back the listed old role,
    on the segments and in the diagnostics, so the voice is not shown as "Host"."""
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): (None, "guest", "Steve McConnell", "host", "wrong")},
    )
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    p = _episode(tmp_path, "Steve McConnell", "host", typed=True)
    _apply(tmp_path)
    assert {r.get("speaker_role") for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01"} == {
        "guest"
    }
    assert _diag_role(p, "SPEAKER_01") == "guest"
    assert _mig().verify(_ctx(tmp_path))[0]


def test_a_wrong_person_with_no_old_role_leaves_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): (None, None, "Steve McConnell", "host", "wrong")},
    )
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    p = _episode(tmp_path, "Steve McConnell", "host", typed=True)
    _apply(tmp_path)
    assert all("speaker_role" not in r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    assert _diag_role(p, "SPEAKER_01") == "unknown"
    gi = fx._r(p["gi"])
    assert not any(
        n["type"] == "Person" and n["id"] == "person:steve-mcconnell" for n in gi["nodes"]
    )
    rows = fx._r(p["seg"])
    rows[1]["speaker_role"] = "host"
    fx._w(p["seg"], rows)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "gives the voice a role other than None" in msg
    rows[1].pop("speaker_role")
    fx._w(p["seg"], rows)
    diag = fx._r(p["diag"])
    for v in diag["voices"]:
        if v["voice"] == "SPEAKER_01":
            v["role"] = "host"
    fx._w(p["diag"], diag)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "diagnostics give the voice role 'host'" in msg


def test_a_wrong_person_verdict_on_a_voice_now_named_otherwise_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    _episode(tmp_path, "Somebody Else", "guest", typed=True)
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "the review judged" in res.details["refused"][0]["why"]


def test_verify_fails_when_a_written_episode_has_an_entry_without_a_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    path = _review(monkeypatch, tmp_path)
    _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _apply(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # the entry's row removed
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "written by this migration, and has no verdict" in msg


def test_verify_judges_an_insight_by_its_first_quote(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _add_insights(p, [("insight:guest_first", ["quote:1", "quote:2"], "unknown")])
    _apply(tmp_path)
    gi = fx._r(p["gi"])
    for n in gi["nodes"]:
        if n["id"] == "insight:guest_first":
            n["properties"].update(surfaceable=False, speaker_voice_type="unknown")
    fx._w(p["gi"], gi)
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "insight:guest_first still hidden" in msg


def test_apply_undo_apply_undo_restores_every_file_once_and_refuses_none(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    before = {k: p[k].read_bytes() for k in ("seg", "gi", "meta", "txt")}
    _apply(tmp_path)
    first, refused = m31.undo(tmp_path)
    assert first and not refused
    _apply(tmp_path)
    again, refused = m31.undo(tmp_path)
    assert (again, refused) == (first, [])
    assert {k: p[k].read_bytes() for k in before} == before


def test_the_moved_functions_stay_identical_to_0032s_copy() -> None:
    """0032 keeps its own copy until its record opens it; until then the two must not drift."""
    import ast

    root = Path(__file__).resolve().parents[4] / "src/podcast_scraper/upgrade/migrations"

    def funcs(name: str) -> dict:
        """The FIRST definition of each name (0032 also has a migration-level ``plan``)."""
        out: dict = {}
        for n in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):
            if isinstance(n, ast.FunctionDef):
                out.setdefault(n.name, ast.dump(n, include_attributes=False))
        return out

    a, b = funcs("m0032_naming_voices_rederived.py"), funcs("voice_rewrite_derived.py")
    for name in ("plan", "_recount", "_carry_admap", "_move_by_diff", "_move_offset"):
        assert a[name] == b[name], name


def test_an_offset_moves_through_a_character_diff() -> None:
    import difflib

    from podcast_scraper.upgrade.migrations.voice_rewrite_derived import _move_by_diff

    old, new = "A: hello\nB: world", "Anna Lee: hello\nB: world"
    ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    assert _move_by_diff(old.index("world"), ops, len(old), len(new)) == new.index("world")
    assert _move_by_diff(len(old), ops, len(old), len(new)) == len(new)


def test_an_offset_in_a_label_gap_keeps_only_its_first_two_characters() -> None:
    from podcast_scraper.upgrade.migrations.voice_rewrite_derived import _move_offset

    old = [(0, 5), (12, 20)]  # text, then "\nX: " style gap of 7, then text
    new = [(0, 5), (17, 25)]  # the label grew by 5
    assert _move_offset(3, old, new, 20, 25) == 3  # inside the first text
    assert _move_offset(14, old, new, 20, 25) == 19  # inside the second text, shifted
    assert _move_offset(5, old, new, 20, 25) == 5  # the gap's first character keeps its place
    assert _move_offset(6, old, new, 20, 25) == 6  # and its second
    assert _move_offset(7, old, new, 20, 25) is None  # inside the rewritten label
    assert _move_offset(12, old, new, 20, 25) == 17  # the gap's end is the next text's start


@pytest.mark.parametrize("old_role", ["guest", None])
def test_a_wrong_person_verdict_on_a_renamed_voice_restores_the_old_name_and_role(
    tmp_path: Path, monkeypatch, old_role
) -> None:
    """0029 RENAMED the voice (old name set): the review's WRONG_PERSON puts back the listed old
    name and role, verify passes on that state, and a second plan writes and refuses nothing."""
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): ("Old Person", old_role, "Steve McConnell", "host", "renamed")},
    )
    _review(monkeypatch, tmp_path, SPEAKER_01="WRONG_PERSON")
    p = _episode(tmp_path, "Steve McConnell", "host", typed=True)
    _apply(tmp_path)
    rows = [r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01"]
    assert {r.get("speaker_label") for r in rows} == {"Old Person"}
    assert {r.get("speaker_role") for r in rows} == {old_role}
    assert _diag_role(p, "SPEAKER_01") == (old_role or "unknown")
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert ok, msg
    again = _apply(tmp_path)
    assert again.details["episodes"] == [] and again.details["refused"] == []


def test_the_plan_row_hashes_exactly_the_files_the_decision_opens(
    tmp_path: Path, monkeypatch
) -> None:
    """The outcome hook diffs plans by these rows: an input missing from them lets a changed file
    pass unseen. An independent spy (io.open) over the same decision gives the same set."""
    import io

    from podcast_scraper.upgrade.read_recorder import ABSENT

    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    [row] = list(m31.dry_run_report(tmp_path))
    assert row["counts"]["voices_shown"] == 1 and row["refused"] is None
    opened = set()
    real = io.open

    def spy(file, mode="r", *a, **k):
        if "r" in mode and "+" not in mode:
            opened.add(Path(file).resolve().relative_to(tmp_path.resolve()).as_posix())
        return real(file, mode, *a, **k)

    verdicts = m31.NamedVoicesShownMigration()._verdicts()  # a plan-level input, read before
    ep = m31._Episode(p["meta"], tmp_path, verdicts=verdicts)
    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr("builtins.open", spy)
    ep.plan()
    monkeypatch.undo()
    present = {k for k, v in row["inputs"].items() if v != ABSENT}
    assert opened <= set(row["inputs"])  # nothing opened that the row does not list
    assert present <= opened  # nothing hashed as read that the decision did not open
    assert all(not (tmp_path / k).exists() for k, v in row["inputs"].items() if v == ABSENT)


def test_an_edited_input_changes_its_plan_row(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    [before] = list(m31.dry_run_report(tmp_path))
    p["seg"].write_text(p["seg"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    [after] = list(m31.dry_run_report(tmp_path))
    rel = p["seg"].relative_to(tmp_path).as_posix()
    assert before["inputs"][rel] != after["inputs"][rel]


def test_verify_passes_where_no_listed_episode_is_served(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m29, "NAMING_ARC_VOICES", {("elsewhere", "SPEAKER_01"): ENTRY[("ep", "SPEAKER_01")]}
    )
    _review(monkeypatch, tmp_path)
    _episode(tmp_path, "Steve McConnell", "guest", typed=True)  # episode "ep", not listed
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert ok and "no listed episode is served here" in msg and "served_listed=0" in msg


def test_verify_fails_on_unreadable_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    _apply(tmp_path)
    p["meta"].write_text("[]", encoding="utf-8")  # valid JSON, not a record
    ok, msg = _mig().verify(_ctx(tmp_path))
    assert not ok and "metadata unreadable" in msg
