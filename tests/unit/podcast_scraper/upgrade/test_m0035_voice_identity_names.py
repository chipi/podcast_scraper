"""m0035: the voice identity rung on served episodes with fingerprints, reviewed, rewritten.

Synthetic feed: two sibling episodes whose host voice is published as the host, and the episode
under test, where the host's voice was left unnamed. Built on m0025's corpus fixture.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.ml.diarization import voice_identity as vi
from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import (
    m0025_one_person_one_entry as m25,
    m0034_talk_floor_names_removed as m34,
    m0035_voice_identity_names as m35,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)
_spec34 = importlib.util.spec_from_file_location(
    "m0034_tests", Path(__file__).with_name("test_m0034_talk_floor_names_removed.py")
)
assert _spec34 and _spec34.loader
t34 = importlib.util.module_from_spec(_spec34)
_spec34.loader.exec_module(t34)

MODEL = "pyannote/speaker-diarization-community-1"
HOST = [1.0, 0.0, 0.0]
GUEST = [0.0, 1.0, 0.0]


def _sibling(
    root: Path,
    run: str,
    eid: str,
    absolute: bool = False,
    vec: List[float] = HOST,
    talk: float = 900,
    model: str = MODEL,
) -> None:
    base = root / "feeds" / "f" / run
    (base / "metadata").mkdir(parents=True)
    (base / "transcripts").mkdir()
    stored = (
        f"/app/output/feeds/f/{run}/transcripts/{eid}.txt" if absolute else f"transcripts/{eid}.txt"
    )
    (base / "metadata" / f"{eid}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": eid}, "content": {"transcript_file_path": stored}})
    )
    (base / "transcripts" / f"{eid}.speakers.diagnostics.json").write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice": "SPEAKER_00",
                        "resolved_name": "Gerald Orban",
                        "named": True,
                        "role": "host",
                        "source": "self_intro",
                        "talk_time_s": talk,
                    },
                ],
                "voice_embeddings": {
                    "model_name": model,
                    "dimension": 3,
                    "by_voice": {"SPEAKER_00": vec},
                },
            }
        )
    )


def _feed(
    tmp_path: Path,
    ad_read: bool = False,
    absolute: bool = False,
    guest_name: str = "Stan McCormack",
    sibling_model: str = MODEL,
) -> Dict[str, Any]:
    turns: List[Tuple[str, str, Optional[str], str]] = []
    for i in range(12):
        turns.append(("SPEAKER_00", "SPEAKER_00", None, f"Question {i} about the book."))
        turns.append(("SPEAKER_01", guest_name, "guest", f"Answer {i} about the book."))
    if ad_read:
        # The host reading an ad: the host's fingerprint, 1 of 25 turns (4%, under the floor).
        turns.append(("SPEAKER_02", "SPEAKER_02", None, "This episode is brought to you by."))
    p: Dict[str, Any] = fx._corpus(
        tmp_path,
        speakers=[fx._speaker(guest_name, "guest", ["SPEAKER_01"], "roster", "guest")],
        turns=turns,
        known_hosts=["Gerald Orban"],
    )
    diag = fx._r(p["diag"])
    for v in diag["voices"]:
        v["talk_time_s"] = 300
        if v["voice"] == "SPEAKER_00":
            v["named"] = False
    diag["voice_embeddings"] = {
        "model_name": MODEL,
        "dimension": 3,
        "by_voice": {"SPEAKER_00": HOST, "SPEAKER_01": GUEST, "SPEAKER_02": HOST},
    }
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    _sibling(tmp_path, "run_0", "e1", absolute, model=sibling_model)
    _sibling(tmp_path, "run_2", "e2", absolute, model=sibling_model)
    return p


def _basis_cols(entry: Tuple, verdict: str) -> Tuple[str, str, str]:
    """``(basis, segment_idx, line_sha256)`` for a fixture verdict: 0035 requires the columns."""
    if verdict == "WRONG_PERSON" and entry[2]:
        return ("undecidable", "", "")  # a rename not added for lack of evidence
    basis = "another_person" if verdict == "WRONG_PERSON" else "addressed"
    return (basis, "0", rv.line_sha256("read"))


def _review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entries: Dict,
    verdict: str,
    with_rows: bool = True,
) -> None:
    d = tmp_path / "_verdicts35"
    d.mkdir(exist_ok=True)
    names = {k: str(e[2] or e[0]) for k, e in entries.items()}
    rows = [
        "\t".join(
            (eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read")
            + _basis_cols(entries[(eid, voice)], verdict)
        )
        for (eid, voice), name in sorted(names.items())
        if with_rows
    ]
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-10-23\n"
    )
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def test_the_detector_names_the_hosts_unnamed_voice_from_the_feeds_fingerprints(
    tmp_path: Path,
) -> None:
    _feed(tmp_path)
    changes = m35.find_changes(tmp_path)
    assert list(changes) == [("ep", "SPEAKER_00")]
    old_name, old_role, new_name, new_role, why = changes[("ep", "SPEAKER_00")]
    assert (old_name, new_name, new_role) == (None, "Gerald Orban", "host")
    assert "fingerprint group at 1.00" in why


def test_an_episode_without_fingerprints_is_not_touched(tmp_path: Path) -> None:
    p = _feed(tmp_path)
    diag = fx._r(p["diag"])
    del diag["voice_embeddings"]
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    assert m35.find_changes(tmp_path) == {}


def test_a_reviewed_rename_is_written_on_every_surface_and_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    mig = m35.VoiceIdentityNamesMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert "\nGerald Orban: " in "\n" + p["txt"].read_text(encoding="utf-8")
    for surface, names in t34.fx_surfaces(p).items():
        assert "Gerald Orban" in names, surface
    host = {v["voice"]: v for v in fx._r(p["diag"])["voices"]}["SPEAKER_00"]
    assert host["named"] and host["resolved_name"] == "Gerald Orban"
    # Review M4: the rung's own names never vote, and this is the rung's name.
    assert host["source"] == vi.SOURCE
    assert mig.verify(_ctx(tmp_path))[0]
    assert (tmp_path / m35.RECEIPTS_FILE).is_file()
    restored, refused = m35.undo(tmp_path)
    assert restored and not refused
    assert "Gerald Orban:" not in p["txt"].read_text(encoding="utf-8")


def test_a_rename_the_review_reads_as_wrong_is_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "WRONG_PERSON")
    before = p["txt"].read_text(encoding="utf-8")
    assert m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert p["txt"].read_text(encoding="utf-8") == before


def test_review_names_a_rename_by_its_new_name_and_an_unnaming_by_its_old_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed = {
        ("e", "SPEAKER_00"): (None, None, "Ann Lee", "host", "match"),
        ("e", "SPEAKER_01"): ("Bo Diaz", "guest", None, None, "clearly not"),
    }
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    assert m35.VoiceIdentityNamesMigration().review_names() == {
        ("e", "SPEAKER_00"): "Ann Lee",
        ("e", "SPEAKER_01"): "Bo Diaz",
    }


def test_registered_with_its_list_and_verdicts_shipped() -> None:
    # Registered 2026-10-26 with 0034, once the list (snapshot-prod-20261025) and its reviewed
    # verdicts existed. A rename is reviewed by its new name, an unnaming by its old one.
    ids = [m.id for m in get_migrations()]
    assert m35.MIGRATION_ID in ids
    assert ids.index(m35.MIGRATION_ID) == ids.index("0034_talk_floor_names_removed") + 1
    entries = m35.load_list(m35.LIST_FILE)
    assert entries
    names = {k: str(e[2] or e[0]) for k, e in entries.items()}
    rv.load(m35.MIGRATION_ID, names, listed=entries, require_basis=True, bases=m35.BASES)
    assert ("Buzzsprout-20005821", "SPEAKER_00") not in entries  # the operator's exclusion


def test_a_listed_voice_without_a_verdict_fails_verify_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT", with_rows=False)
    before = p["txt"].read_text(encoding="utf-8")
    ok, message = m35.VoiceIdentityNamesMigration().verify(_ctx(tmp_path))
    assert not ok and "without a verdict" in message
    assert m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert p["txt"].read_text(encoding="utf-8") == before


def test_a_voice_under_the_talk_floor_is_not_listed_even_when_it_is_the_hosts_voice(
    tmp_path: Path,
) -> None:
    # Review M2: the pipeline floors after the rung; a host-read ad must not get the name back.
    _feed(tmp_path, ad_read=True)
    assert list(m35.find_changes(tmp_path)) == [("ep", "SPEAKER_00")]


@pytest.mark.parametrize("host_s, listed", [(5.0, True), (4.99, False)])
def test_the_talk_floor_boundary_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_s: float, listed: bool
) -> None:
    # Step-8 round 4: a voice at exactly the floor (5.0%) is not under it; `<=` survived every test.
    assert m34.FLOOR == 0.05
    _feed(tmp_path)
    monkeypatch.setattr(
        m34, "talk_by_voice", lambda rows: {"SPEAKER_00": host_s, "SPEAKER_01": 100.0 - host_s}
    )
    assert (("ep", "SPEAKER_00") in m35.find_changes(tmp_path)) is listed


def test_siblings_whose_metadata_stores_an_absolute_path_still_vote(tmp_path: Path) -> None:
    # Review M1: a snapshot copy is not mounted at /app/output.
    _feed(tmp_path, absolute=True)
    assert list(m35.find_changes(tmp_path)) == [("ep", "SPEAKER_00")]


def test_a_name_this_migration_applied_does_not_vote_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path))
    history = vi.feed_history(tmp_path / "feeds" / "f", "someone-else", MODEL)
    mine = [h for h in history if h.episode_id == "ep" and h.vec == vi._unit(HOST)]
    assert mine and all(h.name is None for h in mine)


def test_a_respelling_rename_takes_the_rungs_source_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Second review m2: "Gerald Orben" -> the group's "Gerald Orban" is the same person by
    # same_person, and kept its old source, so it voted. Every 0035 name is the rung's. (Step-8
    # review: the pair here was "Gerald Orben", which same_person rejects, so the test never
    # reached the respelling branch.)
    assert m25.same_person("Gerald Orben", "Gerald Orban")
    p = _feed(tmp_path)
    diag = fx._r(p["diag"])
    seg = fx._r(p["seg"])
    rows = seg if isinstance(seg, list) else seg["segments"]
    for r in rows:
        if r["speaker"] == "SPEAKER_00":
            r["speaker_label"], r["speaker_role"] = "Gerald Orben", "host"
    p["seg"].write_text(json.dumps(seg), encoding="utf-8")
    for v in diag["voices"]:
        if v["voice"] == "SPEAKER_00":
            v.update(named=True, resolved_name="Gerald Orben", source="llm_resolution")
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    entries = m35.find_changes(tmp_path)
    assert entries[("ep", "SPEAKER_00")][:3] == ("Gerald Orben", "host", "Gerald Orban")
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path))
    host = {v["voice"]: v for v in fx._r(p["diag"])["voices"]}["SPEAKER_00"]
    assert host["resolved_name"] == "Gerald Orban" and host["source"] == vi.SOURCE


def test_the_dry_run_rows_carry_counts_and_input_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As 0034's: the outcome hook's plan diff needs each episode's counters and input sha256s.
    _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    rows = list(m35.dry_run_report(tmp_path))
    assert rows and all(r["counts"] and r["inputs"] for r in rows)
    # a file the plan looked for and did not find is recorded "absent", which the diff also compares
    assert all(len(h) == 64 or h == "absent" for r in rows for h in r["inputs"].values())
    assert any(len(h) == 64 for r in rows for h in r["inputs"].values())


def _review_each(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: Dict, verdicts: Dict
) -> None:
    """A verdicts file with one verdict per listed voice (``verdicts[key]``)."""
    d = tmp_path / "_verdicts35"
    d.mkdir(exist_ok=True)
    rows = [
        "\t".join(
            (eid, "m.json", voice, str(e[2] or e[0]), verdicts[(eid, voice)], "read")
            + _basis_cols(e, verdicts[(eid, voice)])
        )
        for (eid, voice), e in sorted(entries.items())
    ]
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-10-26\n"
    )
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)


def test_the_detector_takes_a_name_off_a_voice_clearly_not_that_person(tmp_path: Path) -> None:
    # Step-8 review K7: the unnaming branch had no test. The guest's voice is served under the
    # host's name; its fingerprint is nothing like the host's group, so the name comes off.
    _feed(tmp_path, guest_name="Gerald Orban")
    changes = m35.find_changes(tmp_path)
    assert sorted(changes) == [("ep", "SPEAKER_00"), ("ep", "SPEAKER_01")]
    old_name, old_role, new_name, new_role, why = changes[("ep", "SPEAKER_01")]
    assert (old_name, old_role, new_name, new_role) == ("Gerald Orban", "guest", None, None)
    assert why.startswith("clearly not 'Gerald Orban': similarity 0.00")


def test_an_unnaming_is_measured_against_the_last_group_of_that_name(tmp_path: Path) -> None:
    # decide compares the LAST group a name won; the evidence text must report that one.
    _feed(tmp_path, guest_name="Gerald Orban")
    other = [0.0, 0.0, 1.0]
    _sibling(tmp_path, "run_3", "e3", vec=other, talk=800)
    _sibling(tmp_path, "run_4", "e4", vec=other, talk=800)
    p = next((tmp_path / "feeds" / "f").glob("run_1*/transcripts/*.speakers.diagnostics.json"))
    diag = json.loads(p.read_text(encoding="utf-8"))
    diag["voice_embeddings"]["by_voice"]["SPEAKER_01"] = [0.3, 0.954, 0.0]
    p.write_text(json.dumps(diag), encoding="utf-8")
    changes = m35.find_changes(tmp_path)
    why = changes[("ep", "SPEAKER_01")][4]
    assert "similarity 0.00" in why, why  # 0.30 to the first group, 0.00 to the last
    # A rename's evidence is the group decide matched: the most similar one (1.00, not 0.00).
    assert "fingerprint group at 1.00" in changes[("ep", "SPEAKER_00")][4]


def test_an_episode_without_an_id_is_not_listed(tmp_path: Path) -> None:
    p = _feed(tmp_path)
    meta = next(tmp_path.glob("feeds/f/run_1*/metadata/*.metadata.json"))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["episode"]["episode_id"] = ""
    meta.write_text(json.dumps(doc), encoding="utf-8")
    assert p and m35.find_changes(tmp_path) == {}


def test_an_unnaming_applies_on_wrong_person_and_not_on_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = {
        ("e", "SPEAKER_00"): (None, None, "Ann Lee", "host", "match"),
        ("e", "SPEAKER_01"): ("Bo Diaz", "guest", None, None, "clearly not"),
        ("e", "SPEAKER_02"): ("Cy Ray", "guest", None, None, "clearly not"),
        ("e", "SPEAKER_03"): (None, None, "Di Fox", "guest", "match"),
    }
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    _review_each(
        monkeypatch,
        tmp_path,
        listed,
        {
            ("e", "SPEAKER_00"): "WRONG_PERSON",  # a rename read as wrong: left
            ("e", "SPEAKER_01"): "WRONG_PERSON",  # an unnaming whose name is wrong: applied
            ("e", "SPEAKER_02"): "CORRECT",  # an unnaming whose name is right: kept
            ("e", "SPEAKER_03"): "CORRECT_MISSPELLED",  # a misspelled right name: applied
        },
    )
    apply, missing = m35.VoiceIdentityNamesMigration()._reviewed()
    assert missing == []
    assert sorted(apply) == [("e", "SPEAKER_01"), ("e", "SPEAKER_03")]


def test_a_reviewed_unnaming_is_written_and_the_rename_beside_it_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path, guest_name="Gerald Orban")
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review_each(
        monkeypatch,
        tmp_path,
        entries,
        {("ep", "SPEAKER_00"): "CORRECT", ("ep", "SPEAKER_01"): "WRONG_PERSON"},
    )
    mig = m35.VoiceIdentityNamesMigration()
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    voices = {v["voice"]: v for v in fx._r(p["diag"])["voices"]}
    assert voices["SPEAKER_00"]["resolved_name"] == "Gerald Orban"
    assert not voices["SPEAKER_01"]["named"]
    answers = [ln for ln in p["txt"].read_text(encoding="utf-8").splitlines() if "Answer" in ln]
    assert answers and not any(ln.startswith("Gerald Orban") for ln in answers)
    assert mig.verify(_ctx(tmp_path))[0]


def test_the_backups_and_receipts_are_not_0034s() -> None:
    # Undo restores from the backup tag and reads the receipts file: shared with 0034, undoing
    # one migration would restore the other's files.
    assert m35.BACKUP_TAG != m34.BACKUP_TAG
    assert m35.RECEIPTS_FILE != m34.RECEIPTS_FILE
    mig = m35.VoiceIdentityNamesMigration()
    assert (mig._backup_tag, mig._receipts_file) == (m35.BACKUP_TAG, m35.RECEIPTS_FILE)


def test_the_rungs_constants_are_the_pipelines_defaults() -> None:
    assert m35.THRESHOLD == Config.model_fields["voice_identity_threshold"].default
    assert m35.MIN_TALK_S == inspect.signature(vi.apply_to_roster).parameters["min_talk"].default


def test_siblings_fingerprinted_by_another_model_do_not_vote(tmp_path: Path) -> None:
    _feed(tmp_path, sibling_model="another/model")
    assert m35.find_changes(tmp_path) == {}


def test_a_rename_without_a_role_keeps_the_voices_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    seg = fx._r(p["seg"])
    rows = seg if isinstance(seg, list) else seg["segments"]
    for r in rows:
        if r["speaker"] == "SPEAKER_00":
            r["speaker_role"] = "host"
    p["seg"].write_text(json.dumps(seg), encoding="utf-8")
    monkeypatch.setattr(vi, "decide", lambda *a, **k: {"SPEAKER_00": ("Gerald Orban", None)})
    meta = next(tmp_path.glob("feeds/f/run_1*/metadata/*.metadata.json"))
    _, changes = m35.episode_changes(meta)
    assert changes["SPEAKER_00"][3] == "host"


def test_an_excluded_voice_is_never_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _feed(tmp_path)
    assert list(m35.find_changes(tmp_path)) == [("ep", "SPEAKER_00")]
    monkeypatch.setitem(m35.EXCLUDED, ("ep", "SPEAKER_00"), "test")
    assert m35.find_changes(tmp_path) == {}


def test_an_unnaming_on_undecidable_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unnaming applies only when the text shows someone else; the loader's 0035 basis set
    # allows undecidable on WRONG_PERSON for renames only.
    listed = {("e", "SPEAKER_01"): ("Bo Diaz", "guest", None, None, "clearly not")}
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    d = tmp_path / "_verdicts35"
    d.mkdir()
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(listed)}\n# reviewer: test\n# date: 2026-10-26\n"
    )
    row = "\t".join(
        ("e", "m.json", "SPEAKER_01", "Bo Diaz", "WRONG_PERSON", "x", "undecidable", "", "")
    )
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + row + "\n"
    )
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)
    with pytest.raises(rv.VerdictError, match="only when the text shows someone else"):
        m35.VoiceIdentityNamesMigration()._verdicts()


def test_a_verdicts_file_without_the_basis_columns_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = {("e", "SPEAKER_00"): (None, None, "Ann Lee", "host", "match")}
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    d = tmp_path / "_verdicts35"
    d.mkdir()
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(listed)}\n# reviewer: test\n# date: 2026-10-26\n"
    )
    row = "\t".join(("e", "m.json", "SPEAKER_00", "Ann Lee", "CORRECT", "read"))
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(head + "\t".join(rv.COLUMNS) + "\n" + row + "\n")
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)
    with pytest.raises(rv.VerdictError, match="columns are"):
        m35.VoiceIdentityNamesMigration()._verdicts()


def test_a_rename_on_undecidable_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Step-8 round 2: a rename CORRECT on 'undecidable' would be applied; a name is added only on
    # positive evidence in the text.
    listed = {("e", "SPEAKER_00"): (None, None, "Ann Lee", "host", "match")}
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    d = tmp_path / "_verdicts35"
    d.mkdir()
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(listed)}\n# reviewer: test\n# date: 2026-10-26\n"
    )
    row = "\t".join(("e", "m.json", "SPEAKER_00", "Ann Lee", "CORRECT", "x", "undecidable", "", ""))
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + row + "\n"
    )
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)
    with pytest.raises(rv.VerdictError, match="only on positive evidence"):
        m35.VoiceIdentityNamesMigration()._verdicts()


def test_the_version_and_the_verify_message_are_0035s_own() -> None:
    mig = m35.VoiceIdentityNamesMigration()
    assert mig.to_version == "2.7.29"  # after 0034's 2.7.28, before 0036's 2.7.30
    assert "rename" in mig._verified and "floor" not in mig._verified


def test_a_misspelling_rename_on_undecidable_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CORRECT_MISSPELLED is the other half of "a rename applies": undecidable is no evidence either.
    listed = {("e", "SPEAKER_00"): (None, None, "Ann Lee", "host", "match")}
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    d = tmp_path / "_verdicts35"
    d.mkdir()
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(listed)}\n# reviewer: test\n# date: 2026-10-26\n"
    )
    row = "\t".join(
        ("e", "m.json", "SPEAKER_00", "Ann Lee", "CORRECT_MISSPELLED", "x", "undecidable", "", "")
    )
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + row + "\n"
    )
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)
    with pytest.raises(rv.VerdictError, match="only on positive evidence"):
        m35.VoiceIdentityNamesMigration()._verdicts()


def test_verify_after_a_reviewed_apply_returns_0035s_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    mig = m35.VoiceIdentityNamesMigration()
    mig.apply(_ctx(tmp_path))
    ok, message = mig.verify(_ctx(tmp_path))
    assert ok
    assert "rename" in message and "floor" not in message
