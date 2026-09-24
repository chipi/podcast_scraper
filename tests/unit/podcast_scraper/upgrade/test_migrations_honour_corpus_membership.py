"""Migrations must touch only SERVED copies, and a mispaired episode must never demote silently.

Two defects found during the #2097 chain (2026-09-22), both filed and deferred at the time because
changing what three no-undo migrations select, days before running them irreversibly on prod, was
riskier than the defect. The migration has since run, so they are actionable.

DEFECT 1 — the migrations were the corpus's one consumer that ignored its own membership rule.
``search.corpus_scope.dedupe_metadata_paths_newest_run_per_episode`` is documented there as the
"Central corpus-membership rule"; the serving layer and the repair route both honour it; m0007 /
m0009 / m0010 globbed ``rglob`` and rewrote every run copy. Measured on prod: 2,312 globbed vs
2,002 served = 310 superseded copies that are never served, never searched, and cannot be repaired.

That is not merely wasted work. A superseded copy can hold a roster from before a repair, and on
one of them m0009 demoted Krishna Rao — a real guest — from guest to mentioned.

DEFECT 2 — the hand-read guard was blind to its own failure mode. m0009 flags a demotion for
hand-read when the node also carried a voice, but voices are read from the episode's GI/KG graph;
if the roster is wrong, that graph was built from ANOTHER episode's transcript, so the voices
belong to other people, the demoted person has none, and the guard says nothing. Krishna Rao
carried no ``[HAD A VOICE]`` marker and never reached the 59-entry hand-read list.
"""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.upgrade.corpus_selection import select_served_artifacts
from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import (
    transcript_pairing_is_vouched,
)

FEED_ID = "sha256:feed-aaa"
EP_ID = "ep-krishna"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _episode(corpus: Path, run: str, stem: str, *, transcript: str | None = None) -> Path:
    """Write a metadata + kg pair under ``feeds/f/<run>/`` and return the kg path."""
    base = corpus / "feeds" / "f" / run
    content = {"transcript_file_path": f"transcripts/{transcript}"} if transcript else {}
    _write(
        base / "metadata" / f"{stem}.metadata.json",
        {
            "episode": {"episode_id": EP_ID, "title": "Anthropic's CFO on Managing Compute"},
            "feed": {"feed_id": FEED_ID, "title": "Invest Like the Best"},
            "content": content,
        },
    )
    kg = base / "metadata" / f"{stem}.kg.json"
    _write(kg, {"nodes": []})
    return kg


# ---------------------------------------------------------------------------
# Defect 1
# ---------------------------------------------------------------------------


def test_only_the_newest_run_copy_of_an_episode_is_selected(tmp_path: Path):
    """The superseded copy — the one that carried the stale roster — is not selected at all."""
    corpus = tmp_path / "corpus"
    old = _episode(corpus, "run_20260818-000000", "0001 - Anthropic_20260818-000000")
    new = _episode(corpus, "run_20260826-000000", "0001 - Anthropic_20260826-000000")

    served, superseded = select_served_artifacts(corpus, ".kg.json")

    assert served == [new], f"served set is wrong: {served}"
    assert superseded == [old], f"superseded set is wrong: {superseded}"


def test_a_bare_rglob_would_have_returned_both(tmp_path: Path):
    """Pins the delta rather than trusting it: the old selection really did include the bad copy.

    Without this the test above could pass against a corpus where only one copy exists, proving
    nothing about the fix.
    """
    corpus = tmp_path / "corpus"
    _episode(corpus, "run_20260818-000000", "0001 - Anthropic_20260818-000000")
    _episode(corpus, "run_20260826-000000", "0001 - Anthropic_20260826-000000")

    assert len(sorted(corpus.rglob("*.kg.json"))) == 2
    assert len(select_served_artifacts(corpus, ".kg.json")[0]) == 1


def test_disjoint_episodes_across_runs_all_survive(tmp_path: Path):
    """The incremental-add case must not be mistaken for supersession.

    Dropping these would silently shrink what a migration covers — a worse failure than the one
    being fixed, and the reason this defers to the central rule instead of reimplementing it.
    """
    corpus = tmp_path / "corpus"
    a = _episode(corpus, "run_20260818-000000", "0001 - A_20260818-000000")
    base = corpus / "feeds" / "f" / "run_20260826-000000"
    _write(
        base / "metadata" / "0002 - B_20260826-000000.metadata.json",
        {
            "episode": {"episode_id": "ep-other", "title": "B"},
            "feed": {"feed_id": FEED_ID, "title": "Invest Like the Best"},
            "content": {},
        },
    )
    b = base / "metadata" / "0002 - B_20260826-000000.kg.json"
    _write(b, {"nodes": []})

    served, superseded = select_served_artifacts(corpus, ".kg.json")
    assert sorted(served) == sorted([a, b])
    assert superseded == []


def test_an_artifact_with_no_metadata_sibling_is_kept(tmp_path: Path):
    """Safe direction, stated explicitly: membership is unknowable, so do not silently drop it."""
    corpus = tmp_path / "corpus"
    orphan = corpus / "feeds" / "f" / "run_20260818-000000" / "metadata" / "0009 - Orphan.kg.json"
    _write(orphan, {"nodes": []})

    served, superseded = select_served_artifacts(corpus, ".kg.json")
    assert served == [orphan]
    assert superseded == []


def test_all_three_migrations_select_through_the_central_rule():
    """Structural: a future migration reverting to `rglob` reintroduces the whole class.

    Cheap, and it survives refactors of the helper's internals — which a behavioural test of one
    migration would not.
    """
    import inspect

    from podcast_scraper.upgrade import rewrite_bridges_m0007
    from podcast_scraper.upgrade.migrations import (
        m0009_backfill_speaker_roles,
        m0010_canonical_person_names,
    )

    for mod, fn in (
        (m0009_backfill_speaker_roles, "_iter_kg_files"),
        (m0010_canonical_person_names, "_iter_gi_files"),
        (rewrite_bridges_m0007, "_iter_bridges"),
    ):
        target = getattr(mod, fn)
        src = inspect.getsource(target)
        assert "select_served_artifacts" in src, f"{mod.__name__}.{fn} bypasses the central rule"
        # CODE ONLY, not the docstring. Each of those functions explains in prose that it is
        # deliberately "not a bare rglob", so scanning the whole source failed on its own
        # explanation. Checking the text a comment can trip is not checking the code.
        body = src.replace(target.__doc__ or "", "") if target.__doc__ else src
        assert "rglob" not in body, f"{mod.__name__}.{fn} still globs every run copy"


# ---------------------------------------------------------------------------
# Defect 2
# ---------------------------------------------------------------------------


def test_a_transcript_naming_another_episode_is_not_vouched(tmp_path: Path):
    """The corruption signature — and the one thing the poisoned graph cannot hide."""
    meta = tmp_path / "0001 - Anthropic_s CFO on Managing Co_20260818.metadata.json"
    content = {
        "transcript_file_path": "transcripts/0010 - Re-engineering the Semicondu_20260818.txt"
    }
    assert transcript_pairing_is_vouched(meta, content) is False


def test_a_truncated_title_still_counts_as_the_same_episode(tmp_path: Path):
    """Equality would be WRONG here and the number it produces is wrong.

    The metadata filename truncates the title and the transcript filename does not. A plain
    equality test called 128 correctly-paired episodes mispaired, which is how #2082's headline
    read 275 instead of 147. Flagging those as mispaired would bury the real ones.
    """
    meta = tmp_path / "0006 - This Funding Model is Helping Fi_guid123.metadata.json"
    long_name = "0006 - This Funding Model is Helping Fight Climate Change_guid123.txt"
    content = {"transcript_file_path": f"transcripts/{long_name}"}
    assert transcript_pairing_is_vouched(meta, content) is True


def test_no_recorded_transcript_is_vouched_not_suspect(tmp_path: Path):
    """Nothing to mispair. Treating absence as corruption would flood the hand-read list."""
    meta = tmp_path / "0001 - A_20260818.metadata.json"
    assert transcript_pairing_is_vouched(meta, {}) is True
    assert transcript_pairing_is_vouched(meta, {"transcript_file_path": ""}) is True


def test_the_vouch_check_never_raises(tmp_path: Path):
    """It gates a hand-read list, not correctness — it must not be able to fail a migration."""
    meta = tmp_path / "weird-name-with-no-suffix"
    assert transcript_pairing_is_vouched(meta, {"transcript_file_path": 12345}) in (True, False)
    assert transcript_pairing_is_vouched(meta, None) is True  # type: ignore[arg-type]
