"""An event sink may never write into a file that something else REWRITES.

``emit_event`` APPENDS a line. A mutable registry is read-all / serialise-all / rename. Point one
at the other and the append is destroyed by the next rewrite — and, worse, is first PARSED as a
record of whatever the registry holds.

The instance: ``_FILE_FOR["job"]`` mapped to ``.viewer/jobs.jsonl``, which
``server.pipeline_job_registry.write_jobs_atomic`` rewrites whole on every job status change. The
first ``emit_event("job", ...)`` would have appended a row the next registry write silently
deletes, while ``read_jobs`` parsed the event envelope as a job record — no ``job_id``, so it
lands in the ``no_id`` bucket of ``_dedupe_job_rows_by_id`` and is carried into the rewritten
registry. It never fired only because nothing ever called it; the mapping sat there looking like a
supported feature, which is why it was going to be found the hard way.

These tests pin the CLASS. A test that merely asserted ``"job" not in _FILE_FOR`` would pass again
the moment someone adds ``"job_status"`` pointing at the same file.

The same append-only-vs-mutable confusion is what broke ``podcast-ingest-stalled`` for its entire
life (12260e8d2 here, 9bc438f in homelab), so it is worth a guard rather than a comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.obs import events
from podcast_scraper.server import pipeline_job_registry as registry

# Files (corpus-relative) owned by a writer that REWRITES them. Anything appending here loses
# data and corrupts the owner's parse.
REWRITTEN_WHOLE = {
    f"{registry.VIEWER_DIR}/{registry.JOBS_FILE}",
    f"{registry.VIEWER_DIR}/{registry.LOCK_NAME}",
}


def _norm(rel: str) -> str:
    """Corpus-relative POSIX form, with a leading ``./`` removed — and NOTHING else.

    NOT ``lstrip("./")``: that strips a CHARACTER SET, so ``.viewer/jobs.jsonl`` became
    ``viewer/jobs.jsonl`` and every ``startswith(".viewer/")`` check below silently could not
    fire. Caught by mutation — adding ``.viewer/job_events.jsonl`` passed all seven tests. The
    exact-filename check still worked only because both sides were mangled identically, which is
    the kind of accident that makes a guard look functional.
    """
    p = Path(rel).as_posix()
    return p[2:] if p.startswith("./") else p


def test_no_event_type_maps_into_a_file_that_is_rewritten_whole():
    """The class guard. Fails for ANY new entry pointing at a rewritten file, not just "job"."""
    offenders = {
        event_type: rel
        for event_type, rel in events._FILE_FOR.items()
        if _norm(rel) in {_norm(r) for r in REWRITTEN_WHOLE}
    }
    assert not offenders, (
        f"event type(s) {offenders} map into a file that is rewritten whole. emit_event APPENDS; "
        "the next rewrite deletes the row AND parses it as a record of that file's own type. "
        "Point it at an append-only path (the events/<type>.jsonl fallback is one)."
    )


def test_no_event_type_writes_anywhere_under_the_viewer_registry_dir():
    """Broader than the exact filenames: the whole ``.viewer/`` dir is owned, mutable state.

    Catches a near-miss like ``.viewer/job_events.jsonl`` that the exact-filename check above
    would wave through, and which the registry's own directory handling could still disturb.
    """
    offenders = {
        event_type: rel
        for event_type, rel in events._FILE_FOR.items()
        if _norm(rel).startswith(f"{registry.VIEWER_DIR}/")
    }
    assert (
        not offenders
    ), f"event type(s) {offenders} write inside the mutable {registry.VIEWER_DIR}/ dir"


def test_an_unmapped_type_falls_back_to_an_append_only_events_file(tmp_path: Path):
    """Removing the bad entry is only correct because the fallback is already right.

    ``"job"`` now resolves through the fallback, so job events remain available — at an
    append-only path — without anyone re-adding a mapping.
    """
    out = events._corpus_path(tmp_path, "job")
    assert out == tmp_path / "events" / "job.jsonl"
    assert _norm(str(out.relative_to(tmp_path))) not in {_norm(r) for r in REWRITTEN_WHOLE}


@pytest.mark.parametrize("event_type", sorted(events._FILE_FOR))
def test_every_mapped_path_is_relative_and_stays_inside_the_corpus(event_type: str):
    """A mapping is joined onto the corpus root, so an absolute or climbing path escapes it."""
    rel = events._FILE_FOR[event_type]
    assert not Path(rel).is_absolute(), f"{event_type} maps to an absolute path"
    assert ".." not in Path(rel).parts, f"{event_type} climbs out of the corpus"


def test_the_registry_still_owns_the_path_this_guard_is_written_against():
    """If the registry moves its file, this guard must move with it rather than silently pass.

    The constants are imported from the registry for that reason; this asserts they still
    describe the location the docstring claims, so the guard cannot rot into a no-op.
    """
    assert registry.VIEWER_DIR == ".viewer"
    assert registry.JOBS_FILE == "jobs.jsonl"
