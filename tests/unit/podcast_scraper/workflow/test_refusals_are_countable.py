"""Every silent refusal in the reprocess paths must write an episode-ledger row.

The 2026-09-23 batch reported ``episodes=N ok=N failed=0`` per feed while silently doing less
than it was asked. A reprocess helper that cannot do its job returns ``(False, None, 0)``; the
caller's ``if success:`` guard then skips BOTH the counter and ``update_episode_status``, so the
episode lands in neither ``ok`` nor ``failed`` and vanishes between the selection log line and the
run summary. The run index cannot recover it either, because a reprocess writes metadata back into
the OLD run dir and leaves the new one empty for every episode, successful or not.

Eleven ``_record_unresolved_transcript`` call sites fix that. Six of them — ``NoSpeakerIdentity``,
``NoAudio``, ``NoSegments``, ``NoTranscriptUrl``, ``NoCandidateParsed``, ``TooFewVoices`` — had NO
test at all (review finding M4, confirmed: ``grep -rl`` over ``tests/`` returned zero files for
each), so deleting any of them passed the whole suite. The commit that added them was titled "six
more refusals vanished from the ledger".

WHY STRUCTURAL RATHER THAN SIX FIXTURES. Each of those returns sits deep inside a helper needing
substantial setup, and six hand-built fixtures would pin six specific paths while leaving the NEXT
refusal someone adds just as invisible as these were. The invariant is what matters and it is
mechanically checkable: a refusal return must be accompanied by a ledger write in the same block.
This also fails for a refusal added tomorrow, which is the property that actually protects the
ledger.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "podcast_scraper"
    / "workflow"
    / "episode_processor.py"
)

RECORDER = "_record_unresolved_transcript"

# The refusal signature: these helpers report "I could not do this" as a 3-tuple.
REFUSAL_RETURNS = {"(False, None, 0)", "False, None, 0"}

EXPECTED_ERROR_TYPES = {
    "NoSpeakerIdentity",
    "NoAudio",
    "NoSegments",
    "NoTranscriptUrl",
    "NoCandidateParsed",
    "TooFewVoices",
    "TranscriptUnresolved",
    # Found by the structural guard below, not by review: an unaccounted refusal on the
    # ORDINARY transcribe path, where a misconfigured provider under-delivers on every episode.
    "NoTranscriptionProvider",
}


def _tree():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def _is_refusal_return(node: ast.AST) -> bool:
    if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
        return False
    elts = node.value.elts
    if len(elts) != 3:
        return False
    first, second, third = elts
    return (
        isinstance(first, ast.Constant)
        and first.value is False
        and isinstance(second, ast.Constant)
        and second.value is None
        and isinstance(third, ast.Constant)
        and third.value == 0
    )


def _calls_recorder(stmts) -> bool:
    """True only for an UNCONDITIONAL, top-level recorder call among *stmts*.

    Deliberately not ``ast.walk``. Walking would accept a recorder call nested under an inner
    ``if`` earlier in the same block, which satisfies the guard while the runtime write stays
    conditional — a false pass. Review confirmed no such shape exists in the file today (all 11
    sites are direct ``ast.Expr`` siblings), so this costs nothing now and closes the hole for
    future edits, which is the whole point of a structural guard.
    """
    for stmt in stmts:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        fn = stmt.value.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name == RECORDER:
            return True
    return False


def _blocks(tree):
    """Yield every statement list that can hold a refusal return."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield node, stmts


def test_every_refusal_return_is_accompanied_by_a_ledger_write():
    """A refusal that writes no row is invisible — the whole point of this guard.

    Fails if any ``return False, None, 0`` has no ``_record_unresolved_transcript`` in the same
    block. Deleting one of the six untested calls trips this; so does adding a new refusal
    without accounting for it.
    """
    tree = _tree()
    unaccounted = []
    for _parent, stmts in _blocks(tree):
        for i, stmt in enumerate(stmts):
            if not _is_refusal_return(stmt):
                continue
            # The recorder call is a sibling ahead of the return in the same block.
            if not _calls_recorder(stmts[:i]):
                unaccounted.append(getattr(stmt, "lineno", "?"))
    assert not unaccounted, (
        "refusal return(s) with no ledger write in the same block, at "
        f"episode_processor.py lines {sorted(unaccounted)} — the episode will land in neither "
        "ok nor failed, exactly the 2026-09-23 silent-underwork shape"
    )


def test_the_recorder_call_sites_are_not_silently_removed():
    """Guard against a mass-removal that would make the invariant above vacuous.

    The invariant passes trivially if the refusal returns are deleted along with the writes, so
    the COUNT is pinned separately.

    11, and counted by AST rather than by grep: ``grep -c '_record_unresolved_transcript('``
    reports 11 but one of those is the ``def`` line, so there were only 10 CALLS — a commit
    message of mine said "11 sites" on the strength of that grep and was wrong. The 11th is
    real now: the structural guard above found an unaccounted refusal in
    ``transcribe_media_to_text`` (no transcription provider) and it was wired up.
    """
    tree = _tree()
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == RECORDER
        and not isinstance(n.func, ast.Attribute)
    ]
    assert len(calls) >= 11, f"expected >= 11 {RECORDER} CALLS (not grep lines), found {len(calls)}"


@pytest.mark.parametrize("error_type", sorted(EXPECTED_ERROR_TYPES))
def test_each_distinct_refusal_reason_is_still_distinguishable(error_type: str):
    """One bucket for every refusal would make the ledger useless for deciding what to re-run.

    ``NoAudio`` (needs a download) and ``TooFewVoices`` (needs rediarize_only) call for different
    follow-ups, so they must not collapse into a single generic type.
    """
    tree = _tree()
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != RECORDER:
            continue
        for kw in node.keywords:
            if (
                kw.arg == "error_type"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == error_type
            ):
                found = True
        if error_type == "TranscriptUnresolved" and not any(
            kw.arg == "error_type" for kw in node.keywords
        ):
            found = True  # the default, used by the bare calls
    assert found, f"no refusal records error_type={error_type!r} any more"


def test_the_recorder_itself_keys_on_the_canonical_episode_id():
    """The key must come from the helper, never from a non-existent Episode attribute.

    An earlier accounting fix read ``getattr(episode, "guid", ...)``, which always fell through
    to the raw title: the write missed the pre-created row, appended an ORPHAN, and left the
    episode's real row reading ``ok`` — one episode recorded both ok and failed.
    """
    src = MODULE.read_text(encoding="utf-8")
    start = src.index(f"def {RECORDER}(")
    body = src[start : start + 3000]
    assert "get_episode_id_from_episode" in body, "the recorder no longer derives the canonical id"
    assert 'getattr(job.episode, "guid"' not in body, "keyed on a non-existent Episode attribute"


# ---------------------------------------------------------------------------
# M5: the ledger key must match the row orchestration ALREADY created
# ---------------------------------------------------------------------------


def test_the_abandon_row_lands_ON_the_row_orchestration_created_not_beside_it():
    """Asserting against ``get_episode_id_from_episode`` alone cannot catch the orphan.

    The existing key test derives the expected id with the SAME helper the implementation calls,
    so the two can never disagree and the test proves only self-consistency. The real orphan risk
    is one level up: ``orchestration.py`` pre-creates every episode's row by DUPLICATING the
    derivation inline (its own guid/link/date extraction feeding ``generate_episode_id``) rather
    than calling the helper. They agree field-for-field today; edit either copy alone and orphans
    come back — a ledger showing one episode both ok and failed — with nothing failing.

    So this reproduces orchestration's inline derivation, creates the row the way a real run
    does, then records an abandonment and asserts there is still exactly ONE row.
    """
    import xml.etree.ElementTree as ET

    from podcast_scraper import models
    from podcast_scraper.rss.parser import extract_episode_published_date
    from podcast_scraper.workflow import metrics as metrics_mod
    from podcast_scraper.workflow.metadata_generation import generate_episode_id
    from podcast_scraper.workflow.stages import processing
    from podcast_scraper.workflow.types import ProcessingJob

    feed_url = "https://example.com/podcast.xml"
    item = ET.Element("item")
    ET.SubElement(item, "title").text = "Ep Orphan"
    ET.SubElement(item, "guid").text = "guid-orphan-check"
    ET.SubElement(item, "link").text = "https://example.com/ep/1"
    ET.SubElement(item, "pubDate").text = "Tue, 01 Sep 2026 10:00:00 +0000"
    episode = models.Episode(
        idx=11, title="Ep Orphan", title_safe="Ep Orphan", item=item, transcript_urls=[]
    )

    # --- orchestration.py's inline derivation, reproduced verbatim in shape ---
    guid_elem = item.find("guid")
    episode_guid = guid_elem.text.strip() if guid_elem is not None and guid_elem.text else None
    link_elem = item.find("link")
    episode_link = link_elem.text.strip() if link_elem is not None and link_elem.text else None
    published = extract_episode_published_date(item)
    orchestration_id = generate_episode_id(
        feed_url=feed_url,
        episode_title=episode.title,
        episode_guid=episode_guid,
        published_date=published,
        episode_link=episode_link,
        episode_number=getattr(episode, "number", None),
    )

    pm = metrics_mod.Metrics()
    pm.get_or_create_episode_status(episode_id=orchestration_id, episode_number=episode.idx)
    pm.update_episode_status(episode_id=orchestration_id, status="ok", stage="summarized")
    assert len(pm.episode_statuses) == 1

    class _Cfg:
        rss_url = feed_url

    wrote = processing.record_abandoned_episode(
        ProcessingJob(
            episode=episode,
            transcript_path="/tmp/orphan.txt",
            transcript_source="whisper_transcription",
            detected_names=None,
            whisper_model=None,
        ),
        _Cfg(),
        pm,
        elapsed_seconds=4000.0,
        ceiling_seconds=3600.0,
    )

    assert wrote is True
    rows = [(s.episode_id, s.status) for s in pm.episode_statuses]
    assert len(pm.episode_statuses) == 1, (
        "the abandon appended an ORPHAN row instead of updating the one orchestration created — "
        f"the same episode now reads twice: {rows}"
    )
    row = pm.episode_statuses[0]
    assert row.episode_id == orchestration_id
    assert row.status == "failed", "the pre-existing 'ok' was left in place on an abandoned episode"
    assert row.error_type == "ProcessingAbandoned"
