"""End-to-end rehearsal of the corpus-integrity repair runbook on a real on-disk corpus.

WHY THIS EXISTS
``docs/guides/CORPUS_INTEGRITY_REPAIR_RUNBOOK.md`` is the procedure for repairing a production
corpus, and until now NO test drove it end to end. The prep sheet for the production run says so
in as many words: step 3 (``gi-repair``) "cannot be re-rehearsed locally: zero placeholders exist
in any local corpus", and steps 6-9 were "not rehearsed". The three modules the runbook depends
on -- ``gi/integrity``, ``gi/repair``, ``preprocessing/audit`` -- had 0.0% end-to-end coverage
while carrying 85-90% unit coverage. Unit tests pin each function; nothing checked that the
SEQUENCE an operator actually runs produces a corpus that then passes the gate.

That gap is the expensive kind. Every one of these tools reports on a corpus, and the failure
mode the whole epic was about is a tool that reports success over damage. A gate that passes
because it looked at the wrong files, or a repair that "succeeds" while writing an empty
artifact, is invisible to per-function unit tests and obvious to a sequence test.

WHAT IT REHEARSES, in runbook order, against a corpus built to look like the damaged production
one (a multi-episode run whose preprocessing failed, holding legacy placeholder artifacts,
alongside a healthy run that must be left alone):

    step 1  check_corpus_gi_integrity      -> FAIL, naming the placeholder episodes
    step 2  check_corpus_preprocessing     -> FAIL, naming the damaged run
    step 4  write_work_list                -> the --reprocess-episode-ids file
    step 3  repair_placeholder_artifacts   -> re-derive in place, same path
    step 8  check_corpus_gi_integrity      -> PASS, and the healthy episode untouched

NO PROVIDER IS NEEDED. ``repair_episode`` passes ``insight_texts=_summary_bullets(meta)`` into
the artifact builder, so a repair re-derives from the transcript plus the summary bullets the
metadata already carries. That is what makes this rehearsable offline -- and it is also the
property that makes the production repair possible without re-running summarisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.gi.corpus import LEGACY_PLACEHOLDER_INSIGHT_TEXT
from podcast_scraper.gi.integrity import assess_gi_integrity, check_corpus_gi_integrity
from podcast_scraper.gi.repair import repair_placeholder_artifacts
from podcast_scraper.preprocessing.audit import (
    assess_preprocessing,
    check_corpus_preprocessing,
    damaged_episode_ids,
    write_work_list,
)

pytestmark = [pytest.mark.e2e]


class _Cfg:
    """Minimal config: the repair reads only a model name off it."""

    summary_model = "e2e-rehearsal-model"


_TRANSCRIPT = (
    "Maya said the new trail surface cut maintenance hours nearly in half over the season. "
    "Liam replied that the drainage work mattered more than the surface material itself. "
    "They agreed the volunteer crews were the reason the project finished before winter."
)


def _write_episode(
    run: Path,
    *,
    episode_id: str,
    name: str,
    placeholder: bool,
    bullets: List[str],
    with_kg: bool = False,
) -> Path:
    """One episode: transcript + metadata + gi artifact, wired the way the pipeline writes them."""
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)

    transcript_rel = f"transcripts/{name}.txt"
    (run / transcript_rel).write_text(_TRANSCRIPT, encoding="utf-8")

    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": episode_id, "title": name, "published": "2026-02-01"},
                "feed": {"feed_id": "https://example.com/singletrack.xml"},
                "content": {"transcript_file_path": transcript_rel},
                "summary": {"bullets": bullets},
                "grounded_insights": {"artifact_path": f"metadata/{name}.gi.json"},
            }
        ),
        encoding="utf-8",
    )

    gi_path = run / "metadata" / f"{name}.gi.json"
    texts = [LEGACY_PLACEHOLDER_INSIGHT_TEXT] if placeholder else ["Drainage drove the outcome."]
    nodes: List[Dict[str, Any]] = [{"id": episode_id, "type": "Episode", "properties": {}}]
    for i, text in enumerate(texts):
        nodes.append({"id": f"{episode_id}:i{i}", "type": "Insight", "properties": {"text": text}})
    gi_path.write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": []}), encoding="utf-8"
    )

    if with_kg:
        (run / "metadata" / f"{name}.kg.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "nodes": [
                        {"id": "kg:1", "type": "Topic", "properties": {"name": "trail drainage"}}
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
    return gi_path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A corpus shaped like the damaged production one.

    Feed A: a TWO-episode run whose preprocessing was attempted and completed for neither
    (``preprocessing_attempts=2, preprocessing_count=0``) -- the #18/#558 signature -- holding two
    legacy placeholder artifacts. Two episodes on purpose: a one-episode run is the degenerate
    case where run-level metrics attribute exactly, and the audit's own docstring records that
    validating only against that case is how the damage rule was wrong four times.

    Feed B: a healthy single-episode run. It exists to catch the opposite failure -- a gate or a
    repair that touches episodes it was never pointed at.
    """
    root = tmp_path / "corpus"
    (root / "feeds").mkdir(parents=True)
    (root / ".podcast_scraper").write_text("", encoding="utf-8")

    damaged = root / "feeds" / "singletrack" / "run_20260810-090000"
    damaged.mkdir(parents=True)
    (damaged / "metrics.json").write_text(
        json.dumps(
            {
                "preprocessing_attempts": 2,
                "preprocessing_count": 0,
                "avg_preprocessing_wall_ms": 297183,
                "transcribe_count": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_episode(
        damaged,
        episode_id="ep-alpha",
        name="0001 - Alpha",
        placeholder=True,
        bullets=["Maintenance hours fell by half.", "Drainage mattered more than surface."],
        with_kg=True,
    )
    _write_episode(
        damaged,
        episode_id="ep-beta",
        name="0002 - Beta",
        placeholder=True,
        bullets=["Volunteer crews finished before winter."],
    )

    healthy = root / "feeds" / "switchback" / "run_20260812-090000"
    healthy.mkdir(parents=True)
    (healthy / "metrics.json").write_text(
        json.dumps({"preprocessing_attempts": 1, "preprocessing_count": 1, "transcribe_count": 1}),
        encoding="utf-8",
    )
    _write_episode(
        healthy,
        episode_id="ep-gamma",
        name="0003 - Gamma",
        placeholder=False,
        bullets=["A genuinely useful claim."],
    )
    return root


def test_runbook_end_to_end_damaged_corpus_becomes_clean(corpus: Path) -> None:
    """The whole sequence, in order, with the assertions an operator actually relies on."""

    # --- step 1: the integrity gate must REFUSE a corpus serving placeholders ---------------
    ok, report = check_corpus_gi_integrity(corpus)
    assert ok is False, f"gate passed a corpus holding placeholders:\n{report}"
    assert "ep-alpha" in report and "ep-beta" in report, report

    before = assess_gi_integrity(corpus)
    assert {e["episode_id"] for e in before["legacy_placeholders"]} == {"ep-alpha", "ep-beta"}
    assert [e["episode_id"] for e in before["healthy"]] == ["ep-gamma"]

    # --- step 2: the preprocessing audit must find the damaged run --------------------------
    pp_ok, pp_report = check_corpus_preprocessing(corpus)
    assert pp_ok is False, f"audit cleared a run transcribed from raw audio:\n{pp_report}"

    runs = assess_preprocessing(corpus)
    damaged_runs = [r for r in runs if r.damaged]
    assert len(damaged_runs) == 1, [r.run_dir for r in damaged_runs]
    # Two episodes, one run-level metric: attribution CANNOT be exact, and the audit must say so
    # rather than implying it knows which episode was hurt.
    assert damaged_runs[0].episodes_in_run == 2
    assert damaged_runs[0].attribution_is_exact is False

    # --- step 4: the work-list feeding --reprocess-episode-ids -------------------------------
    ids = damaged_episode_ids(corpus)
    assert ids == ["ep-alpha", "ep-beta"], ids
    assert "ep-gamma" not in ids, "the healthy run must never reach the reprocess work-list"

    worklist = corpus / "preprocessing_repair_worklist.txt"
    count = write_work_list(corpus, worklist)
    assert count == 2
    written = [
        line.strip()
        for line in worklist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert written == ["ep-alpha", "ep-beta"], written

    # --- step 3: repair the placeholders IN PLACE -------------------------------------------
    alpha_gi = corpus / "feeds/singletrack/run_20260810-090000/metadata/0001 - Alpha.gi.json"
    gamma_gi = corpus / "feeds/switchback/run_20260812-090000/metadata/0003 - Gamma.gi.json"
    gamma_before = gamma_gi.read_text(encoding="utf-8")

    audit_log = corpus / "repair_audit.jsonl"
    report_obj = repair_placeholder_artifacts(corpus, _Cfg(), audit_path=audit_log)

    assert len(report_obj.repaired) == 2, [f.error for f in report_obj.failed]
    assert report_obj.failed == []

    # Same path rewritten -- NOT a second artifact in a new run dir, which is what a pipeline
    # re-run would produce and what makes newest-wins and oldest-wins resolvers disagree.
    assert alpha_gi.is_file()
    repaired = json.loads(alpha_gi.read_text(encoding="utf-8"))
    insights = [n for n in repaired["nodes"] if n.get("type") == "Insight"]
    assert insights, "repair wrote an artifact with no insights -- empty is not repaired"
    assert not any(
        str(n["properties"].get("text", "")).strip() == LEGACY_PLACEHOLDER_INSIGHT_TEXT
        for n in insights
    ), "the placeholder survived its own repair"

    # --- the healthy episode must be byte-identical -----------------------------------------
    assert gamma_gi.read_text(encoding="utf-8") == gamma_before, "repair touched a healthy episode"

    # --- an auditable trail, one row per episode --------------------------------------------
    rows = [json.loads(x) for x in audit_log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 2, rows
    assert {r.get("episode_id") for r in rows} == {"ep-alpha", "ep-beta"}

    # --- step 8: the gate must now PASS -----------------------------------------------------
    ok_after, report_after = check_corpus_gi_integrity(corpus)
    assert ok_after is True, f"gate still failing after a successful repair:\n{report_after}"

    after = assess_gi_integrity(corpus)
    assert after["legacy_placeholders"] == []
    assert {e["episode_id"] for e in after["healthy"]} == {"ep-alpha", "ep-beta", "ep-gamma"}


def test_repair_is_idempotent_and_refuses_healthy_artifacts(corpus: Path) -> None:
    """Re-running the repair must be a no-op, not a second rewrite.

    A production repair gets re-run -- after an interruption, or because an operator is unsure it
    finished. If the second pass rewrote healthy artifacts it would destroy the very work the
    first pass did, and the gate would still say PASS.
    """
    first = repair_placeholder_artifacts(corpus, _Cfg())
    assert len(first.repaired) == 2

    second = repair_placeholder_artifacts(corpus, _Cfg())
    assert second.repaired == [], "a second pass rewrote already-repaired artifacts"
    assert second.failed == [], second.failed


def test_dry_run_reports_the_work_without_writing(corpus: Path) -> None:
    """``--dry-run`` is what an operator runs first against production. It must not write."""
    alpha_gi = corpus / "feeds/singletrack/run_20260810-090000/metadata/0001 - Alpha.gi.json"
    before = alpha_gi.read_text(encoding="utf-8")

    report = repair_placeholder_artifacts(corpus, _Cfg(), dry_run=True)

    assert len(report.skipped_dry_run) == 2, report.skipped_dry_run
    assert report.repaired == []
    assert alpha_gi.read_text(encoding="utf-8") == before, "dry run modified the corpus"

    ok, _ = check_corpus_gi_integrity(corpus)
    assert ok is False, "a dry run must leave the corpus failing the gate"


def test_a_failed_repair_leaves_the_placeholder_for_the_gate_to_find(corpus: Path) -> None:
    """Every failure path must be NON-destructive. This is what makes it safe to point at prod.

    A repair that half-writes leaves an episode that is neither the placeholder the gate would
    catch nor the artifact the corpus needs -- damage invisible to both. So each refusal is
    checked for the same two properties: the repair reports failure, AND the placeholder is
    still exactly where step 1 will find it on the next pass.
    """
    from podcast_scraper.gi.repair import repair_episode

    run = corpus / "feeds/singletrack/run_20260810-090000"
    alpha_gi = run / "metadata/0001 - Alpha.gi.json"
    original = alpha_gi.read_text(encoding="utf-8")

    # 1. metadata deleted -> cannot know the transcript path
    meta = run / "metadata/0001 - Alpha.metadata.json"
    meta_body = meta.read_text(encoding="utf-8")
    meta.unlink()
    result = repair_episode(alpha_gi, _Cfg())
    assert result.ok is False and result.error
    assert alpha_gi.read_text(encoding="utf-8") == original, "placeholder was modified on failure"
    meta.write_text(meta_body, encoding="utf-8")

    # 2. transcript missing -> nothing to re-derive from
    transcript = run / "transcripts/0001 - Alpha.txt"
    body = transcript.read_text(encoding="utf-8")
    transcript.unlink()
    result = repair_episode(alpha_gi, _Cfg())
    assert result.ok is False and result.error
    assert alpha_gi.read_text(encoding="utf-8") == original
    transcript.write_text(body, encoding="utf-8")

    # 3. transcript present but empty -> must refuse rather than build from nothing
    transcript.write_text("   \n", encoding="utf-8")
    result = repair_episode(alpha_gi, _Cfg())
    assert result.ok is False
    assert "empty" in (result.error or "").lower(), result.error
    assert alpha_gi.read_text(encoding="utf-8") == original
    transcript.write_text(body, encoding="utf-8")

    # 4. a HEALTHY artifact must be refused outright -- never rewrite good data
    gamma_gi = corpus / "feeds/switchback/run_20260812-090000/metadata/0003 - Gamma.gi.json"
    gamma_before = gamma_gi.read_text(encoding="utf-8")
    result = repair_episode(gamma_gi, _Cfg())
    assert result.ok is False
    assert "placeholder" in (result.error or "").lower(), result.error
    assert gamma_gi.read_text(encoding="utf-8") == gamma_before

    # 5. unreadable artifact -> classified, not crashed
    broken = run / "metadata/0002 - Beta.gi.json"
    broken.write_text('{"episode_id": "ep-beta", "nodes": [', encoding="utf-8")
    result = repair_episode(broken, _Cfg())
    assert result.ok is False and result.error

    # After every one of those, the corpus is still exactly as damaged as it was -- the gate
    # must still refuse it, which is the property that makes a retry safe.
    ok, _ = check_corpus_gi_integrity(corpus)
    assert ok is False


def test_the_audit_reports_an_empty_corpus_without_pretending_it_is_clean(tmp_path: Path) -> None:
    """A corpus with no runs must not read as PASS -- that is the rm-and-pass failure mode.

    ``make corpus-placeholder-check`` was retired precisely because "the bad string is absent"
    passes on a corpus whose artifacts were deleted and never regenerated. Both gates are
    checked here for the shape of their answer on nothing at all.
    """
    empty = tmp_path / "empty-corpus"
    (empty / "feeds").mkdir(parents=True)
    (empty / ".podcast_scraper").write_text("", encoding="utf-8")

    pp_ok, pp_report = check_corpus_preprocessing(empty)
    assert "metrics.json" in pp_report, pp_report

    result = assess_gi_integrity(empty)
    assert result["metadata_scanned"] == 0, result
    assert result["healthy"] == []
