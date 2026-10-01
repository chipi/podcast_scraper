"""Golden: which transcript variant does each reader actually resolve? (#2170 / S2.1a)

The instrument the transcript-resolver refactor is measured by. Every reader below picks
between ``<base>.txt`` and ``<base>.adfree.txt`` (or their segment sidecars) with its own
inline precedence, and the precedences DISAGREE — correctly, because the analysis readers
need the coordinate space GI's ``char_start`` lives in while the player needs the timeline
the unbridged audio runs on. A refactor that unified them into one precedence would break
one side silently: a plausible wrong segment looks exactly like a right one.

So this records the current answer per reader per episode, and the refactor must not move a
single row. `docs/wip/2170-TRANSCRIPT-RESOLVER-INVENTORY.md` carries the full inventory and
the group letters used here.

Two fixture corpora, chosen because together they cover both branches of every precedence:

- ``viewer-validation-corpus/v3`` — 40 episodes, EVERY one has ``.adfree.*``, flat layout
  (the feed dir *is* the run root, no ``run_*``)
- ``app-validation-corpus/v3`` — 40 episodes, NONE has ``.adfree.*``, real ``run_*`` dirs

COVERAGE, stated honestly. This covers every resolution site the refactor routed, including
A3 — whose precedence used to be inlined mid-function with nothing to call, and which is
pinned here by its OUTPUT (the speaker record) rather than by a path it no longer chooses.
The one regression proven by assertion rather than injection is called out at the bottom.

The tests at the end INJECT each regression the golden exists to catch and assert which rows
move. A golden that has only ever been green is indistinguishable from no golden — see
`docs/architecture/MULTILINGUAL_ARC.md` §3.1. Regenerate with::

    REGENERATE_TRANSCRIPT_RESOLVER_GOLDEN=1 .venv/bin/python -m pytest \
        tests/integration/workflow/test_transcript_resolver_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast, Dict, List, Optional

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]
_GOLDEN = _REPO / "tests" / "fixtures" / "goldens" / "transcript_resolver.golden.json"
_CORPORA = {
    "viewer-validation-corpus/v3": _REPO / "tests/fixtures/viewer-validation-corpus/v3",
    "app-validation-corpus/v3": _REPO / "tests/fixtures/app-validation-corpus/v3",
}


def _rel(path: Optional[Path], base: Path) -> Optional[str]:
    """``path`` as a POSIX relpath under ``base``, or None. Keeps the golden machine-agnostic."""
    if path is None:
        return None
    try:
        return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()
    except ValueError:
        return str(path)


def _episodes() -> List[Dict[str, Any]]:
    """Every episode in both corpora, with the paths the readers need.

    ``run_root`` is the directory ``transcript_file_path`` is relative to: the metadata
    file's grandparent, flat layout and ``run_*`` layout alike.
    """
    out: List[Dict[str, Any]] = []
    for corpus_name, corpus_root in sorted(_CORPORA.items()):
        for meta_path in sorted(corpus_root.glob("feeds/**/metadata/*.metadata.json")):
            try:
                doc = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            content = doc.get("content") or {}
            rel = content.get("transcript_file_path")
            if not isinstance(rel, str) or not rel.strip():
                continue
            out.append(
                {
                    "corpus": corpus_name,
                    "corpus_root": corpus_root,
                    "run_root": meta_path.parent.parent,
                    "meta_path": meta_path,
                    "doc": doc,
                    "transcript_rel": rel.strip(),
                    "key": f"{corpus_name}::{_rel(meta_path, corpus_root)}",
                }
            )
    return out


def _sidecar_variant_of(text: str, run_root: Path, transcript_rel: str) -> str:
    """Which SEGMENTS sidecar the audit's opening came out of.

    ``_transcript_opening`` reads ``.segments.json`` / ``.adfree.segments.json`` /
    ``.transcript.json`` — not the rendered ``.txt``. Comparing against the ``.txt`` bodies
    said "raw+adfree" for every episode, because the same sentence is in both renderings:
    a discriminator that cannot discriminate. This looks in the files it actually reads,
    and reports ambiguity as ambiguity rather than picking a side.
    """
    if not text:
        return "empty"
    base = (run_root / transcript_rel).with_suffix("")
    found = []
    for label, suffix in (("raw", ".segments.json"), ("adfree", ".adfree.segments.json")):
        p = base.with_name(base.name + suffix)
        try:
            if p.is_file() and text in p.read_text(encoding="utf-8"):
                found.append(label)
        except OSError:
            continue
    if not found:
        return "neither"
    return found[0] if len(found) == 1 else "ambiguous:" + "+".join(found)


def _first_existing(run_root: Path, relpaths: List[str]) -> Optional[str]:
    """The candidate a caller iterating ``relpaths`` would actually open.

    ``segments_relpaths_for_transcript`` is pure — it returns an ORDER, and the resolution
    is that order intersected with what is on disk. Recording only the order would have
    missed that the viewer corpus has no raw ``.segments.json`` at all, so the player there
    falls through to the ad-free sidecar.
    """
    for rel in relpaths:
        if (run_root / rel).is_file():
            return rel
    return None


def _probe(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Every callable resolution site, as it behaves today. No reader is modified here."""
    from podcast_scraper.capability_audit import _transcript_opening
    from podcast_scraper.gi.load import _transcript_path_from_artifact_path
    from podcast_scraper.gi.repair import _segments_for, _transcript_text_for
    from podcast_scraper.search.indexer import _transcript_path
    from podcast_scraper.server.routes.corpus_text_file import (
        _resolve_readable_file_under_corpus,
    )
    from podcast_scraper.server.segments_view import segments_relpaths_for_transcript
    from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import _segments_sidecar
    from podcast_scraper.workflow.adfree_transcript import load_processing_transcript
    from podcast_scraper.workflow.metadata_generation import (
        _build_speakers_from_diarized_segments,
    )

    run_root: Path = ep["run_root"]
    corpus_root: Path = ep["corpus_root"]
    rel: str = ep["transcript_rel"]
    row: Dict[str, Any] = {}

    # A1/A2 — GI and KG, the only two readers already using the shared resolver.
    #
    # `has_text` is here because a fault-injection test proved the row blind without it:
    # when nothing resolves, `load_transcript` reports the CANONICAL relpath as its ref with
    # `is_adfree=False` — which on a corpus with no ad-free bodies is byte-identical to a
    # successful raw load. So deleting the fallback moved five other readers' rows and left
    # this one green. The text is not recorded (it is megabytes); whether there IS text is.
    loaded = load_processing_transcript(str(run_root), rel)
    row["A1A2_gi_kg"] = {
        "ref": loaded.transcript_ref,
        "is_adfree": loaded.is_adfree,
        "has_text": bool(loaded.text),
        "has_segments": loaded.segments is not None,
    }

    # A3 — the speaker record built from the diarized sidecar. Its precedence used to be
    # inlined mid-function with nothing to call; it goes through the resolver now, so what
    # is worth pinning is its OUTPUT, not a path it no longer chooses for itself.
    speakers, num_speakers = _build_speakers_from_diarized_segments(str(run_root), rel, None)
    row["A3_speaker_record"] = {
        "num_speakers": num_speakers,
        "named": sorted({s.name for s in speakers or [] if getattr(s, "name", None)}),
    }

    # A4 — search indexer.
    row["A4_search_indexer"] = _rel(_transcript_path(run_root, ep["doc"]), run_root)

    # A5 — GI repair text. Returns (text, ref); ref is a basename.
    try:
        _text, ref = _transcript_text_for(run_root, rel)
        row["A5_gi_repair_text"] = ref
    except OSError:
        row["A5_gi_repair_text"] = None

    # A6 — GI repair segments. Returns the list, not a path, so discriminate by shape:
    # only the ad-free sidecar carries char offsets.
    segs = _segments_for(run_root, rel)
    if segs is None:
        row["A6_gi_repair_segments"] = None
    else:
        first = segs[0] if segs and isinstance(segs[0], dict) else {}
        row["A6_gi_repair_segments"] = {
            "count": len(segs),
            "shape": "adfree" if "char_start" in first else "raw",
        }

    # A8 — the m0009 speaker-role backfill.
    row["A8_m0009_sidecar"] = _rel(
        _segments_sidecar(ep["meta_path"], ep["doc"].get("content")), run_root
    )

    # A9 — capability audit's transcript opening. Returns text; discriminate by variant.
    # This is the one analysis reader whose precedence is INVERTED (raw before ad-free).
    opening = _transcript_opening(corpus_root, _rel(ep["meta_path"], corpus_root) or "")
    row["A9_capability_audit_opening_variant"] = _sidecar_variant_of(opening, run_root, rel)

    # A10 — GI evidence loading. Always raw, while the offsets it slices with are ad-free:
    # the live coordinate-space bug this slice fixes.
    gi_artifact = ep["meta_path"].with_name(
        ep["meta_path"].name.replace(".metadata.json", ".gi.json")
    )
    row["A10_gi_load_evidence"] = _rel(_transcript_path_from_artifact_path(gi_artifact), run_root)

    # B1 — player segments contract. Pure; raw-first is deliberate (unbridged audio).
    # Record the order AND what a caller would really open, since the order alone hides
    # the fall-through when the raw sidecar is absent.
    b1_order = segments_relpaths_for_transcript(rel)
    row["B1_player_segments_order"] = b1_order
    row["B1_player_segments_resolved"] = _first_existing(run_root, b1_order)

    # B2 — viewer text route, asked for the raw relpath.
    resolved = _resolve_readable_file_under_corpus(
        corpus_root, (_rel(run_root, corpus_root) or "") + "/" + rel
    )
    row["B2_viewer_text_route"] = resolved[1] if resolved else None

    # C1/C2/C3 — summary and both faithfulness checks: a bare join, no variant logic.
    row["C_summary_faithfulness"] = rel

    return row


def _build() -> Dict[str, Any]:
    eps = _episodes()
    assert eps, "no episodes found in either fixture corpus — the golden would be vacuous"
    rows = {ep["key"]: _probe(ep) for ep in eps}

    # A7 — recurrent-host scan, per FEED root rather than per episode: it globs
    # `run_*/metadata/*` itself. The viewer corpus is flat, so it legitimately finds
    # nothing there — that asymmetry is part of what the golden pins.
    from podcast_scraper.workflow.stages.processing import _newest_run_transcripts

    a7: Dict[str, List[str]] = {}
    for corpus_name, corpus_root in sorted(_CORPORA.items()):
        for feed_root in sorted((corpus_root / "feeds").glob("*")):
            if not feed_root.is_dir():
                continue
            found = _newest_run_transcripts(feed_root)
            a7[f"{corpus_name}::{feed_root.name}"] = sorted(p.name for p in found)

    return {
        "_readme": (
            "Which transcript variant each reader resolves, per episode. Generated by "
            "tests/integration/workflow/test_transcript_resolver_golden.py. A refactor that "
            "changes any row has changed behaviour — see "
            "docs/wip/2170-TRANSCRIPT-RESOLVER-INVENTORY.md."
        ),
        "A7_recurrent_host_scan_per_feed": a7,
        "per_episode": rows,
    }


def test_transcript_resolver_golden() -> None:
    actual = _build()
    if os.environ.get("REGENERATE_TRANSCRIPT_RESOLVER_GOLDEN"):
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"regenerated {_GOLDEN.relative_to(_REPO)}")
    assert _GOLDEN.is_file(), (
        f"{_GOLDEN.relative_to(_REPO)} is missing — regenerate with "
        "REGENERATE_TRANSCRIPT_RESOLVER_GOLDEN=1"
    )
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert actual["per_episode"] == expected["per_episode"]
    assert actual["A7_recurrent_host_scan_per_feed"] == expected["A7_recurrent_host_scan_per_feed"]


def _write_episode_with_every_variant(root: Path) -> str:
    """An episode carrying ALL FOUR artifacts. Returns ``transcript_file_path``.

    Neither committed corpus has both segment sidecars — the viewer one ships only
    ``.adfree.segments.json``, the app one only ``.segments.json``. So for the segments
    readers the golden above records *which file exists*, never *which is preferred*, and
    the preference is precisely what diverges: B1 must take raw, A6/A8/A9 must take
    ad-free. This builds the state that tells them apart.
    """
    tr = root / "transcripts"
    md = root / "metadata"
    tr.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)

    raw_text = "Sponsor: buy things at example dot com.\nMaya: Welcome back to the show.\n"
    adfree_text = "Maya: Welcome back to the show.\n"
    (tr / "e01.txt").write_text(raw_text, encoding="utf-8")
    (tr / "e01.adfree.txt").write_text(adfree_text, encoding="utf-8")

    # Raw sidecar: the diarizer's shape — carries `id`, no char offsets, ad segment included.
    (tr / "e01.segments.json").write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "text": "Sponsor: buy things at example dot com.",
                    "speaker_label": "Maya",
                },
                {
                    "id": 1,
                    "start": 4.0,
                    "end": 8.0,
                    "text": "Welcome back to the show.",
                    "speaker_label": "Maya",
                },
            ]
        ),
        encoding="utf-8",
    )
    # Ad-free sidecar: carries char offsets into `adfree_text`, ad segment dropped, no `id`.
    (tr / "e01.adfree.segments.json").write_text(
        json.dumps(
            [
                {
                    "start": 4.0,
                    "end": 8.0,
                    "text": "Welcome back to the show.",
                    "speaker_label": "Maya",
                    "char_start": adfree_text.index("Welcome"),
                    "char_end": adfree_text.index("Welcome") + len("Welcome back to the show."),
                }
            ]
        ),
        encoding="utf-8",
    )
    (tr / "e01.adfree.admap.json").write_text(json.dumps({"excised": []}), encoding="utf-8")
    (md / "e01.metadata.json").write_text(
        json.dumps({"content": {"transcript_file_path": "transcripts/e01.txt"}}),
        encoding="utf-8",
    )
    return "transcripts/e01.txt"


def test_precedence_when_both_segment_sidecars_exist(tmp_path: Path) -> None:
    """The divergence itself, asserted rather than snapshotted.

    Explicit assertions, not a golden row: this is the contract the refactor must keep,
    and a snapshot would happily record it changing.
    """
    from podcast_scraper.gi.repair import _segments_for
    from podcast_scraper.server.segments_view import segments_relpaths_for_transcript
    from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import _segments_sidecar
    from podcast_scraper.workflow.adfree_transcript import load_processing_transcript

    run_root = tmp_path / "feeds" / "pX"
    rel = _write_episode_with_every_variant(run_root)
    meta_path = run_root / "metadata" / "e01.metadata.json"
    content = {"transcript_file_path": rel}

    # ANALYSIS readers take the ad-free variant: offsets must match the text GI indexed.
    loaded = load_processing_transcript(str(run_root), rel)
    assert loaded.is_adfree
    assert loaded.transcript_ref == "transcripts/e01.adfree.txt"

    segs = _segments_for(run_root, rel)
    assert segs is not None and len(segs) == 1, "GI repair must take the ad-free sidecar"
    assert "char_start" in segs[0]

    sidecar = _segments_sidecar(meta_path, content)
    assert sidecar is not None and sidecar.name == "e01.adfree.segments.json"

    # TIMELINE reader takes the raw variant: the player streams unbridged audio, so the
    # ad-free sidecar (one segment short here, minutes short in reality) would drift it.
    # S2.1b puts the English sidecar ahead of both — a LANGUAGE choice, not a timing one, since
    # the English render carries the source segments' times. The contract this test exists for
    # is RAW-before-AD-FREE, so that is asserted on the source-language candidates directly
    # rather than being read off position 0 where an added head can quietly displace it.
    order = segments_relpaths_for_transcript(rel)
    assert order[0] == "transcripts/e01.en.segments.json"
    source_order = [c for c in order if ".en." not in c]
    assert source_order[0] == "transcripts/e01.segments.json"
    assert source_order.index("transcripts/e01.segments.json") < source_order.index(
        "transcripts/e01.adfree.segments.json"
    )
    assert _first_existing(run_root, order) == "transcripts/e01.segments.json"

    # The two really do land on different files for the same episode. That is the whole
    # reason the resolver takes a `purpose` instead of having one precedence.
    assert _first_existing(run_root, order) != "transcripts/e01.adfree.segments.json"


def test_gi_evidence_reads_the_text_its_offsets_index(tmp_path: Path) -> None:
    """The A10 fix, asserted rather than left to a snapshot row.

    ``get_evidence_span`` slices with ``char_start``/``char_end`` that GI computed against
    ``.adfree.txt``. Reading the raw body instead returned the right NUMBER of characters
    from the wrong place — plausible text, no error. A golden row alone would happily record
    this regressing, so the offsets are exercised end to end here.
    """
    from podcast_scraper.gi.load import (
        _transcript_path_from_artifact_path,
        get_evidence_span,
        load_transcript_for_evidence,
    )

    run_root = tmp_path / "feeds" / "pX"
    _write_episode_with_every_variant(run_root)
    artifact = run_root / "metadata" / "e01.gi.json"

    resolved = _transcript_path_from_artifact_path(artifact)
    assert resolved.name == "e01.adfree.txt"

    text = load_transcript_for_evidence(resolved)
    assert text is not None

    # The offsets the ad-free sidecar carries for its only segment.
    segs = json.loads((run_root / "transcripts" / "e01.adfree.segments.json").read_text())
    span = get_evidence_span(text, segs[0]["char_start"], segs[0]["char_end"])
    assert span.excerpt == "Welcome back to the show."

    # Against the raw body the very same offsets land inside the sponsor line, mid-word:
    # "r: buy things at example ". Asserted so the fix cannot be read as cosmetic — this is
    # what `gi inspect` printed as an insight's evidence on any ad-excised episode. Note it
    # is not even a sentence, and still nothing raised.
    raw_text = (run_root / "transcripts" / "e01.txt").read_text(encoding="utf-8")
    displaced = get_evidence_span(raw_text, segs[0]["char_start"], segs[0]["char_end"])
    assert displaced.excerpt == "r: buy things at example "
    assert displaced.excerpt != span.excerpt


def test_golden_covers_both_precedence_branches() -> None:
    """A golden over one branch is not a regression guard.

    The ad-free precedence has two outcomes and the corpora must exercise both, or a
    refactor that deleted the fallback would pass.
    """
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    flags = {row["A1A2_gi_kg"]["is_adfree"] for row in expected["per_episode"].values()}
    assert flags == {True, False}, f"only saw is_adfree={flags}; need both branches covered"

    shapes = {
        row["A6_gi_repair_segments"]["shape"]
        for row in expected["per_episode"].values()
        if row["A6_gi_repair_segments"]
    }
    assert shapes == {"adfree", "raw"}, f"segment sidecar shapes covered: {shapes}"


# ---------------------------------------------------------------------------
# Proving the golden fires. See MULTILINGUAL_ARC.md §3.1: every assertion above
# is satisfiable by a golden that can never fail, and one that has only ever
# been green is indistinguishable from no golden. So each regression the golden
# exists to catch gets injected here, and the test asserts WHICH rows move.
# ---------------------------------------------------------------------------


def _committed() -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(_GOLDEN.read_text(encoding="utf-8")))


def _moved_fields(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, int]:
    """Which per-episode fields differ, and on how many episodes.

    The same comparison used by hand before regenerating the golden for the A10 fix — kept
    here so "one field, forty episodes" is a thing tests can assert rather than a claim.
    """
    moved: Dict[str, int] = {}
    for key, arow in actual["per_episode"].items():
        erow = expected["per_episode"][key]
        for field in arow:
            if arow[field] != erow[field]:
                moved[field] = moved.get(field, 0) + 1
    return moved


def _app_corpus_episode_count() -> int:
    """How many episodes the app corpus holds. Counted, never written as a literal."""
    return len(
        [
            m
            for m in (_REPO / "tests/fixtures/app-validation-corpus/v3").glob(
                "feeds/*/**/metadata/*.metadata.json"
            )
        ]
    )


def _translated_app_episodes_with_an_adfree_body() -> int:
    """App-corpus episodes whose ad-free body is the ENGLISH one (`.en.adfree.txt`).

    These are the episodes RFC-124 gave an analysis base to. They are the difference between "every
    app episode falls back to the raw body" (true before translation landed in the corpus) and
    "every app episode that has no ad-free body does" (true now).
    """
    total = 0
    app_root = _REPO / "tests/fixtures/app-validation-corpus/v3"
    for ep in _episodes():
        if app_root not in ep["meta_path"].parents:
            continue
        src = ep["run_root"] / ep["transcript_rel"]
        if not src.name.endswith(".txt"):
            continue
        stem = src.name[: -len(".txt")]
        if src.with_name(stem + ".en.adfree.txt").is_file():
            total += 1
    return total


def _episodes_with_an_adfree_body() -> int:
    """Episodes where the ad-free and raw bodies are DIFFERENT files, so purpose is observable.

    Counts an English ad-free body (`.en.adfree.txt`) as well as a source one: once p10-p14's
    renders landed, the translated app episodes gained an ad-free body too, and a reader switching
    purpose moves on them for exactly the same reason it moves on the viewer corpus.

    Derived rather than written as `40`, which is what that literal used to say — the number is a
    property of the corpora and changes whenever a fixture is translated or an ad-free body is
    added, neither of which is a statement about the resolver.
    """
    total = 0
    for ep in _episodes():
        src = ep["run_root"] / ep["transcript_rel"]
        if not src.name.endswith(".txt"):
            continue
        stem = src.name[: -len(".txt")]
        if (
            src.with_name(stem + ".adfree.txt").is_file()
            or src.with_name(stem + ".en.adfree.txt").is_file()
        ):
            total += 1
    return total


def _moved_episodes(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, List[str]]:
    """Which EPISODES differ, and in which fields.

    The per-episode counterpart of :func:`_moved_fields`. Needed once the corpus contains both
    translated and untranslated episodes: "one field moved on five episodes" and "one field moved
    on five of the WRONG episodes" are the same number.
    """
    moved: Dict[str, List[str]] = {}
    for key, arow in actual["per_episode"].items():
        erow = expected["per_episode"][key]
        diff = sorted(f for f in arow if arow[f] != erow[f])
        if diff:
            moved[key] = diff
    return moved


def _translated_episode_keys() -> List[str]:
    """Episode keys whose corpus artifacts include an `.en.txt` — derived, never listed.

    Derived from disk so that translating a sixth fixture episode is a fixture change rather than
    a test edit, and so this cannot silently disagree with what the corpus actually holds.
    """
    keys = []
    for ep in _episodes():
        src = ep["run_root"] / ep["transcript_rel"]
        if (
            src.name.endswith(".txt")
            and src.with_name(src.name[: -len(".txt")] + ".en.txt").is_file()
        ):
            keys.append(ep["key"])
    return sorted(keys)


def test_golden_catches_a_reader_switching_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression the resolver exists to prevent: an analysis reader served the timeline.

    Only the search indexer is flipped, so the assertion is not merely "something changed" —
    it is that the golden points at the ONE reader and says how many episodes.
    """
    from podcast_scraper.search import indexer
    from podcast_scraper.workflow import transcript_resolution as tr

    def forced_timeline(output_dir, relpath, *, purpose, include_cleaned=False):  # type: ignore[no-untyped-def]
        return tr.resolve_text_path(
            output_dir,
            relpath,
            purpose=tr.TranscriptPurpose.TIMELINE,
            include_cleaned=include_cleaned,
        )

    monkeypatch.setattr(indexer, "resolve_text_path", forced_timeline)
    moved = _moved_fields(_build(), _committed())
    assert set(moved) == {"A4_search_indexer"}, f"expected only A4 to move, got {moved}"
    # Only the episodes whose ad-free and raw bodies are DIFFERENT files can move: where the two
    # purposes resolve the same file, flipping purpose is unobservable — which is why a golden over
    # one corpus would have proved nothing. Counted, because the set grew when p10-p14 were
    # translated and gained an `.en.adfree.txt`.
    expected_a4 = _episodes_with_an_adfree_body()
    assert moved["A4_search_indexer"] == expected_a4, (
        f"{moved['A4_search_indexer']} episodes moved but {expected_a4} have a distinct ad-free "
        "body"
    )


def test_golden_catches_the_adfree_fallback_being_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the fallback breaks the pre-#974 corpus, and the golden must say so.

    A resolver that returned only its first choice would look correct on any corpus where
    every episode has an ad-free body — i.e. on the viewer fixtures alone.
    """
    from podcast_scraper.workflow import transcript_resolution as tr

    real = tr.text_relpath_candidates

    def first_source_choice_only(relpath, *, purpose, include_cleaned=False):  # type: ignore[no-untyped-def]
        """Keep the English head and the FIRST source-language candidate, dropping the fallback.

        Truncating to ``got[:1]`` — what this did before S2.1b — now also removes the English
        head's successor, so EVERY episode breaks instead of only the app-corpus ones that
        actually depend on the fallback, and the number stops meaning anything. An injected
        fault has to be the one the test names, and nothing else.
        """
        got = real(relpath, purpose=purpose, include_cleaned=include_cleaned)
        english = [c for c in got if ".en." in c]
        source = [c for c in got if ".en." not in c]
        return english + source[:1]

    monkeypatch.setattr(tr, "text_relpath_candidates", first_source_choice_only)
    moved = _moved_fields(_build(), _committed())
    assert moved, "deleting the fallback moved nothing — the golden is not load-bearing"
    # Every app-corpus episode WITHOUT an ad-free body loses the raw fallback. It used to be
    # every episode full stop, because none of them had one; p10-p14's English renders carry an
    # `.en.adfree.txt`, so those five resolve to an ad-free body and have no fallback to lose.
    # Counted rather than written as a literal — it was `40`, went stale when `p10` landed, and
    # the number is a property of the corpus rather than a statement about the resolver.
    expected = _app_corpus_episode_count() - _translated_app_episodes_with_an_adfree_body()
    assert "A1A2_gi_kg" in moved
    assert moved["A1A2_gi_kg"] == expected, (
        f"{moved['A1A2_gi_kg']} episodes moved but the app corpus has {expected} — the fault "
        "injected here should break every one of them"
    )


def test_removing_the_english_head_moves_EXACTLY_the_translated_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two-language golden this replaced a tripwire to become.

    WHAT THIS USED TO BE. `test_the_english_branch_is_inert_on_this_corpus` asserted that removing
    the English head moved NOTHING, because no fixture episode had a `.en.*` artifact. Its own
    docstring called itself a tripwire and said that the moment one gained them, the fix was to
    convert it into a real two-language golden rather than to relax it. That moment arrived when
    p10-p14's renders were captured into the corpus, so this is that conversion.

    WHY THE OLD SHAPE WAS WORTH REPLACING RATHER THAN DELETING. An inert branch and a broken one
    are indistinguishable on an all-English corpus, so the old test could never have caught the
    English branch resolving wrongly — only that it was not firing. This asserts the branch fires
    on exactly the episodes that have a render and on no others, which is the claim that actually
    protects D-38.

    THE PARTITION IS THE POINT. `moved` must equal the translated set exactly:

    * a translated episode that does NOT move means the English branch is not resolving its
      render — the corpus would claim `translation_status: translated` while every reader still
      served the source body, which is the `.en.adfree.txt` fall-through trap in
      `translation_stage.py` and is strictly worse than having no render at all;
    * an UNTRANSLATED episode that DOES move means the branch reaches episodes with no render,
      which is S2.1b's pure-addition claim broken.
    """
    from podcast_scraper.workflow import transcript_resolution as tr

    real = tr.text_relpath_candidates

    def source_candidates_only(relpath, *, purpose, include_cleaned=False):  # type: ignore[no-untyped-def]
        got = real(relpath, purpose=purpose, include_cleaned=include_cleaned)
        return [c for c in got if ".en." not in c]

    monkeypatch.setattr(tr, "text_relpath_candidates", source_candidates_only)
    built = _build()
    committed = _committed()

    translated = _translated_episode_keys()
    assert translated, (
        "no fixture episode has an `.en.txt`, so this golden is vacuous — if the corpus lost its "
        "renders, fix the corpus rather than this test"
    )

    moved = _moved_episodes(built, committed)
    # `B1_player_segments_order` is the candidate ORDER, not a resolved path: it legitimately
    # shrinks on every episode when the head is removed. Judge only the rows that name a file.
    resolved_moves = {
        k: [f for f in fields if f != "B1_player_segments_order"] for k, fields in moved.items()
    }
    resolved_moves = {k: v for k, v in resolved_moves.items() if v}

    assert sorted(resolved_moves) == translated, (
        "the English branch fired on a different set of episodes than the ones carrying a "
        f"render.\n  moved:      {sorted(resolved_moves)}\n  translated: {translated}"
    )

    # And the direction: with the head removed each one falls back to its SOURCE body.
    for key in translated:
        before = committed["per_episode"][key]["A1A2_gi_kg"]["ref"]
        after = built["per_episode"][key]["A1A2_gi_kg"]["ref"]
        assert ".en." in before, f"{key}: the committed golden does not resolve English: {before}"
        assert ".en." not in after, f"{key}: head removed but still resolving English: {after}"


def test_golden_catches_the_gi_load_fix_being_reverted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reverting A10 must move A10 and nothing else.

    This row is the one with real evidence behind it: the fix moved exactly this field on
    exactly these episodes when it landed. Injecting the revert closes the loop.
    """
    from podcast_scraper.gi import load as gi_load

    monkeypatch.setattr(gi_load, "resolve_text_path", lambda *a, **k: None)
    moved = _moved_fields(_build(), _committed())
    assert set(moved) == {"A10_gi_load_evidence"}, f"expected only A10 to move, got {moved}"
    # Every app-corpus episode, counted rather than hardcoded — `40` went stale when p10 landed.
    expected_a10 = _app_corpus_episode_count()
    assert (
        moved["A10_gi_load_evidence"] == expected_a10
    ), f"{moved['A10_gi_load_evidence']} episodes moved but the app corpus has {expected_a10}"


def test_the_sidecar_mismatch_regression_is_covered_by_assertion_not_injection() -> None:
    """Honest note in executable form.

    The fourth regression in the arc doc's table — ``load_transcript`` resolving its sidecar
    independently of the body it loaded — is NOT proven by injection here. Forcing that
    mismatch means reimplementing the wrong version, which tests the reimplementation rather
    than the guard. It is covered by direct assertion in
    ``test_transcript_resolution.py::test_segments_always_come_from_the_body_that_was_loaded``,
    which pairs telltale sidecar contents with each body and checks they match.

    This test exists so the gap is recorded where someone reading the proofs will see it.
    """
    from tests.unit.podcast_scraper.workflow import test_transcript_resolution as unit

    assert hasattr(
        unit.TestLoadTranscript, "test_segments_always_come_from_the_body_that_was_loaded"
    )
