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

COVERAGE, stated honestly. This covers the 11 resolution sites that are callable as
functions today. It does NOT yet cover A3 (`_build_speakers_from_diarized_segments`, line
1199) — its precedence is inlined mid-function and there is nothing to call. Extracting it
is part of the refactor, and it joins this golden then. Regenerate with::

    REGENERATE_TRANSCRIPT_RESOLVER_GOLDEN=1 .venv/bin/python -m pytest \
        tests/integration/workflow/test_transcript_resolver_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    run_root: Path = ep["run_root"]
    corpus_root: Path = ep["corpus_root"]
    rel: str = ep["transcript_rel"]
    row: Dict[str, Any] = {}

    # A1/A2 — GI and KG, the only two readers already using the shared resolver.
    loaded = load_processing_transcript(str(run_root), rel)
    row["A1A2_gi_kg"] = {"ref": loaded.transcript_ref, "is_adfree": loaded.is_adfree}

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
    order = segments_relpaths_for_transcript(rel)
    assert order[0] == "transcripts/e01.segments.json"
    assert _first_existing(run_root, order) == "transcripts/e01.segments.json"

    # The two really do land on different files for the same episode. That is the whole
    # reason the resolver takes a `purpose` instead of having one precedence.
    assert _first_existing(run_root, order) != "transcripts/e01.adfree.segments.json"


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
