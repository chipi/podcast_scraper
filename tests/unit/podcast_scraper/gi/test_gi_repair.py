"""gi-repair must rewrite the SAME artifact, and must never leave a half-repaired episode.

Context (2026-08-16 rehearsal on a copy of a real corpus): no pipeline flag combination can
re-derive a placeholder episode — the skip predicates key on transcript/metadata PRESENCE and
never look at GI, so a placeholder reads as "done". And even once forcing works, a pipeline run
writes into a fresh ``run_<ts>/`` dir, producing a SECOND artifact while the placeholder
survives — a duplicate that ``_scan_corpus_metadata_index`` (oldest-wins) and search's
``merged_episode_gi_paths`` (newest-wins) resolve differently.

Hence: standalone, in-place, same path. These tests pin the two properties that make it safe to
point at a production corpus — it rewrites exactly one path per episode, and a failure leaves
the placeholder exactly where the integrity gate will find it again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.gi.corpus import LEGACY_PLACEHOLDER_INSIGHT_TEXT
from podcast_scraper.gi.repair import (
    repair_episode,
    repair_placeholder_artifacts,
)

pytestmark = [pytest.mark.unit]


class _Cfg:
    summary_model = "test-model"


def _make_episode(
    corpus: Path,
    *,
    episode_id: str = "ep-1",
    name: str = "0001 - Episode",
    placeholder: bool = True,
    with_transcript: bool = True,
    with_metadata: bool = True,
    with_kg: bool = False,
) -> Path:
    run = corpus / "feeds" / "feed_a" / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)

    transcript_rel = f"transcripts/{name}.txt"
    if with_transcript:
        (run / transcript_rel).write_text(
            "Alice said the platform reduces latency substantially. "
            "Bob replied that cost matters more than speed.",
            encoding="utf-8",
        )

    if with_metadata:
        (run / "metadata" / f"{name}.metadata.json").write_text(
            json.dumps(
                {
                    "episode": {"episode_id": episode_id, "title": name, "published": "2026-01-01"},
                    "feed": {"feed_id": "https://example.com/feed.xml"},
                    "content": {"transcript_file_path": transcript_rel},
                    "summary": {"bullets": ["Latency dropped.", "Cost beats speed."]},
                    "grounded_insights": {"artifact_path": f"metadata/{name}.gi.json"},
                }
            ),
            encoding="utf-8",
        )

    gi_path = run / "metadata" / f"{name}.gi.json"
    texts = [LEGACY_PLACEHOLDER_INSIGHT_TEXT] if placeholder else ["A real insight."]
    gi_path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "nodes": [{"id": episode_id, "type": "Episode", "properties": {}}]
                + [
                    {"id": f"{episode_id}:i{i}", "type": "Insight", "properties": {"text": t}}
                    for i, t in enumerate(texts)
                ]
                + [{"id": "topic:old-slug", "type": "Topic", "properties": {"label": "old slug"}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    if with_kg:
        (run / "metadata" / f"{name}.kg.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "nodes": [
                        {
                            "id": "kgtopic:latency",
                            "type": "Topic",
                            "properties": {"label": "Latency"},
                        },
                        {"id": "kgtopic:cost", "type": "Topic", "properties": {"label": "Cost"}},
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
    return gi_path


def _fake_build(insight_texts: List[str]):
    """A build_artifact double returning a healthy artifact."""

    def _build(episode_id: str, transcript_text: str, **kw: Any) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = [{"id": episode_id, "type": "Episode", "properties": {}}]
        for i, t in enumerate(insight_texts):
            nodes.append(
                {"id": f"{episode_id}:new{i}", "type": "Insight", "properties": {"text": t}}
            )
        return {"episode_id": episode_id, "nodes": nodes, "edges": []}

    return _build


def _write_json(path: Path, payload: Dict[str, Any], validate: bool = True) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_it_rewrites_the_same_path_not_a_new_run_dir(tmp_path):
    """THE property that makes this safe: one artifact per episode, before and after."""
    gi_path = _make_episode(tmp_path)
    before_paths = sorted(p.name for p in gi_path.parent.glob("*.gi.json"))

    result = repair_episode(
        gi_path, _Cfg(), build_fn=_fake_build(["Real one.", "Real two."]), write_fn=_write_json
    )

    assert result.ok, result.error
    assert sorted(p.name for p in gi_path.parent.glob("*.gi.json")) == before_paths
    assert len(list(tmp_path.rglob("*.gi.json"))) == 1, "no second artifact anywhere in the corpus"

    doc = json.loads(gi_path.read_text(encoding="utf-8"))
    texts = [n["properties"]["text"] for n in doc["nodes"] if n.get("type") == "Insight"]
    assert texts == ["Real one.", "Real two."]
    assert LEGACY_PLACEHOLDER_INSIGHT_TEXT not in texts
    assert result.insights_before == 1 and result.insights_after == 2


def test_a_build_failure_leaves_the_placeholder_untouched(tmp_path):
    """Fail loudly, never partially: the episode must stay on the gate's red list."""
    gi_path = _make_episode(tmp_path)
    before = gi_path.read_bytes()

    def _explode(*_a: Any, **_k: Any) -> Dict[str, Any]:
        raise RuntimeError("provider down")

    result = repair_episode(gi_path, _Cfg(), build_fn=_explode, write_fn=_write_json)

    assert result.ok is False
    assert "provider down" in (result.error or "")
    assert gi_path.read_bytes() == before, "a failed repair must not modify the artifact"


def test_a_write_failure_leaves_the_placeholder_untouched(tmp_path):
    gi_path = _make_episode(tmp_path)
    before = gi_path.read_bytes()

    def _bad_write(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    result = repair_episode(gi_path, _Cfg(), build_fn=_fake_build(["Real."]), write_fn=_bad_write)

    assert result.ok is False
    assert gi_path.read_bytes() == before


def test_it_refuses_to_rewrite_a_healthy_artifact(tmp_path):
    """Belt and braces: the work-list should never contain one, but if it did, do not touch it."""
    gi_path = _make_episode(tmp_path, placeholder=False)
    before = gi_path.read_bytes()

    result = repair_episode(
        gi_path, _Cfg(), build_fn=_fake_build(["Something else."]), write_fn=_write_json
    )

    assert result.ok is False
    assert "not a legacy placeholder" in (result.error or "")
    assert gi_path.read_bytes() == before


def test_missing_metadata_fails_without_writing(tmp_path):
    gi_path = _make_episode(tmp_path, with_metadata=False)
    before = gi_path.read_bytes()

    result = repair_episode(gi_path, _Cfg(), build_fn=_fake_build(["x"]), write_fn=_write_json)

    assert result.ok is False
    assert "metadata unreadable" in (result.error or "")
    assert gi_path.read_bytes() == before


def test_missing_transcript_fails_without_writing(tmp_path):
    gi_path = _make_episode(tmp_path, with_transcript=False)
    before = gi_path.read_bytes()

    result = repair_episode(gi_path, _Cfg(), build_fn=_fake_build(["x"]), write_fn=_write_json)

    assert result.ok is False
    assert "transcript unreadable" in (result.error or "")
    assert gi_path.read_bytes() == before


def test_topics_are_aligned_with_the_kg_when_one_exists(tmp_path):
    """Without this the repaired episode carries bullet slugs while every other carries KG
    labels, and the CIL bridge cannot merge them by ID (#585/#653)."""
    gi_path = _make_episode(tmp_path, with_kg=True)

    result = repair_episode(gi_path, _Cfg(), build_fn=_fake_build(["Real."]), write_fn=_write_json)

    assert result.ok, result.error
    assert result.topics_aligned == 2
    doc = json.loads(gi_path.read_text(encoding="utf-8"))
    labels = sorted(n["properties"]["label"] for n in doc["nodes"] if n.get("type") == "Topic")
    assert labels == ["Cost", "Latency"], "GI topics must be the KG's canonical labels"
    assert any(e.get("type") == "ABOUT" for e in doc["edges"]), "ABOUT edges must be reconnected"


def test_no_kg_means_topics_are_left_alone(tmp_path):
    gi_path = _make_episode(tmp_path, with_kg=False)

    result = repair_episode(gi_path, _Cfg(), build_fn=_fake_build(["Real."]), write_fn=_write_json)

    assert result.ok, result.error
    assert result.topics_aligned == 0


def test_untouched_episodes_are_byte_identical(tmp_path):
    """A corpus-wide pass must not perturb episodes that were never on the work-list."""
    _make_episode(tmp_path, episode_id="ep-broken", name="0001 - Broken")
    healthy = _make_episode(tmp_path, episode_id="ep-fine", name="0002 - Fine", placeholder=False)
    before = hashlib.sha256(healthy.read_bytes()).hexdigest()

    report = repair_placeholder_artifacts(tmp_path, _Cfg(), dry_run=True)

    assert len(report.skipped_dry_run) == 1, "only the placeholder episode is on the work-list"
    assert "0001 - Broken" in report.skipped_dry_run[0]
    assert hashlib.sha256(healthy.read_bytes()).hexdigest() == before


def test_dry_run_writes_nothing(tmp_path):
    gi_path = _make_episode(tmp_path)
    before = gi_path.read_bytes()

    report = repair_placeholder_artifacts(tmp_path, _Cfg(), dry_run=True)

    assert report.ok
    assert len(report.skipped_dry_run) == 1
    assert gi_path.read_bytes() == before


def test_the_audit_trail_records_every_episode(tmp_path):
    """A corpus repair that leaves no record cannot be reviewed after the fact."""
    _make_episode(tmp_path)
    audit = tmp_path / "gi_repair_report.jsonl"

    repair_placeholder_artifacts(tmp_path, _Cfg(), audit_path=audit)

    assert audit.is_file()
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) >= {"episode_id", "gi_path", "ok", "insights_before", "insights_after"}


def test_the_report_verdict_fails_when_any_episode_failed(tmp_path):
    _make_episode(tmp_path, with_metadata=False)

    report = repair_placeholder_artifacts(tmp_path, _Cfg())

    assert report.ok is False
    assert "VERDICT: FAIL" in report.format()
    assert "placeholder left intact" in report.format()
