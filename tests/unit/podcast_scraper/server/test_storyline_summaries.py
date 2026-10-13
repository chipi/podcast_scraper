"""#2344: every storyline gets a summary; the prompt adapts to its size; orphans are dropped."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server import storyline_summaries as ss

pytestmark = pytest.mark.unit


def _corpus(root: Path, clusters: list[dict]) -> None:
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps({"data": {"clusters": clusters}}), encoding="utf-8"
    )


def _cluster(cid: str, label: str, eps: list[str]) -> dict:
    return {
        "graph_compound_parent_id": cid,
        "canonical_label": label,
        "members": [
            {"topic_id": "topic:a", "label": "a", "episode_ids": eps, "lift_to_cluster": 2.0},
            {"topic_id": "topic:b", "label": "b", "episode_ids": eps[:1], "lift_to_cluster": 1.0},
        ],
    }


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, data = tmp_path / "corpus", tmp_path / "app"
    _corpus(root, [_cluster("thc:one", "One", ["e1", "e2"]), _cluster("thc:two", "Two", ["e3"])])
    rows = {
        e: SimpleNamespace(episode_id=e, publish_date=f"2025-0{i + 1}-01")
        for i, e in enumerate(["e1", "e2", "e3"])
    }
    monkeypatch.setattr(ss, "_rows_by_episode_id", lambda root: dict(rows))
    monkeypatch.setattr(
        ss,
        "build_material",
        lambda root, s, rows, lim: (
            {"label": s.label, "episode_count": len(s.episode_ids & set(rows))}
            if s.episode_ids & set(rows)
            else None
        ),
    )
    return root, data, rows


def _sweep(root: Path, data: Path, calls: list, **kw):
    def summarize(material: dict) -> str:
        calls.append(material["label"])
        return f"About {material['label']}."

    return ss.sweep(
        root, data, summarize=summarize, price=lambda m: 0.0, model="m", prompt="p", **kw
    )


def test_every_storyline_is_summarised_and_nothing_changed_calls_nothing(world) -> None:
    root, data, _ = world
    calls: list = []
    result = _sweep(root, data, calls)
    assert sorted(result.summarised) == ["thc:one", "thc:two"] and len(calls) == 2
    assert ss.read_summaries(data)["thc:two"]["text"] == "About Two."
    again = _sweep(root, data, calls)
    assert again.summarised == [] and again.counts["unchanged"] == 2 and len(calls) == 2


def test_a_storyline_that_went_has_its_summary_dropped(world) -> None:
    root, data, _ = world
    _sweep(root, data, [])
    _corpus(root, [_cluster("thc:one", "One", ["e1", "e2"])])
    result = _sweep(root, data, [])
    assert result.counts["orphan_dropped"] == 1
    assert set(ss.read_summaries(data)) == {"thc:one"}


def test_an_episode_that_left_the_corpus_redoes_the_summary(world) -> None:
    root, data, rows = world
    calls: list = []
    _sweep(root, data, calls)
    rows.pop("e2")
    result = _sweep(root, data, calls)
    assert result.summarised == ["thc:one"]


def test_the_cap_spreads_the_first_fill_over_sweeps(world) -> None:
    root, data, _ = world
    first = _sweep(root, data, [], limits=ss.Limits(max_per_sweep=1))
    assert len(first.summarised) == 1 and first.counts["sweep_cap"] == 1
    second = _sweep(root, data, [], limits=ss.Limits(max_per_sweep=1))
    assert len(second.summarised) == 1 and set(ss.read_summaries(data)) == {"thc:one", "thc:two"}


def test_a_dry_run_calls_and_writes_nothing(world) -> None:
    root, data, _ = world
    calls: list = []
    result = _sweep(root, data, calls, dry_run=True)
    assert result.counts["would_summarise"] == 2 and calls == [] and ss.read_summaries(data) == {}


def test_the_page_reads_it_by_anchor_topic_with_stale_when_episodes_changed(world) -> None:
    root, data, rows = world
    _sweep(root, data, [])
    got = ss.summary_for(root, data, "topic:a")  # thc:one's anchor (highest lift)
    assert got is not None and got["stale"] is False
    rows.pop("e1")
    later = ss.summary_for(root, data, "thc:one")
    assert later is not None and later["stale"] is True
    assert ss.summary_for(root, data, "topic:nothing") is None


def test_spread_keeps_both_ends_and_samples_evenly() -> None:
    assert ss._spread(list(range(10)), 4) == [0, 3, 6, 9]
    assert ss._spread([1, 2], 12) == [1, 2]


def test_the_prompt_adapts_to_the_threads_size() -> None:
    """The user template says how many episodes the thread spans; the system prompt asks a
    short thread for what each episode adds and a long one also for its development."""
    from podcast_scraper.prompts.store import render_prompt

    system = render_prompt("shared/summarization/storyline_system_v1")
    assert "fewer than five episodes" in system and "how it developed over time" in system
    user = render_prompt(
        "shared/summarization/storyline_user_v1",
        label="Risk",
        topics=["risk", "systems"],
        episode_count=30,
        episodes=[{"title": "T", "date": "2025-01-01"}],
    )
    assert "spans 30 episodes, on risk, systems" in user and "spread across its time" in user
