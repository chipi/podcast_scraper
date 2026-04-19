"""Unit tests for run_compare profile join helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_compare.data import (
    compact_baseline_select_labels,
    compact_run_display_names,
    discover_profiles,
    filter_joined_releases,
    infer_run_type_bucket,
    invert_compact_display_map,
    join_releases,
    JoinedRelease,
    load_profile,
    longest_common_prefix,
    profile_has_rfc065_trace,
    profile_stage_delta_rows,
    profile_trend_long_rows,
    ProfileEntry,
    release_key_from_run_entry,
    RUN_TYPE_BULLETS,
    RUN_TYPE_OTHER,
    RUN_TYPE_PARAGRAPH,
    RunEntry,
)


def _write_profile(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def test_load_profile_minimal(tmp_path: Path) -> None:
    yaml_text = """
release: v1-test
date: '2026-01-02T00:00:00Z'
dataset_id: ds_a
episodes_processed: 2
environment:
  hostname: host-a
stages:
  summarization:
    wall_time_s: 10.5
    peak_rss_mb: 100
    avg_cpu_pct: 5.2
totals:
  wall_time_s: 20
  peak_rss_mb: 120
  avg_wall_time_per_episode_s: 10
"""
    path = _write_profile(tmp_path, "v1-test.yaml", yaml_text)
    p = load_profile(path)
    assert p.release == "v1-test"
    assert p.dataset_id == "ds_a"
    assert p.hostname == "host-a"
    assert p.episodes_processed == 2
    assert p.stages["summarization"]["wall_time_s"] == 10.5
    assert p.totals["wall_time_s"] == 20
    assert p.monitor_log_path is None
    assert p.rfc065_monitor is None
    assert not profile_has_rfc065_trace(p)


def test_load_profile_picks_up_monitor_log_and_stage_truth(tmp_path: Path) -> None:
    yaml_text = """
release: v2-mon
date: '2026-03-01T00:00:00Z'
dataset_id: ds_x
episodes_processed: 1
environment: {hostname: h}
stages: {}
totals: {wall_time_s: 1, peak_rss_mb: 1}
"""
    path = _write_profile(tmp_path, "v2-mon.yaml", yaml_text)
    (tmp_path / "v2-mon.monitor.log").write_text("tick1\n", encoding="utf-8")
    st_doc = {
        "profile_stage_truth_version": 1,
        "rfc065_monitor": {
            "enabled": True,
            "archived_log": "data/profiles/v2-mon.monitor.log",
            "lines": 42,
            "bytes": 100,
        },
    }
    (tmp_path / "v2-mon.stage_truth.json").write_text(
        json.dumps(st_doc),
        encoding="utf-8",
    )
    p = load_profile(path)
    assert p.monitor_log_path is not None
    assert p.monitor_log_path.name == "v2-mon.monitor.log"
    assert p.monitor_trace_lines == 42
    assert p.monitor_trace_bytes == 100
    assert p.rfc065_monitor is not None
    assert p.rfc065_monitor["archived_log"] == "data/profiles/v2-mon.monitor.log"
    assert profile_has_rfc065_trace(p)


def test_discover_profiles_skips_bad_file(tmp_path: Path) -> None:
    _write_profile(
        tmp_path, "good.yaml", "release: a\ndate: '2026-01-01T00:00:00Z'\nenvironment: {}\n"
    )
    _write_profile(tmp_path, "bad.yaml", "not: [ valid yaml maybe [[[")
    profiles = discover_profiles(tmp_path)
    assert len(profiles) == 1
    assert profiles[0].release == "a"


def test_release_key_from_fingerprint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "my_run"
    run_dir.mkdir(parents=True)
    fp = {
        "run_context": {"release": "v9.9.9", "dataset_id": "x"},
    }
    (run_dir / "fingerprint.json").write_text(json.dumps(fp), encoding="utf-8")
    e = RunEntry(run_id="my_run", rel_label="runs/my_run", path=run_dir, category="run")
    assert release_key_from_run_entry(e) == "v9.9.9"


def test_release_key_fallback_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "only_id"
    run_dir.mkdir()
    e = RunEntry(run_id="only_id", rel_label="runs/only_id", path=run_dir, category="run")
    assert release_key_from_run_entry(e) == "only_id"


def test_join_releases_outer(tmp_path: Path) -> None:
    yaml_a = """
release: rel-a
date: '2026-01-01T00:00:00Z'
dataset_id: d1
environment: {hostname: h1}
stages:
  summarization:
    wall_time_s: 1
totals: {wall_time_s: 1, peak_rss_mb: 10}
"""
    pa = load_profile(_write_profile(tmp_path, "a.yaml", yaml_a))
    run_dir = tmp_path / "eval_rel_b"
    run_dir.mkdir(parents=True)
    (run_dir / "fingerprint.json").write_text(
        json.dumps({"run_context": {"release": "rel-b"}}),
        encoding="utf-8",
    )
    eb = RunEntry(run_id="eval_rel_b", rel_label="x", path=run_dir, category="run")
    joined, warns = join_releases([eb], [pa])
    assert len(joined) == 2
    rels = {j.release for j in joined}
    assert rels == {"rel-a", "rel-b"}
    assert warns == []
    a_row = next(j for j in joined if j.release == "rel-a")
    assert a_row.profile_entry is not None and a_row.eval_entry is None
    b_row = next(j for j in joined if j.release == "rel-b")
    assert b_row.eval_entry is not None and b_row.profile_entry is None


def test_filter_joined_by_hostname() -> None:
    p1 = ProfileEntry(
        release="r1",
        date="",
        dataset_id="d",
        hostname="h1",
        path=Path("/x/1.yaml"),
        stages={},
        totals={},
        environment={},
        episodes_processed=0,
        sort_ts=0.0,
    )
    p2 = ProfileEntry(
        release="r2",
        date="",
        dataset_id="d",
        hostname="h2",
        path=Path("/x/2.yaml"),
        stages={},
        totals={},
        environment={},
        episodes_processed=0,
        sort_ts=0.0,
    )
    j1 = JoinedRelease("r1", None, p1)
    j2 = JoinedRelease("r2", None, p2)
    out = filter_joined_releases([j1, j2], hostnames=["h1"])
    assert len(out) == 1 and out[0].release == "r1"


def test_profile_stage_delta_rows() -> None:
    base = ProfileEntry(
        release="b",
        date="",
        dataset_id="",
        hostname="",
        path=Path("/b.yaml"),
        stages={"summarization": {"wall_time_s": 10.0, "peak_rss_mb": 100.0, "avg_cpu_pct": 1.0}},
        totals={
            "wall_time_s": 10.0,
            "peak_rss_mb": 100.0,
            "avg_wall_time_per_episode_s": 5.0,
        },
        environment={},
        episodes_processed=1,
        sort_ts=0.0,
    )
    cand = ProfileEntry(
        release="c",
        date="",
        dataset_id="",
        hostname="",
        path=Path("/c.yaml"),
        stages={"summarization": {"wall_time_s": 12.0, "peak_rss_mb": 90.0, "avg_cpu_pct": 2.0}},
        totals={
            "wall_time_s": 12.0,
            "peak_rss_mb": 90.0,
            "avg_wall_time_per_episode_s": 6.0,
        },
        environment={},
        episodes_processed=1,
        sort_ts=0.0,
    )
    rows = profile_stage_delta_rows(base, [cand])
    wall_rows = [
        r for r in rows if r["stage"] == "summarization" and r["metric_key"] == "wall_time_s"
    ]
    assert len(wall_rows) == 1
    assert wall_rows[0]["delta"] == 2.0
    assert wall_rows[0]["good"] is False


def test_profile_trend_long_rows_orders_stages() -> None:
    p = ProfileEntry(
        release="r",
        date="",
        dataset_id="",
        hostname="",
        path=Path("/r.yaml"),
        stages={
            "vector_indexing": {"wall_time_s": 0.1, "peak_rss_mb": 1.0},
            "summarization": {"wall_time_s": 5.0, "peak_rss_mb": 50.0},
        },
        totals={},
        environment={},
        episodes_processed=0,
        sort_ts=0.0,
    )
    rows = profile_trend_long_rows([p])
    stages = [r["stage"] for r in rows]
    assert "summarization" in stages
    assert "vector_indexing" in stages
    assert stages.index("summarization") < stages.index("vector_indexing")


def test_longest_common_prefix() -> None:
    assert longest_common_prefix(["a/b/c", "a/b/d"]) == "a/b/"
    assert longest_common_prefix(["x"]) == "x"
    assert longest_common_prefix([]) == ""


def test_compact_run_display_names_shared_prefix() -> None:
    m = compact_run_display_names(["runs/x/foo_v1", "runs/x/foo_v2"])
    assert m["runs/x/foo_v1"] != m["runs/x/foo_v2"]
    # LCP is ``runs/x/foo_v`` → tails ``1`` and ``2``
    assert m["runs/x/foo_v1"].endswith("1")
    assert m["runs/x/foo_v2"].endswith("2")


def test_compact_run_display_names_single() -> None:
    m = compact_run_display_names(["only/one/path"])
    assert "only/one/path" in m
    assert len(m) == 1


def test_compact_baseline_select_labels_suffix_first() -> None:
    m = compact_baseline_select_labels(
        ["runs/prefix/foo_v1", "runs/prefix/foo_v2"],
        max_chars=40,
    )
    assert m["runs/prefix/foo_v1"] == "foo_v1"
    assert m["runs/prefix/foo_v2"] == "foo_v2"
    assert not m["runs/prefix/foo_v1"].startswith("…")


def test_compact_baseline_select_labels_parent_when_leaf_collides() -> None:
    m = compact_baseline_select_labels(["x/foo", "y/foo"], max_chars=40)
    assert m["x/foo"] == "x/foo"
    assert m["y/foo"] == "y/foo"


def test_compact_baseline_select_labels_single_fits_width() -> None:
    m = compact_baseline_select_labels(["only/one/thing"], max_chars=12)
    assert m["only/one/thing"] == "one/thing"


def test_invert_compact_display_map() -> None:
    m = compact_run_display_names(["a/x", "a/y"])
    inv = invert_compact_display_map(m)
    assert inv[m["a/x"]] == "a/x"
    assert inv[m["a/y"]] == "a/y"


def test_infer_run_type_bucket() -> None:
    assert infer_run_type_bucket("runs/x/smoke_bullets_v1") == RUN_TYPE_BULLETS
    assert infer_run_type_bucket("baselines/foo_paragraph_v2") == RUN_TYPE_PARAGRAPH
    assert infer_run_type_bucket("references/silver/silver_gpt4o_smoke_v1") == RUN_TYPE_OTHER
