"""Unit tests for freeze_profile helpers (Issue #510)."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from podcast_scraper import config as ps_config

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_FREEZE_PATH = ROOT / "scripts" / "eval" / "profile" / "freeze_profile.py"
_spec = importlib.util.spec_from_file_location("freeze_profile_under_test", _FREEZE_PATH)
assert _spec and _spec.loader
_fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fp)

wall_seconds_by_stage = _fp.wall_seconds_by_stage
_proportional_stage_edges = _fp._proportional_stage_edges
_attribute_samples_to_stages = _fp._attribute_samples_to_stages
_apply_short_stage_peak_fallback = _fp._apply_short_stage_peak_fallback
build_profile_document = _fp.build_profile_document
build_stage_truth_document = _fp.build_stage_truth_document
_rss_is_e2e_placeholder = _fp._rss_is_e2e_placeholder
_find_metrics_json_path = _fp._find_metrics_json_path
_find_monitor_log_path = _fp._find_monitor_log_path
_repo_relative = _fp._repo_relative

pytestmark = [pytest.mark.unit]


def test_wall_seconds_by_stage_aggregates() -> None:
    m = {
        "time_scraping": 1.0,
        "time_parsing": 0.5,
        "download_media_count": 2,
        "avg_download_media_seconds": 3.0,
        "preprocessing_count": 2,
        "avg_preprocessing_seconds": 1.0,
        "transcribe_count": 2,
        "avg_transcribe_seconds": 10.0,
        "extract_names_count": 2,
        "avg_extract_names_seconds": 0.5,
        "cleaning_count": 2,
        "avg_cleaning_seconds": 0.25,
        "summarize_count": 2,
        "avg_summarize_seconds": 2.0,
        "gi_count": 1,
        "avg_gi_seconds": 1.5,
        "kg_count": 1,
        "avg_kg_seconds": 2.5,
        "vector_index_seconds": 0.4,
    }
    w = wall_seconds_by_stage(m)
    assert w["rss_feed_fetch"] == pytest.approx(1.5)
    assert w["media_download"] == pytest.approx(6.0)
    assert w["audio_preprocessing"] == pytest.approx(2.0)
    assert w["transcription"] == pytest.approx(20.0)
    assert w["vector_indexing"] == pytest.approx(0.4)


def test_attribute_samples_proportional_windows() -> None:
    t0, t1 = 100.0, 110.0
    samples = [
        (100.0, 100.0, 10.0),
        (105.0, 200.0, 50.0),
        (109.0, 150.0, 20.0),
    ]
    weights = [("a", 5.0), ("b", 5.0)]
    edges = _proportional_stage_edges(t0, t1, weights)
    res = _attribute_samples_to_stages(samples, edges)
    assert "a" in res and "b" in res
    # ts=105 falls in the first window; ts=109 in the second
    assert res["a"]["peak_rss_mb"] == 200.0
    assert res["b"]["peak_rss_mb"] == 150.0


def test_vector_indexing_peak_fallback_uses_global_when_window_empty() -> None:
    """Short last stage may have no psutil tick in its proportional window."""
    t0, t1 = 0.0, 10.0
    weights = [("early", 9.0), ("vector_indexing", 1.0)]
    edges = _proportional_stage_edges(t0, t1, weights)
    # Samples only in early window; vector slice (9,10] has no points
    samples = [(1.0, 300.0, 5.0), (2.0, 400.0, 10.0)]
    res = _attribute_samples_to_stages(samples, edges)
    wall = {"early": 9.0, "vector_indexing": 1.0}
    _apply_short_stage_peak_fallback(res, wall, edges, samples, global_peak_mb=400.0)
    assert res["vector_indexing"]["peak_rss_mb"] == 400


def test_build_stage_truth_document_shape() -> None:
    doc = build_stage_truth_document(
        release="v9.9.9",
        dataset_id="ds1",
        source_metrics_path="/tmp/run/metrics.json",
        sample_interval_s=0.25,
        run_wall_s=10.0,
        wall_by_stage={"a": 8.0, "b": 4.0},
        resource_by_stage={"a": {"peak_rss_mb": 100.0, "avg_cpu_pct": 1.0}},
        global_peak_rss_mb_sampled=120.0,
        metrics={"schema_version": 2, "vector_index_seconds": 0.1},
    )
    assert doc["profile_stage_truth_version"] == 1
    assert doc["parallelism_hint_ratio"] == pytest.approx(1.2)
    assert "metrics_excerpt" in doc
    assert doc["metrics_excerpt"]["schema_version"] == 2
    assert "rfc065_monitor" not in doc


def test_build_stage_truth_document_includes_rfc065_monitor() -> None:
    doc = build_stage_truth_document(
        release="v1.0.0",
        dataset_id="ds1",
        source_metrics_path="/tmp/m.json",
        sample_interval_s=0.5,
        run_wall_s=1.0,
        wall_by_stage={"a": 1.0},
        resource_by_stage={},
        global_peak_rss_mb_sampled=10.0,
        metrics={"schema_version": 2},
        rfc065_monitor={"enabled": True, "archived_log": "data/profiles/x.monitor.log"},
    )
    assert doc["rfc065_monitor"]["enabled"] is True
    assert doc["rfc065_monitor"]["archived_log"] == "data/profiles/x.monitor.log"


def test_rss_is_e2e_placeholder_detects_sample_urls() -> None:
    assert _rss_is_e2e_placeholder(
        ps_config.Config.model_validate(
            {
                "rss": "https://example.invalid/e2e-placeholder-single-feed.xml",
                "output_dir": ".",
            },
        )
    )
    assert _rss_is_e2e_placeholder(
        ps_config.Config.model_validate(
            {"rss": "https://EXAMPLE.INVALID/x", "output_dir": "."},
        )
    )
    assert not _rss_is_e2e_placeholder(
        ps_config.Config.model_validate(
            {"rss": "https://feeds.example.com/podcast.xml", "output_dir": "."},
        )
    )


def test_find_metrics_json_path_prefers_newest_under_run_dirs(tmp_path: Path) -> None:
    old_run = tmp_path / "run_old"
    new_run = tmp_path / "run_new"
    old_run.mkdir()
    new_run.mkdir()
    (old_run / "metrics.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    time.sleep(0.02)
    (new_run / "metrics.json").write_text(json.dumps({"x": 2}), encoding="utf-8")

    cfg = ps_config.Config.model_validate(
        {"rss": "https://example.com/f.xml", "output_dir": str(tmp_path)},
    )
    found = _find_metrics_json_path(cfg, str(tmp_path))
    assert found == str((new_run / "metrics.json").resolve())


def test_find_metrics_json_path_explicit_metrics_output(tmp_path: Path) -> None:
    target = tmp_path / "custom_metrics.json"
    target.write_text("{}", encoding="utf-8")
    cfg = ps_config.Config.model_validate(
        {
            "rss": "https://example.com/f.xml",
            "output_dir": str(tmp_path),
            "metrics_output": str(target),
        },
    )
    assert _find_metrics_json_path(cfg, str(tmp_path)) == str(target.resolve())


def test_find_monitor_log_path_prefers_newest_under_tree(tmp_path: Path) -> None:
    old_run = tmp_path / "run_old"
    new_run = tmp_path / "run_new"
    old_run.mkdir()
    new_run.mkdir()
    (old_run / ".monitor.log").write_text("old\n", encoding="utf-8")
    time.sleep(0.02)
    (new_run / ".monitor.log").write_text("new\n", encoding="utf-8")

    cfg = ps_config.Config.model_validate(
        {"rss": "https://example.com/f.xml", "output_dir": str(tmp_path)},
    )
    found = _find_monitor_log_path(cfg, str(tmp_path))
    assert found == str((new_run / ".monitor.log").resolve())


def test_repo_relative_under_root(tmp_path: Path) -> None:
    sub = tmp_path / "a" / "b.txt"
    sub.parent.mkdir(parents=True)
    sub.write_text("x", encoding="utf-8")
    rel = _repo_relative(sub, tmp_path)
    assert rel.replace("\\", "/") == "a/b.txt"


def test_find_metrics_json_path_disabled_returns_none(tmp_path: Path) -> None:
    cfg = ps_config.Config.model_validate(
        {
            "rss": "https://example.com/f.xml",
            "output_dir": str(tmp_path),
            "metrics_output": "",
        },
    )
    assert _find_metrics_json_path(cfg, str(tmp_path)) is None


def test_build_profile_document_totals() -> None:
    doc = build_profile_document(
        release="v1.0.0",
        dataset_id="test_ds",
        environment={"hostname": "h1"},
        resource_by_stage={
            "rss_feed_fetch": {"peak_rss_mb": 100.0, "avg_cpu_pct": 5.0},
        },
        wall_by_stage={"rss_feed_fetch": 2.0},
        episodes_processed=2,
        run_wall_s=99.0,
        peak_rss_mb_global=250.0,
    )
    assert doc["release"] == "v1.0.0"
    assert doc["totals"]["wall_time_s"] == pytest.approx(99.0)
    assert doc["totals"]["avg_wall_time_per_episode_s"] == pytest.approx(49.5)
    assert doc["totals"]["peak_rss_mb"] == 250
    assert "rss_feed_fetch" in doc["stages"]
