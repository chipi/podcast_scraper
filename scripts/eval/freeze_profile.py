#!/usr/bin/env python3
"""Capture a frozen performance profile (RFC-064, Issue #510).

Runs the real pipeline with psutil sampling, then writes data/profiles/<version>.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Repo root on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psutil is required for profile freeze. Install dev deps: pip install -e '.[dev]'"
    ) from exc

from podcast_scraper import config as ps_config
from podcast_scraper.evaluation.fingerprint import generate_provider_fingerprint
from podcast_scraper.monitor.runner import MONITOR_FILE_LOG_ENV
from podcast_scraper.workflow import run_pipeline

logger = logging.getLogger(__name__)

# Default E2E RSS fixture when config uses placeholder URL (sample acceptance style)
_DEFAULT_E2E_PODCAST = "podcast1_mtb"

# Operational defaults file — merged under the provider profile so profile
# YAMLs stay provider-only. See config/profiles/freeze/_defaults.yaml.
_FREEZE_DEFAULTS_PATH = _ROOT / "config" / "profiles" / "freeze" / "_defaults.yaml"

# Default psutil poll interval (seconds); finer windows for short stages (Issue #510 / RFC-064)
_DEFAULT_SAMPLE_INTERVAL_S = 0.5

# Keys copied into *.stage_truth.json for audits (trimmed metrics.json)
_METRICS_EXCERPT_KEYS: Tuple[str, ...] = (
    "schema_version",
    "run_duration_seconds",
    "episodes_scraped_total",
    "episodes_skipped_total",
    "errors_total",
    "time_scraping",
    "time_parsing",
    "time_normalizing",
    "time_writing_storage",
    "io_and_waiting_wall_seconds",
    "io_and_waiting_thread_sum_seconds",
    "time_io_and_waiting",
    "download_media_count",
    "avg_download_media_seconds",
    "preprocessing_count",
    "avg_preprocessing_seconds",
    "transcribe_count",
    "avg_transcribe_seconds",
    "extract_names_count",
    "avg_extract_names_seconds",
    "cleaning_count",
    "avg_cleaning_seconds",
    "summarize_count",
    "avg_summarize_seconds",
    "gi_count",
    "avg_gi_seconds",
    "kg_count",
    "avg_kg_seconds",
    "vector_index_seconds",
    "download_media_time_by_episode",
    "transcribe_time_by_episode",
    "extract_names_time_by_episode",
    "summarize_time_by_episode",
    "cleaning_time_by_episode",
)

# Pipeline order for proportional sample attribution (RFC-064 stage model)
_STAGE_ORDER = [
    "rss_feed_fetch",
    "media_download",
    "audio_preprocessing",
    "transcription",
    "speaker_detection",
    "transcript_cleaning",
    "summarization",
    "gi_generation",
    "kg_extraction",
    "vector_indexing",
]


class ResourceSampler:
    """Background polling of RSS and CPU% for the current process."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # (monotonic_timestamp_s, rss_mb, cpu_percent)
        self.samples: List[Tuple[float, float, float]] = []

    def start(self) -> None:
        def _loop() -> None:
            proc = psutil.Process()

            def _sample() -> None:
                with proc.oneshot():
                    rss_mb = proc.memory_info().rss / (1024**2)
                    cpu = proc.cpu_percent(interval=None)
                self.samples.append((time.monotonic(), rss_mb, cpu))

            _sample()
            while not self._stop.wait(self._interval):
                _sample()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2.0)


def wall_seconds_by_stage(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Map saved metrics.json fields to RFC-064 stage wall times (aggregate)."""
    m = metrics
    out: Dict[str, float] = {}

    out["rss_feed_fetch"] = float(m.get("time_scraping", 0) or 0) + float(
        m.get("time_parsing", 0) or 0
    )

    dm_n = int(m.get("download_media_count", 0) or 0)
    dm_avg = float(m.get("avg_download_media_seconds", 0) or 0)
    out["media_download"] = dm_avg * dm_n

    prep_n = int(m.get("preprocessing_count", 0) or 0)
    prep_avg = float(m.get("avg_preprocessing_seconds", 0) or 0)
    out["audio_preprocessing"] = prep_avg * prep_n

    tr_n = int(m.get("transcribe_count", 0) or 0)
    tr_avg = float(m.get("avg_transcribe_seconds", 0) or 0)
    out["transcription"] = tr_avg * tr_n

    ex_n = int(m.get("extract_names_count", 0) or 0)
    ex_avg = float(m.get("avg_extract_names_seconds", 0) or 0)
    out["speaker_detection"] = ex_avg * ex_n

    cl_n = int(m.get("cleaning_count", 0) or 0)
    cl_avg = float(m.get("avg_cleaning_seconds", 0) or 0)
    out["transcript_cleaning"] = cl_avg * cl_n

    sum_n = int(m.get("summarize_count", 0) or 0)
    sum_avg = float(m.get("avg_summarize_seconds", 0) or 0)
    out["summarization"] = sum_avg * sum_n

    gi_n = int(m.get("gi_count", 0) or 0)
    gi_avg = float(m.get("avg_gi_seconds", 0) or 0)
    out["gi_generation"] = gi_avg * gi_n

    kg_n = int(m.get("kg_count", 0) or 0)
    kg_avg = float(m.get("avg_kg_seconds", 0) or 0)
    out["kg_extraction"] = kg_avg * kg_n

    out["vector_indexing"] = float(m.get("vector_index_seconds", 0) or 0)

    return {k: round(v, 4) for k, v in out.items() if v > 0.0}


def _proportional_stage_edges(
    t0: float,
    t1: float,
    stage_weights: List[Tuple[str, float]],
) -> List[Tuple[str, float, float]]:
    """Map stage names to contiguous [a,b] intervals along [t0,t1] by wall-time share."""
    if t1 <= t0 or not stage_weights:
        return []
    total_w = sum(w for _, w in stage_weights)
    if total_w <= 0:
        return []
    edges: List[Tuple[str, float, float]] = []
    cursor = t0
    for name, w in stage_weights:
        span = (w / total_w) * (t1 - t0)
        edges.append((name, cursor, cursor + span))
        cursor += span
    return edges


def _attribute_samples_to_stages(
    samples: List[Tuple[float, float, float]],
    edges: List[Tuple[str, float, float]],
) -> Dict[str, Dict[str, float]]:
    """Assign samples to proportional stage windows; compute peak RSS and mean CPU."""
    if not edges or not samples:
        return {}

    run_t0 = edges[0][1]
    run_t1 = edges[-1][2]
    stage_samples: Dict[str, List[Tuple[float, float]]] = {n: [] for n, _, _ in edges}
    for ts, rss, cpu in samples:
        if ts < run_t0 or ts > run_t1:
            continue
        for name, a, b in edges:
            if a <= ts <= b:
                stage_samples[name].append((rss, cpu))
                break

    out: Dict[str, Dict[str, float]] = {}
    for name, _, _ in edges:
        pts = stage_samples.get(name) or []
        if not pts:
            continue
        peak = max(p[0] for p in pts)
        avg_cpu = sum(p[1] for p in pts) / len(pts)
        out[name] = {
            "peak_rss_mb": round(peak, 2),
            "avg_cpu_pct": round(avg_cpu, 2),
        }
    return out


def _apply_short_stage_peak_fallback(
    res: Dict[str, Dict[str, float]],
    wall_by_stage: Dict[str, float],
    edges: List[Tuple[str, float, float]],
    samples: List[Tuple[float, float, float]],
    global_peak_mb: float,
) -> None:
    """Fill peak_rss_mb when proportional windows miss psutil ticks (e.g. vector_indexing).

    Uses max RSS in the stage window when any sample falls inside; for ``vector_indexing``
    only, falls back to the global sampled peak so YAML does not show 0 for non-zero wall.
    """
    for name, a, b in edges:
        wt = float(wall_by_stage.get(name, 0) or 0)
        if wt <= 0:
            continue
        entry = res.setdefault(name, {"peak_rss_mb": 0.0, "avg_cpu_pct": 0.0})
        peak = float(entry.get("peak_rss_mb", 0) or 0)
        if peak > 0:
            continue
        pts = [(rss, cpu) for ts, rss, cpu in samples if a <= ts <= b]
        if pts:
            entry["peak_rss_mb"] = round(max(p[0] for p in pts), 2)
            entry["avg_cpu_pct"] = round(sum(p[1] for p in pts) / len(pts), 2)
        elif name == "vector_indexing" and global_peak_mb > 0:
            entry["peak_rss_mb"] = int(round(global_peak_mb))
            if samples:
                entry["avg_cpu_pct"] = round(
                    sum(c for _, _, c in samples) / len(samples),
                    2,
                )


def _metrics_excerpt(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of metrics.json for stage_truth companion files."""
    return {k: metrics[k] for k in _METRICS_EXCERPT_KEYS if k in metrics}


def build_stage_truth_document(
    *,
    release: str,
    dataset_id: str,
    source_metrics_path: str,
    sample_interval_s: float,
    run_wall_s: float,
    wall_by_stage: Dict[str, float],
    resource_by_stage: Dict[str, Dict[str, float]],
    global_peak_rss_mb_sampled: float,
    metrics: Dict[str, Any],
    rfc065_monitor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """JSON-serializable audit bundle alongside the frozen YAML profile."""
    sum_mapped = round(sum(wall_by_stage.values()), 4)
    run_w = round(run_wall_s, 4)
    ratio = round(sum_mapped / run_w, 4) if run_w > 0 else 0.0
    doc: Dict[str, Any] = {
        "profile_stage_truth_version": 1,
        "release": release,
        "dataset_id": dataset_id,
        "source_metrics_path": source_metrics_path,
        "sample_interval_s": round(sample_interval_s, 4),
        "run_wall_s": run_w,
        "sum_mapped_stage_wall_s": sum_mapped,
        "parallelism_hint_ratio": ratio,
        "wall_seconds_by_stage": wall_by_stage,
        "resource_by_stage_psutil": resource_by_stage,
        "global_peak_rss_mb_sampled": round(global_peak_rss_mb_sampled, 2),
        "metrics_excerpt": _metrics_excerpt(metrics),
    }
    if rfc065_monitor is not None:
        doc["rfc065_monitor"] = rfc065_monitor
    return doc


def _machine_environment_block() -> Dict[str, Any]:
    """Host-level fields to merge with ProviderFingerprint (RFC-064)."""
    try:
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        ram_gb = 0.0
    phys = psutil.cpu_count(logical=False)
    logic = psutil.cpu_count(logical=True)
    return {
        "hostname": platform.node(),
        "cpu": platform.processor() or "unknown",
        "cpu_cores_physical": phys or 0,
        "cpu_cores_logical": logic or 0,
        "ram_total_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def _resolve_metrics_path(cfg: ps_config.Config, output_dir: str) -> Optional[str]:
    if cfg.metrics_output is not None:
        if not cfg.metrics_output:
            return None
        p = cfg.metrics_output
        return p if os.path.isabs(p) else os.path.abspath(p)
    return os.path.abspath(os.path.join(output_dir, "metrics.json"))


def _find_metrics_json_path(cfg: ps_config.Config, output_dir: str) -> Optional[str]:
    """Path to metrics.json from the last ``run_pipeline`` under ``output_dir``.

    The pipeline writes under ``output_dir/run_<stamp>_<suffix>/metrics.json`` when using
    run-scoped directories; the legacy layout uses ``output_dir/metrics.json`` directly.
    """
    out_abs = os.path.abspath(output_dir)
    if cfg.metrics_output is not None:
        if not cfg.metrics_output:
            return None
        p = cfg.metrics_output
        abs_p = p if os.path.isabs(p) else os.path.abspath(p)
        return abs_p if os.path.isfile(abs_p) else None

    candidates: List[str] = [os.path.join(out_abs, "metrics.json")]
    root = Path(out_abs)
    if root.is_dir():
        candidates.extend(str(p.resolve()) for p in root.rglob("metrics.json"))

    existing = [c for c in candidates if os.path.isfile(c)]
    if not existing:
        return None
    return max(existing, key=lambda p: os.stat(p).st_mtime)


def _find_monitor_log_path(cfg: ps_config.Config, output_dir: str) -> Optional[str]:
    """Newest ``.monitor.log`` under ``output_dir`` (including run-scoped subdirs)."""
    out_abs = os.path.abspath(output_dir)
    candidates: List[str] = [os.path.join(out_abs, ".monitor.log")]
    root = Path(out_abs)
    if root.is_dir():
        candidates.extend(str(p.resolve()) for p in root.rglob(".monitor.log"))
    existing = [c for c in candidates if os.path.isfile(c)]
    if not existing:
        return None
    return max(existing, key=lambda p: os.stat(p).st_mtime)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _fingerprint_device(cfg: ps_config.Config) -> Optional[str]:
    return (
        cfg.summarization_device
        or cfg.summary_device
        or cfg.transcription_device
        or cfg.whisper_device
    )


def _fingerprint_model_name(cfg: ps_config.Config) -> str:
    if getattr(cfg, "summary_model", None):
        return str(cfg.summary_model)
    if getattr(cfg, "whisper_model", None):
        return f"whisper:{cfg.whisper_model}"
    return "podcast_scraper_pipeline"


def build_profile_document(
    *,
    release: str,
    dataset_id: str,
    environment: Dict[str, Any],
    resource_by_stage: Dict[str, Dict[str, float]],
    wall_by_stage: Dict[str, float],
    episodes_processed: int,
    run_wall_s: float,
    peak_rss_mb_global: float,
) -> Dict[str, Any]:
    """Assemble the YAML-serializable profile dict."""
    stages: Dict[str, Any] = {}
    for name in _STAGE_ORDER:
        wt = wall_by_stage.get(name)
        if wt is None or wt <= 0:
            continue
        rs = resource_by_stage.get(name, {})
        stages[name] = {
            "wall_time_s": round(wt, 4),
            "peak_rss_mb": int(rs.get("peak_rss_mb", 0)),
            "avg_cpu_pct": float(rs.get("avg_cpu_pct", 0.0)),
        }

    stage_peak_max = max((d.get("peak_rss_mb", 0) for d in resource_by_stage.values()), default=0)
    totals_peak = int(max(peak_rss_mb_global, float(stage_peak_max)))

    ep = max(episodes_processed, 1)
    doc: Dict[str, Any] = {
        "release": release,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": dataset_id,
        "episodes_processed": episodes_processed,
        "environment": environment,
        "stages": stages,
        "totals": {
            "peak_rss_mb": totals_peak,
            "wall_time_s": round(run_wall_s, 4),
            "avg_wall_time_per_episode_s": round(run_wall_s / ep, 4),
            "sampling_note": (
                "Per-stage RSS/CPU uses proportional wall-time windows from metrics.json; "
                "parallel work can mis-attribute samples."
            ),
        },
    }
    return doc


def _run_measured_pipeline(
    cfg: ps_config.Config,
    *,
    sample_interval_s: float,
) -> Tuple[Dict[str, Any], float, Dict[str, Dict[str, float]], float, str]:
    """Run pipeline with sampling.

    Returns:
        metrics dict, run wall seconds, resource metrics per stage, global peak RSS (MB),
        absolute path to metrics.json used.
    """
    out_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    sampler = ResourceSampler(interval_s=sample_interval_s)
    t_wall0 = time.perf_counter()
    t_mono0 = time.monotonic()
    sampler.start()
    prev_monitor_file_log = os.environ.get(MONITOR_FILE_LOG_ENV)
    if cfg.monitor:
        os.environ[MONITOR_FILE_LOG_ENV] = "1"
    try:
        run_pipeline(cfg)
    finally:
        t_mono1 = time.monotonic()
        sampler.stop()
        if cfg.monitor:
            if prev_monitor_file_log is None:
                os.environ.pop(MONITOR_FILE_LOG_ENV, None)
            else:
                os.environ[MONITOR_FILE_LOG_ENV] = prev_monitor_file_log
    t_wall1 = time.perf_counter()
    run_wall = t_wall1 - t_wall0

    mpath = _find_metrics_json_path(cfg, out_dir)
    if not mpath:
        hint = _resolve_metrics_path(cfg, out_dir) or os.path.join(out_dir, "metrics.json")
        raise FileNotFoundError(
            f"metrics.json not found after pipeline run (looked under {out_dir!r}, "
            f"legacy path {hint!r}). Ensure metrics_output is not disabled."
        )
    metrics = json.loads(Path(mpath).read_text(encoding="utf-8"))
    wall_by_stage = wall_seconds_by_stage(metrics)

    weights = [(n, wall_by_stage[n]) for n in _STAGE_ORDER if wall_by_stage.get(n, 0) > 0]
    edges = _proportional_stage_edges(t_mono0, t_mono1, weights)
    res = _attribute_samples_to_stages(sampler.samples, edges)
    sample_peak = max((rss for _, rss, _ in sampler.samples), default=0.0)
    _apply_short_stage_peak_fallback(
        res,
        wall_by_stage,
        edges,
        sampler.samples,
        sample_peak,
    )

    return metrics, run_wall, res, sample_peak, mpath


def _rss_is_e2e_placeholder(cfg: ps_config.Config) -> bool:
    u = (cfg.rss_url or "").lower()
    return "example.invalid" in u or "e2e-placeholder" in u


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a release performance profile (RFC-064).")
    parser.add_argument("--version", required=True, help='Release tag, e.g. "v2.6.0"')
    parser.add_argument(
        "--pipeline-config",
        required=True,
        help="YAML/JSON file of podcast_scraper Config fields",
    )
    parser.add_argument(
        "--dataset-id",
        default="indicator_v1",
        help="Dataset label stored in the profile (default: indicator_v1)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output YAML path (default: data/profiles/<version>.yaml)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warm-up run (not recommended for comparable profiles)",
    )
    parser.add_argument(
        "--e2e-feed",
        default="",
        metavar="PODCAST",
        help=(
            "E2E mock RSS: start tests.e2e fixture HTTP server and set rss to "
            "feeds/PODCAST/feed.xml (e.g. podcast1_mtb). If omitted and rss in "
            "config is an example.invalid / e2e-placeholder URL, defaults to "
            f"{_DEFAULT_E2E_PODCAST}."
        ),
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=_DEFAULT_SAMPLE_INTERVAL_S,
        metavar="SEC",
        help=(
            "psutil poll interval in seconds for RSS/CPU sampling "
            f"(default: {_DEFAULT_SAMPLE_INTERVAL_S}). Lower = finer stage windows, "
            "slightly more overhead."
        ),
    )
    parser.add_argument(
        "--no-stage-truth-snapshot",
        action="store_true",
        help="Do not write <version>.stage_truth.json next to the profile YAML.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help=(
            "Enable RFC-065 live monitor for the measured run only (warm-up unchanged). "
            "Archives ticks to <version>.monitor.log beside the YAML; forces file logging "
            f"via {MONITOR_FILE_LOG_ENV} so capture works even under a TTY."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg_path = Path(args.pipeline_config)
    if not cfg_path.is_file():
        raise SystemExit(f"Pipeline config not found: {cfg_path}")

    # Merge freeze-operational defaults (rss/output_dir/max_episodes/whisper_device)
    # under the provider profile so profile YAMLs stay provider-only.
    merged: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}
    if _FREEZE_DEFAULTS_PATH.is_file():
        defaults = ps_config.load_config_file(str(_FREEZE_DEFAULTS_PATH))
        merged.update(defaults)
    merged.update(ps_config.load_config_file(str(cfg_path)))
    # Derive per-profile output_dir suffix when the operator hasn't overridden
    # the root from freeze/_defaults.yaml. Profile stem is already the provider
    # name (e.g., config/profiles/freeze/gemini.yaml -> "gemini").
    if merged.get("output_dir") == defaults.get("output_dir") and merged.get("output_dir"):
        merged["output_dir"] = str(Path(merged["output_dir"]) / cfg_path.stem)
    base_cfg = ps_config.Config.model_validate(merged)

    e2e_server = None
    e2e_podcast = (args.e2e_feed or "").strip()
    if not e2e_podcast and _rss_is_e2e_placeholder(base_cfg):
        e2e_podcast = _DEFAULT_E2E_PODCAST
    if e2e_podcast:
        try:
            from tests.e2e.fixtures.e2e_http_server import E2EHTTPServer
        except ImportError as exc:
            raise SystemExit(
                "E2E feed requested but tests.e2e.fixtures could not be imported. "
                "Run from repo root with dev deps installed."
            ) from exc
        e2e_server = E2EHTTPServer()
        e2e_server.start()
        feed_url = e2e_server.urls.feed(e2e_podcast)
        base_cfg = base_cfg.model_copy(update={"rss_url": feed_url}, deep=True)
        logger.info("E2E fixture server: %s", feed_url)

    out_default = _ROOT / "data" / "profiles" / f"{args.version}.yaml"
    out_path = Path(args.output) if args.output else out_default
    out_path.parent.mkdir(parents=True, exist_ok=True)

    monitor_for_measure = bool(base_cfg.monitor) or bool(args.monitor)
    measured_cfg = base_cfg.model_copy(
        update={"monitor": monitor_for_measure},
        deep=True,
    )

    try:
        if not args.skip_warmup:
            with tempfile.TemporaryDirectory(prefix="profile_freeze_warmup_") as tmp:
                warm = base_cfg.model_copy(
                    update={"max_episodes": 1, "output_dir": tmp, "monitor": False},
                    deep=True,
                )
                logger.info("Warm-up run (max_episodes=1) ...")
                run_pipeline(warm)

        if monitor_for_measure:
            logger.info("Measured run (RFC-065 monitor enabled) ...")
        else:
            logger.info("Measured run ...")
        metrics, run_wall, res_by_stage, peak_rss_mb, metrics_path = _run_measured_pipeline(
            measured_cfg,
            sample_interval_s=float(args.sample_interval),
        )
    finally:
        if e2e_server is not None:
            e2e_server.stop()
            logger.info("E2E fixture server stopped")

    fp = generate_provider_fingerprint(
        model_name=_fingerprint_model_name(base_cfg),
        device=_fingerprint_device(base_cfg),
        preprocessing_profile=getattr(base_cfg, "preprocessing_profile", None),
    )
    fp_dict = fp.to_dict()
    env = _machine_environment_block()
    env.update(
        {
            "package_version": fp_dict.get("package_version"),
            "git_sha": fp_dict.get("git_commit"),
            "git_dirty": fp_dict.get("git_dirty"),
            "device": fp_dict.get("device"),
            "precision": fp_dict.get("precision"),
            "library_versions": fp_dict.get("library_versions") or {},
        }
    )
    dev = (env.get("device") or "") or ""
    if str(dev).lower() == "mps":
        env["rss_measurement_note"] = (
            "Process RSS excludes much GPU memory on Apple MPS; treat as a lower bound."
        )

    wall_by_stage = wall_seconds_by_stage(metrics)
    episodes_processed = int(metrics.get("episodes_scraped_total", 0) or 0)

    doc = build_profile_document(
        release=args.version,
        dataset_id=args.dataset_id,
        environment=env,
        resource_by_stage=res_by_stage,
        wall_by_stage=wall_by_stage,
        episodes_processed=episodes_processed,
        run_wall_s=run_wall,
        peak_rss_mb_global=peak_rss_mb,
    )

    out_path.write_text(
        yaml.safe_dump(
            doc,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote profile: %s", out_path)

    rfc065_monitor_audit: Optional[Dict[str, Any]] = None
    if monitor_for_measure:
        mlog = _find_monitor_log_path(measured_cfg, measured_cfg.output_dir)
        dest_log = out_path.parent / f"{out_path.stem}.monitor.log"
        if mlog:
            shutil.copy2(mlog, dest_log)
            line_count = sum(1 for _ in dest_log.open(encoding="utf-8", errors="replace"))
            rfc065_monitor_audit = {
                "enabled": True,
                "forced_file_log_env": MONITOR_FILE_LOG_ENV,
                "source_log": os.path.abspath(mlog),
                "archived_log": _repo_relative(dest_log, _ROOT),
                "lines": line_count,
                "bytes": dest_log.stat().st_size,
            }
            logger.info("Archived monitor log: %s", dest_log)
        else:
            rfc065_monitor_audit = {
                "enabled": True,
                "forced_file_log_env": MONITOR_FILE_LOG_ENV,
                "archived_log": None,
                "note": "No .monitor.log found under output_dir after run.",
            }
            logger.warning(
                "Monitor was enabled but no .monitor.log found under %s",
                measured_cfg.output_dir,
            )

    if not args.no_stage_truth_snapshot:
        st_path = out_path.parent / f"{out_path.stem}.stage_truth.json"
        st_doc = build_stage_truth_document(
            release=args.version,
            dataset_id=args.dataset_id,
            source_metrics_path=metrics_path,
            sample_interval_s=float(args.sample_interval),
            run_wall_s=run_wall,
            wall_by_stage=wall_by_stage,
            resource_by_stage=res_by_stage,
            global_peak_rss_mb_sampled=peak_rss_mb,
            metrics=metrics,
            rfc065_monitor=rfc065_monitor_audit,
        )
        st_path.write_text(json.dumps(st_doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        logger.info("Wrote stage truth snapshot: %s", st_path)


if __name__ == "__main__":
    main()
