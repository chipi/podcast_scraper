#!/usr/bin/env python3
"""E2E acceptance test runner.

This script runs multiple config files sequentially, collects structured data
(logs, outputs, timing, exit codes, resource usage), and saves results for analysis.

Usage:
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "config/examples/config.example.yaml" \
        --output-dir .test_outputs/acceptance \
        [--compare-baseline baseline_id] \
        [--save-as-baseline baseline_id]

Examples:
    # Run example configs
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "examples/config.example.yaml"

    # Run with baseline comparison
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "config/examples/config.example.yaml" \
        --compare-baseline baseline_v1

    # Save current run as baseline
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "config/examples/config.example.yaml" \
        --save-as-baseline baseline_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper import config
from podcast_scraper.rss.feed_cache import ENV_RSS_CACHE_DIR
from podcast_scraper.workflow.helpers import estimated_llm_cost_usd_from_metrics_dict

# Import E2E server
try:
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPServer, E2EServerURLs
except ImportError:
    E2EHTTPServer = None  # type: ignore
    E2EServerURLs = None  # type: ignore

logger = logging.getLogger(__name__)

# Exit code used when a run is killed due to --timeout (per-run timeout)
EXIT_TIMEOUT = 124
# Exit code when a run finishes within --timeout but exceeds the per-run wall budget
EXIT_PER_RUN_WALL_BUDGET = 125

# Tracked acceptance matrix file (``--from-fast-stems`` / ``--fast-only``)
MAIN_ACCEPTANCE_CONFIG_BASENAME = "MAIN_ACCEPTANCE_CONFIG.yaml"
# Baseline wall-clock comparison: flag when current >= baseline * this ratio
WALLTIME_REGRESSION_RATIO = 1.25


def _resolved_per_run_wall_seconds(cli_value: Optional[int]) -> int:
    """Per-run wall budget in seconds; ``0`` disables the post-run budget check.

    Default comes from ``ACCEPTANCE_PER_RUN_WALL_SECONDS`` (fallback ``600``) when
    the CLI flag is omitted.
    """
    if cli_value is not None:
        return max(0, int(cli_value))
    raw = os.environ.get("ACCEPTANCE_PER_RUN_WALL_SECONDS", "600").strip()
    if not raw:
        return 600
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid ACCEPTANCE_PER_RUN_WALL_SECONDS=%r; using 600", raw)
        return 600


def compute_walltime_vs_baseline_summary(
    runs: List[Dict[str, Any]],
    baseline_id: str,
    output_dir: Path,
    *,
    regression_ratio: float = WALLTIME_REGRESSION_RATIO,
) -> Optional[Dict[str, Any]]:
    """Compare wall clock per run to baseline; flag slowdowns ≥ ``regression_ratio``."""
    baseline_path = output_dir / "baselines" / baseline_id / "baseline.json"
    if not baseline_path.is_file():
        return None
    try:
        meta = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read baseline for wall-time compare: %s", exc)
        return None
    baseline_runs = meta.get("runs")
    if not isinstance(baseline_runs, list):
        return None
    by_name: Dict[str, Dict[str, Any]] = {}
    for br in baseline_runs:
        if isinstance(br, dict):
            name = br.get("config_name")
            if isinstance(name, str) and name.strip():
                by_name[name.strip()] = br
    regressions: List[Dict[str, Any]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        name = r.get("config_name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name not in by_name:
            continue
        base = by_name[name]
        cur_wall = float(r.get("wall_clock_seconds", r.get("duration_seconds", 0)) or 0.0)
        base_wall = float(base.get("wall_clock_seconds", base.get("duration_seconds", 0)) or 0.0)
        if base_wall <= 0:
            continue
        if cur_wall >= base_wall * regression_ratio:
            regressions.append(
                {
                    "config_name": name,
                    "baseline_wall_seconds": round(base_wall, 2),
                    "current_wall_seconds": round(cur_wall, 2),
                    "slowdown_ratio": round(cur_wall / base_wall, 4),
                }
            )
    return {
        "baseline_id": baseline_id,
        "baseline_path": str(baseline_path.resolve()),
        "regression_ratio_threshold": regression_ratio,
        "regressions_ge_threshold": regressions,
    }


def _apply_per_run_wall_budget_failure(run_data: Dict[str, Any], budget_seconds: int) -> None:
    """If wall clock exceeds ``budget_seconds``, fail loud and persist updated ``run_data``."""
    if budget_seconds <= 0:
        run_data.setdefault("per_run_wall_budget_exceeded", False)
        return
    wall = float(run_data.get("wall_clock_seconds", run_data.get("duration_seconds", 0)) or 0.0)
    run_data["per_run_wall_budget_seconds"] = budget_seconds
    if wall <= budget_seconds:
        run_data["per_run_wall_budget_exceeded"] = False
        return

    run_data["per_run_wall_budget_exceeded"] = True
    prev_exit = int(run_data.get("exit_code", 1))
    if prev_exit == 0:
        run_data["exit_code"] = EXIT_PER_RUN_WALL_BUDGET
        logger.error(
            "PER-RUN WALL BUDGET EXCEEDED: config=%s wall_clock=%.2fs budget=%ds "
            "(service exit was 0; failing with exit_code=%d). "
            "Increase ACCEPTANCE_PER_RUN_WALL_SECONDS / --per-run-wall-seconds, "
            "or reduce work per run.",
            run_data.get("config_name"),
            wall,
            budget_seconds,
            EXIT_PER_RUN_WALL_BUDGET,
        )
    else:
        logger.error(
            "PER-RUN WALL BUDGET EXCEEDED (run already failed): config=%s wall_clock=%.2fs "
            "budget=%ds exit_code=%d",
            run_data.get("config_name"),
            wall,
            budget_seconds,
            prev_exit,
        )
    outp = run_data.get("output_dir")
    if isinstance(outp, str) and outp.strip():
        rp = Path(outp) / "run_data.json"
        try:
            with open(rp, "w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2)
        except OSError as exc:
            logger.warning("Could not rewrite run_data.json after wall budget check: %s", exc)


def _log_session_estimated_llm_cost(runs_data: List[Dict[str, Any]]) -> None:
    """Log session-level estimated LLM API cost after all runs complete."""
    session_costs = [
        r.get("estimated_cost_usd")
        for r in runs_data
        if isinstance(r.get("estimated_cost_usd"), (int, float))
    ]
    if session_costs:
        session_total = sum(session_costs)
        logger.info("")
        logger.info(
            "Session estimated LLM API cost (USD): $%.4f  "
            "(sum of per-run estimates; see session.json / run_data.json)",
            session_total,
        )
    else:
        logger.info("")
        logger.info(
            "Session estimated LLM API cost: n/a  "
            "(no billable LLM usage in this session, or cost could not be computed)"
        )


def _estimate_llm_cost_usd_from_stdout_log(stdout_path: Path) -> Optional[float]:
    """Sum ``Total estimated cost: $X`` lines from service stdout (multi-feed fallback)."""
    if not stdout_path.is_file():
        return None
    pat = re.compile(r"Total estimated cost:\s*\$([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    total = 0.0
    found = False
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(text):
            total += float(m.group(1))
            found = True
    except OSError as exc:
        logger.debug("Could not read stdout for cost fallback %s: %s", stdout_path, exc)
        return None
    return round(total, 6) if found else None


def _estimate_llm_cost_usd_for_run_dir(run_output_dir: Path) -> Optional[float]:
    """Estimate billable LLM cost for one run (same formula as pipeline summary).

    Sums all ``metrics.json`` files under the run directory (single-feed flat layout and
    multi-feed ``feeds/.../metrics.json``). Falls back to parsing ``stdout.log`` for
    ``Total estimated cost: $...`` lines when metrics yield no billable total.

    Args:
        run_output_dir: Directory containing run artifacts.

    Returns:
        Estimated USD total, or ``None`` if missing files or no billable LLM usage.
    """
    cfg_path = run_output_dir / "config.original.yaml"
    if not cfg_path.is_file():
        return None
    try:
        cfg_dict = config.load_config_file(str(cfg_path))
        cfg_model = config.Config.model_validate(cfg_dict)
    except Exception as exc:
        logger.debug("Could not load config for cost estimate %s: %s", run_output_dir, exc)
        return None

    metrics_files = sorted(
        {p.resolve() for p in run_output_dir.rglob("metrics.json") if ".git" not in p.parts}
    )
    total = 0.0
    found_positive = False
    for metrics_path in metrics_files:
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics_dict = json.load(f)
            part = estimated_llm_cost_usd_from_metrics_dict(cfg_model, metrics_dict)
            if part is not None and part > 0:
                total += float(part)
                found_positive = True
        except Exception as exc:
            logger.debug("Skip metrics %s for cost: %s", metrics_path, exc)
            continue

    if found_positive:
        return round(total, 6)

    return _estimate_llm_cost_usd_from_stdout_log(run_output_dir / "stdout.log")


# ── Self-deriving artifact assertions (#622) ─────────────────────────
#
# Expectations derived from config flags + RSS feed metadata.
# No separate .assertions.yaml files — the config IS the spec.


def _check_metadata_speakers(run_output_dir: Path, failures: list[str]) -> None:
    """Assert at least one episode has detected_hosts in metadata."""
    metadata_files = sorted(run_output_dir.rglob("*.metadata.json"))
    if not metadata_files:
        failures.append("metadata: no .metadata.json files found")
        return
    eps_with_hosts = sum(
        1 for mf in metadata_files if _safe_json_field(mf, "content", "detected_hosts")
    )
    if eps_with_hosts == 0:
        failures.append("metadata: no episodes have detected_hosts")


def _check_gi_artifacts(run_output_dir: Path, failures: list[str]) -> list[Path]:
    """Assert GI artifacts exist with Insight nodes."""
    gi_files = sorted(run_output_dir.rglob("*.gi.json"))
    if not gi_files:
        failures.append("gi: no .gi.json files found")
        return []
    for gf in gi_files:
        try:
            data = json.loads(gf.read_text(encoding="utf-8"))
            insights = [n for n in data.get("nodes", []) if n.get("type") == "Insight"]
            if not insights:
                failures.append(f"gi: {gf.name} has no Insight nodes")
        except Exception:
            pass
    return gi_files


def _check_kg_artifacts(
    run_output_dir: Path, auto_speakers: bool, failures: list[str]
) -> list[Path]:
    """Assert KG artifacts exist; check Person entities if speakers enabled."""
    kg_files = sorted(run_output_dir.rglob("*.kg.json"))
    if not kg_files:
        failures.append("kg: no .kg.json files found")
        return []
    for kf in kg_files:
        try:
            data = json.loads(kf.read_text(encoding="utf-8"))
            topics = [n for n in data.get("nodes", []) if n.get("type") == "Topic"]
            if not topics:
                failures.append(f"kg: {kf.name} has no Topic nodes")
            if auto_speakers:
                entities = [n for n in data.get("nodes", []) if n.get("type") == "Entity"]
                persons = [
                    n for n in entities if n.get("properties", {}).get("entity_kind") == "person"
                ]
                if persons:
                    roles = {n["properties"].get("role") for n in persons}
                    if "host" not in roles:
                        failures.append(
                            f"kg: {kf.name} Person entities exist but none has role=host"
                        )
        except Exception:
            pass
    return kg_files


def _check_bridge_artifacts(
    gi_files: list[Path],
    kg_files: list[Path],
    run_output_dir: Path,
    failures: list[str],
) -> None:
    """Assert bridge.json exists when both GI and KG are present."""
    if gi_files and kg_files:
        bridge_files = sorted(run_output_dir.rglob("*.bridge.json"))
        if not bridge_files:
            failures.append("bridge: GI and KG exist but no .bridge.json found")


def _safe_json_field(path: Path, *keys: str) -> Any:
    """Read a nested field from a JSON file, returning None on any error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for k in keys:
            data = data[k]
        return data
    except Exception:
        return None


def _assert_artifacts_from_config(
    run_output_dir: Path,
    config_path: Path,
    episodes_processed: int = 0,
) -> tuple[bool, list[str]]:
    """Validate artifact content — expectations derived from config flags.

    Checks are determined by what the config enables:
    - generate_gi=true → expect .gi.json with Insight nodes
    - generate_kg=true → expect .kg.json with Topic nodes
    - auto_speakers=true → expect detected_hosts in metadata, host Person in KG
    - generate_gi + generate_kg → expect bridge.json

    No separate spec files — the config IS the spec.
    """
    try:
        cfg_dict = config.load_config_file(str(config_path))
        # Resolve ``profile:`` so flags match service behavior (materialized matrix YAMLs
        # only set ``profile`` + fragments + defaults at top level).
        cfg_model = config.Config.model_validate(cfg_dict)
    except Exception as exc:
        logger.warning("Cannot load config for assertions: %s", exc)
        return True, []

    generate_gi = bool(cfg_model.generate_gi)
    generate_kg = bool(cfg_model.generate_kg)
    auto_speakers = bool(cfg_model.auto_speakers)

    if not generate_gi and not generate_kg and not auto_speakers:
        return True, []

    failures: list[str] = []
    gi_files: list[Path] = []
    kg_files: list[Path] = []

    if auto_speakers:
        _check_metadata_speakers(run_output_dir, failures)
    if generate_gi:
        gi_files = _check_gi_artifacts(run_output_dir, failures)
    if generate_kg:
        kg_files = _check_kg_artifacts(run_output_dir, auto_speakers, failures)
    if generate_gi and generate_kg:
        _check_bridge_artifacts(gi_files, kg_files, run_output_dir, failures)

    ok = len(failures) == 0
    if not ok:
        for f in failures:
            logger.error("Artifact assertion FAILED: %s", f)
    else:
        checks = sum([auto_speakers, generate_gi, generate_kg, generate_gi and generate_kg])
        logger.info("Artifact assertions: all passed (%d checks)", checks)

    return ok, failures


def _assess_vector_index_run(
    run_output_dir: Path,
    cfg_model: config.Config,
    episodes_processed: int,
    is_dry_run: bool,
) -> tuple[bool, str]:
    """Return (ok, note) for FAISS output when vector_search is enabled."""
    if is_dry_run:
        return True, "dry_run"
    if getattr(cfg_model, "vector_search", False) is not True:
        return True, "vector_search_off"
    if getattr(cfg_model, "vector_backend", "faiss") != "faiss":
        return True, "not_faiss_backend"
    if episodes_processed <= 0:
        return True, "no_episodes"

    meta_path = run_output_dir / "search" / "metadata.json"
    if not meta_path.is_file():
        return False, "missing_search_metadata_json"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "invalid_search_metadata_json"
    if isinstance(data, dict) and len(data) > 0:
        return True, "indexed"

    id_map_path = run_output_dir / "search" / "id_map.json"
    if id_map_path.is_file():
        try:
            idm = json.loads(id_map_path.read_text(encoding="utf-8"))
            if isinstance(idm, dict) and len(idm) > 0:
                return True, "id_map_nonempty"
        except json.JSONDecodeError:
            pass
    return False, "empty_vector_index"


def _strict_vector_index_requested() -> bool:
    """True if CLI or env requests failing runs when FAISS produced no rows."""
    val = os.environ.get("STRICT_VECTOR_INDEX", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _find_config_files_one(pattern: str) -> List[Path]:
    """Resolve a single glob pattern to config paths (relative to project root)."""
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        base_dir = pattern_path.parent
        glob_pattern = pattern_path.name
    else:
        project_root = Path(__file__).parent.parent.parent
        base_dir = project_root / pattern_path.parent
        glob_pattern = pattern_path.name

    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        return []

    matches = list(base_dir.glob(glob_pattern))
    matches.sort()
    if not matches:
        logger.warning(f"No config files found matching pattern: {pattern}")
    else:
        logger.info(f"Found {len(matches)} config file(s) matching pattern: {pattern}")
    return matches


def find_config_files(pattern: str) -> List[Path]:
    """Find config files matching one or more whitespace-separated glob patterns.

    Args:
        pattern: Single glob, or multiple globs separated by whitespace (e.g.
            "config/acceptance/*.yaml")

    Returns:
        Sorted list of unique matching config file paths
    """
    parts = pattern.split()
    if not parts:
        return []
    seen_resolved: set[Path] = set()
    merged: List[Path] = []
    for pat in parts:
        for m in _find_config_files_one(pat):
            key = m.resolve()
            if key not in seen_resolved:
                seen_resolved.add(key)
                merged.append(m)
    merged.sort(key=lambda p: str(p))
    if merged:
        logger.info("Total unique config files: %d", len(merged))
    return merged


def _acceptance_config_root() -> Path:
    """Directory containing ``MAIN_ACCEPTANCE_CONFIG.yaml`` and ``fragments/``."""
    return Path(__file__).resolve().parent.parent.parent / "config" / "acceptance"


def _main_acceptance_config_path() -> Path:
    return _acceptance_config_root() / MAIN_ACCEPTANCE_CONFIG_BASENAME


def load_fast_config_matrix() -> Dict[str, Any]:
    """Load the tracked main acceptance matrix (YAML)."""
    path = _main_acceptance_config_path()
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_fast_matrix_ids() -> set[str]:
    """Matrix row ``id`` values for ``--fast-only`` / ``--from-fast-stems`` (enabled rows only)."""
    data = load_fast_config_matrix()
    runs = data.get("runs")
    if not isinstance(runs, list):
        return set()
    out: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            continue
        if run.get("enabled") is False:
            continue
        rid = run.get("id")
        if isinstance(rid, str) and rid.strip():
            out.add(rid.strip())
    return out


def load_fast_config_stems() -> set[str]:
    """Backward-compatible alias: main acceptance matrix row ids."""
    ids = load_fast_matrix_ids()
    if ids:
        logger.info(
            "Loaded %d acceptance matrix row id(s) from %s",
            len(ids),
            _main_acceptance_config_path(),
        )
    return ids


def deep_merge_acceptance_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge operator fragments; ``feeds`` / ``rss_urls`` lists replace (do not append)."""
    out = dict(base)
    for k, v in overlay.items():
        if k in ("feeds", "rss_urls") and isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_acceptance_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _feeds_fragment_path_for_run(run: Dict[str, Any]) -> Path:
    root = _acceptance_config_root()
    feeds_rel = run.get("feeds")
    if isinstance(feeds_rel, str) and feeds_rel.strip():
        return (root / feeds_rel.strip()).resolve()
    shape = run.get("feeds_shape")
    if shape == "single":
        return (root / "fragments" / "feeds_single.yaml").resolve()
    if shape == "multi":
        return (root / "fragments" / "feeds_multi.yaml").resolve()
    raise ValueError(f"acceptance matrix run missing 'feeds' or valid 'feeds_shape': {run!r}")


def _feeds_shape_tag(run: Dict[str, Any]) -> str:
    if isinstance(run.get("feeds"), str):
        return Path(run["feeds"]).stem
    return str(run.get("feeds_shape", "unknown"))


def materialize_fast_matrix_configs(session_dir: Path) -> List[Path]:
    """Write ``session_dir/materialized/{id}.yaml`` for each enabled matrix row."""
    matrix = load_fast_config_matrix()
    runs = matrix.get("runs")
    defaults_rel = matrix.get("defaults", "fragments/acceptance_defaults.yaml")
    root = _acceptance_config_root()
    defaults_path = (root / str(defaults_rel)).resolve()
    if not defaults_path.is_file():
        raise FileNotFoundError(
            f"{MAIN_ACCEPTANCE_CONFIG_BASENAME} defaults not found: {defaults_path}"
        )
    defaults_dict = config.load_config_file(str(defaults_path))
    if not isinstance(defaults_dict, dict):
        defaults_dict = {}

    mat_dir = session_dir / "materialized"
    mat_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    if not isinstance(runs, list):
        return out

    for run in runs:
        if not isinstance(run, dict):
            continue
        if run.get("enabled") is False:
            continue
        rid = run.get("id")
        prof = run.get("profile")
        if not isinstance(rid, str) or not isinstance(prof, str):
            logger.warning("Skipping invalid acceptance matrix run (missing id/profile): %s", run)
            continue
        rid = rid.strip()
        prof = prof.strip()
        feeds_path = _feeds_fragment_path_for_run(run)
        if not feeds_path.is_file():
            raise FileNotFoundError(f"Feeds fragment not found for run {rid}: {feeds_path}")
        feeds_dict = config.load_config_file(str(feeds_path))
        if not isinstance(feeds_dict, dict):
            feeds_dict = {}
        merged = deep_merge_acceptance_dicts(defaults_dict, feeds_dict)
        merged["profile"] = prof
        shape_tag = _feeds_shape_tag(run)
        merged["run_id"] = f"acceptance_{rid}_{prof}_{shape_tag}"
        dest = mat_dir / f"{rid}.yaml"
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
        out.append(dest)

    out.sort(key=lambda p: p.name)
    return out


def resolve_yaml_paths_from_stems(stems: set[str]) -> List[Path]:
    """Map each stem to an existing YAML path (acceptance dir, then examples).

    Prefer ``config/acceptance/<stem>.yaml``; ``config/examples/`` is a legacy
    fallback for stems that still ship a second copy there.

    Args:
        stems: Config file stems (no ``.yaml``).

    Returns:
        Sorted list of paths that exist. Stems with no file log a warning and are skipped.
    """
    project_root = Path(__file__).parent.parent.parent
    acceptance_dir = project_root / "config" / "acceptance"
    examples_dir = project_root / "config" / "examples"
    out: List[Path] = []
    for stem in sorted(stems):
        for base in (acceptance_dir, examples_dir):
            candidate = base / f"{stem}.yaml"
            if candidate.is_file():
                out.append(candidate)
                break
        else:
            logger.warning(
                "Fast stem %r: no %s.yaml under config/acceptance or config/examples",
                stem,
                stem,
            )
    return out


def filter_fast_configs(config_files: List[Path], fast_stems: set[str]) -> List[Path]:
    """Return only configs whose stem is in fast_stems (for --fast-only)."""
    if not fast_stems:
        return config_files
    filtered = [p for p in config_files if p.stem in fast_stems]
    return filtered


def apply_session_rss_cache_env(session_dir: Path) -> Path:
    """Create ``session_dir/rss_cache`` and set ``PODCAST_SCRAPER_RSS_CACHE_DIR``.

    Child CLI processes inherit this so sequential acceptance configs reuse feed XML
    for the same ``rss_url`` (see ``podcast_scraper.rss.feed_cache``).

    Args:
        session_dir: Timestamped session folder under the acceptance output directory.

    Returns:
        Absolute path to the ``rss_cache`` directory.
    """
    rss_cache_dir = (session_dir / "rss_cache").resolve()
    rss_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_RSS_CACHE_DIR] = str(rss_cache_dir)
    return rss_cache_dir


def _log_acceptance_session_header(
    *,
    num_configs: int,
    use_fixtures: bool,
    e2e_server: Optional[E2EHTTPServer],
    output_dir: Path,
    session_dir: Path,
    rss_cache_dir: Path,
    runs_dir: Path,
    baselines_dir: Path,
) -> None:
    """Log one batch-wide header (paths, count, mode)—not repeated per config."""
    logger.info("=" * 70)
    logger.info("Acceptance session (batch)")
    logger.info("  Configs: %d", num_configs)
    if use_fixtures and e2e_server is not None:
        logger.info("  Mode: fixtures — E2E server at %s", e2e_server.base_url)
    else:
        logger.info("  Mode: real RSS feeds and API providers")
        logger.info("  Note: set API keys and valid RSS URLs in the environment for this batch.")
    logger.info("  Output directory: %s", output_dir.absolute())
    logger.info("  Session folder: %s", session_dir)
    logger.info("  RSS feed cache (session): %s", rss_cache_dir)
    logger.info("  Run data: %s", runs_dir)
    logger.info("  Baselines: %s", baselines_dir)
    logger.info("=" * 70)
    logger.info("")


def modify_config_for_fixtures(
    config_path: Path,
    e2e_server: Optional[E2EHTTPServer],
    session_dir: Path,
    run_output_dir: Path,
    use_fixtures: bool = True,
) -> Path:
    """Modify config for running (with optional fixture support).

    Args:
        config_path: Original config file path
        e2e_server: E2E server instance (None if using real feeds/APIs)
        session_dir: Session directory (for storing modified configs if using fixtures)
        run_output_dir: Specific run output directory (already created)
        use_fixtures: If True, use E2E server fixtures; if False, use real RSS/APIs

    Returns:
        Path to modified config file
    """
    # Load original config
    config_dict = config.load_config_file(str(config_path))

    # Set output directory to the run-specific location
    config_dict["output_dir"] = str(run_output_dir)

    if use_fixtures and e2e_server:
        # Multi-feed: replace external URLs with distinct local fixture feeds (hash isolation).
        # Slot 0: podcast1_mtb → p01_mtb even when fast E2E mode is on (not p01_fast).
        # Slots 1–4: podcast2..5 → p02..p05. Samples keep five generic placeholders for copying.
        # (see tests/e2e/fixtures/e2e_http_server.py).
        fixture_feed_urls = [
            e2e_server.urls.feed("podcast1_mtb"),
            e2e_server.urls.feed("podcast2"),
            e2e_server.urls.feed("podcast3"),
            e2e_server.urls.feed("podcast4"),
            e2e_server.urls.feed("podcast5"),
        ]

        def _replace_external_feed_urls(urls: list) -> list:
            out: list = []
            for i, raw in enumerate(urls):
                u = str(raw).strip() if raw is not None else ""
                if u and not u.startswith("http://127.0.0.1"):
                    out.append(fixture_feed_urls[i % len(fixture_feed_urls)])
                else:
                    out.append(raw)
            return out

        feeds_like = config_dict.get("feeds")
        rss_urls_like = config_dict.get("rss_urls")
        rss_val = config_dict.get("rss")
        if isinstance(feeds_like, list) and feeds_like:
            config_dict["feeds"] = _replace_external_feed_urls(feeds_like)
            logger.debug("Replaced feeds with E2E fixture URLs: %s", config_dict["feeds"])
        elif isinstance(rss_urls_like, list) and rss_urls_like:
            config_dict["rss_urls"] = _replace_external_feed_urls(rss_urls_like)
            logger.debug("Replaced rss_urls with E2E fixture URLs: %s", config_dict["rss_urls"])
        elif isinstance(rss_val, list) and rss_val:
            config_dict["rss"] = _replace_external_feed_urls(rss_val)
            logger.debug("Replaced rss list with E2E fixture URLs: %s", config_dict["rss"])
        else:
            # Single-string rss (legacy / single-feed acceptance)
            original_rss = config_dict.get("rss", "")
            if (
                isinstance(original_rss, str)
                and original_rss
                and not original_rss.startswith("http://127.0.0.1")
            ):
                config_dict["rss"] = e2e_server.urls.feed("podcast1_multi_episode")
                logger.debug("Replaced RSS URL: %s -> %s", original_rss, config_dict["rss"])

        # Set API base URLs to E2E server for all providers
        # This ensures we use mock APIs, not real ones
        os.environ["OPENAI_API_BASE"] = e2e_server.urls.openai_api_base()
        os.environ["GEMINI_API_BASE"] = e2e_server.urls.gemini_api_base()
        os.environ["MISTRAL_API_BASE"] = e2e_server.urls.mistral_api_base()
        os.environ["GROK_API_BASE"] = e2e_server.urls.grok_api_base()
        os.environ["DEEPSEEK_API_BASE"] = e2e_server.urls.deepseek_api_base()
        os.environ["OLLAMA_API_BASE"] = e2e_server.urls.ollama_api_base()
        os.environ["ANTHROPIC_API_BASE"] = e2e_server.urls.anthropic_api_base()

        # Set dummy API keys (required for config validation, but won't be used with mocks)
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-bulk-tests"
        if "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "MISTRAL_API_KEY" not in os.environ:
            os.environ["MISTRAL_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "GROK_API_KEY" not in os.environ:
            os.environ["GROK_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "DEEPSEEK_API_KEY" not in os.environ:
            os.environ["DEEPSEEK_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "ANTHROPIC_API_KEY" not in os.environ:
            # Prefix must satisfy AnthropicProvider key-format check (sk-ant-…).
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-acceptance-ci-dummy-key"

    # Effective feed URLs: fixture replacement or YAML (session-wide mode is in main()).
    feed_list = config_dict.get("feeds") or config_dict.get("rss_urls")
    rss_val = config_dict.get("rss")
    if isinstance(feed_list, list) and feed_list:
        logger.info("  feeds (this config, %d urls): %s", len(feed_list), feed_list)
    elif isinstance(rss_val, list) and rss_val:
        logger.info("  rss list (this config, %d urls): %s", len(rss_val), rss_val)
    else:
        logger.info("  rss (this config): %s", config_dict.get("rss", "not set"))

    # Save modified config in the run directory
    # This is needed for the service to run and also serves as a record of what was used
    modified_config_path = run_output_dir / "config.yaml"

    with open(modified_config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    return modified_config_path


def get_config_hash(config_path: Path) -> str:
    """Get hash of config file for tracking changes.

    Args:
        config_path: Path to config file

    Returns:
        SHA256 hash of config file
    """
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def monitor_process_resources(process: subprocess.Popen, interval: float = 0.5) -> Dict[str, Any]:
    """Monitor process resource usage continuously.

    Args:
        process: Process object to monitor
        interval: Sampling interval in seconds (default: 0.5)

    Returns:
        Dict with resource usage metrics
    """
    if psutil is None:
        return {
            "peak_memory_mb": None,
            "cpu_time_seconds": None,
            "cpu_percent": None,
        }

    peak_memory_mb = 0.0
    cpu_samples = []
    cpu_times_total = None

    try:
        psutil_process = psutil.Process(process.pid)

        # Monitor continuously while process is running
        while process.poll() is None:
            try:
                # Get process memory
                mem_info = psutil_process.memory_info()
                mem_mb = mem_info.rss / (1024**2)

                # Get children memory (may fail on macOS due to permissions)
                children_mem_mb = 0.0
                try:
                    for child in psutil_process.children(recursive=True):
                        try:
                            child_mem = child.memory_info()
                            children_mem_mb += child_mem.rss / (1024**2)
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            PermissionError,
                            OSError,
                        ):
                            pass
                except (psutil.AccessDenied, PermissionError, OSError):
                    # macOS may restrict access to child processes - just use main process memory
                    # This is expected on macOS and not a critical error
                    pass

                total_mem_mb = mem_mb + children_mem_mb
                peak_memory_mb = max(peak_memory_mb, total_mem_mb)

                # Sample CPU percent
                try:
                    cpu_pct = psutil_process.cpu_percent(interval=None)
                    if cpu_pct is not None:
                        cpu_samples.append(cpu_pct)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    PermissionError,
                    OSError,
                ):
                    pass

                # Get CPU times
                try:
                    cpu_times = psutil_process.cpu_times()
                    cpu_times_total = cpu_times.user + cpu_times.system
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    PermissionError,
                    OSError,
                ):
                    pass

                time.sleep(interval)
            except psutil.NoSuchProcess:
                # Process finished
                break
            except (psutil.AccessDenied, PermissionError, OSError) as e:
                # Permission issues - log but continue
                logger.debug(f"Permission issue monitoring process: {e}")
                time.sleep(interval)

        # Get final CPU times if not already captured
        if cpu_times_total is None:
            try:
                cpu_times = psutil_process.cpu_times()
                cpu_times_total = cpu_times.user + cpu_times.system
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                PermissionError,
                OSError,
            ):
                cpu_times_total = None

        # Calculate average CPU percent
        avg_cpu_percent = statistics.mean(cpu_samples) if cpu_samples else None

        return {
            "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb > 0 else None,
            "cpu_time_seconds": round(cpu_times_total, 2) if cpu_times_total else None,
            "cpu_percent": round(avg_cpu_percent, 2) if avg_cpu_percent else None,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError) as e:
        logger.debug(f"Error monitoring process resources: {e}")
        return {
            "peak_memory_mb": None,
            "cpu_time_seconds": None,
            "cpu_percent": None,
        }


def _line_is_debug_for_console_filter(line_stripped: str) -> bool:
    """Return True if this line is clearly a DEBUG-level log line.

    Used to hide DEBUG from the console while still writing full output to stdout.log/stderr.log.
    INFO, WARNING, ERROR, CRITICAL, and lines without a recognized level are not treated as DEBUG.
    """
    if not line_stripped:
        return False
    line_lower = line_stripped.lower()
    if any(x in line_lower for x in ("levelname=debug", "level=debug")):
        return True
    # Standard library text format: "DEBUG module.name: message"
    if line_stripped.startswith("DEBUG ") or " DEBUG " in line_stripped:
        return True
    return False


def _detect_log_level(line_lower: str, line_stripped: str) -> tuple[bool | None, bool | None]:
    """Detect if line contains error or warning log level indicators.

    Args:
        line_lower: Lowercase version of the line
        line_stripped: Stripped version of the line

    Returns:
        Tuple of (is_error, is_warning) where None means no explicit log level found
    """
    # Check for structured log formats first (most explicit)
    has_error_level = any(
        level in line_lower
        for level in [
            "level=error",
            "level=critical",
            "levelname=error",
            "levelname=critical",
        ]
    )
    has_warning_level = any(
        level in line_lower
        for level in [
            "level=warning",
            "level=warn",
            "levelname=warning",
            "levelname=warn",
        ]
    )
    has_info_level = any(level in line_lower for level in ["level=info", "levelname=info"])
    has_debug_level = any(level in line_lower for level in ["level=debug", "levelname=debug"])

    # Check for uppercase log level indicators (standard Python logging format)
    has_error_uppercase = (
        " ERROR " in line_stripped
        or line_stripped.startswith("ERROR ")
        or " CRITICAL " in line_stripped
        or line_stripped.startswith("CRITICAL ")
    )
    has_warning_uppercase = (
        " WARNING " in line_stripped
        or line_stripped.startswith("WARNING ")
        or "FutureWarning:" in line_stripped
    )
    has_info_uppercase = " INFO " in line_stripped or line_stripped.startswith("INFO ")
    has_debug_uppercase = " DEBUG " in line_stripped or line_stripped.startswith("DEBUG ")

    # Classify based on log level (priority order: ERROR > WARNING > INFO/DEBUG)
    if has_error_level or has_error_uppercase:
        return (True, False)
    elif has_warning_level or has_warning_uppercase:
        return (False, True)
    elif has_info_level or has_info_uppercase or has_debug_level or has_debug_uppercase:
        # INFO/DEBUG logs are never errors or warnings
        return (False, False)
    else:
        # No explicit log level found
        return (None, None)  # type: ignore[return-value]


def _classify_line_by_patterns(
    line_lower: str,
    error_patterns: list[str],
    error_exclusions: list[str],
    warning_patterns: list[str],
    warning_exclusions: list[str],
) -> tuple[bool, bool]:
    """Classify line as error or warning using pattern matching.

    Args:
        line_lower: Lowercase version of the line
        error_patterns: Patterns that indicate errors
        error_exclusions: Patterns that exclude false positives
        warning_patterns: Patterns that indicate warnings
        warning_exclusions: Patterns that exclude false positives

    Returns:
        Tuple of (is_error, is_warning)
    """
    is_error = False
    is_warning = False

    # Check for Python traceback patterns (always errors)
    if "traceback (most recent call last)" in line_lower or "traceback:" in line_lower:
        return (True, False)

    # Check for error patterns (but exclude false positives)
    if any(pattern in line_lower for pattern in error_patterns):
        if not any(exclusion in line_lower for exclusion in error_exclusions):
            # Also exclude success indicators
            if not any(
                success_indicator in line_lower
                for success_indicator in [
                    "failed=0",
                    "failed: 0",
                    "failed= 0",
                    "ok=",
                    "ok:",
                    "result: episodes=",
                ]
            ):
                # Exclude degradation policy warnings
                if "degradation" in line_lower and "policy" in line_lower:
                    is_warning = True  # Degradation messages are warnings
                else:
                    is_error = True

    # Check for warning patterns (but exclude false positives)
    elif any(pattern in line_lower for pattern in warning_patterns):
        if not any(exclusion in line_lower for exclusion in warning_exclusions):
            # Also exclude parameter names and other non-warning contexts
            if not any(
                non_warning in line_lower
                for non_warning in [
                    "suppress_fp16_warning",
                    "warning=",
                    "warning_count",
                    "warnings total",
                    "no warning",
                    "disable_warning",
                    "ignore_warning",
                ]
            ):
                is_warning = True

    return (is_error, is_warning)


def _normalize_log_line(line_stripped: str) -> str:
    """Normalize log line for deduplication.

    Args:
        line_stripped: Stripped log line

    Returns:
        Normalized line (timestamp, log level, and logger name removed)
    """
    normalized = line_stripped
    # Try to remove timestamp prefix (format: "YYYY-MM-DD HH:MM:SS,mmm")
    normalized = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} ", "", normalized)
    # Remove log level prefix if present
    normalized = re.sub(
        r"^(ERROR|CRITICAL|WARNING|INFO|DEBUG)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    # Remove logger name prefix (format: "logger_name: ")
    normalized = re.sub(r"^[a-zA-Z0-9_.]+\s*:\s*", "", normalized)
    return normalized.strip()


def collect_logs_from_output(output_dir: Path) -> Dict[str, Any]:
    """Collect log information from output directory.

    Args:
        output_dir: Output directory for the run

    Returns:
        Dict with log analysis: ``errors`` / ``warnings`` (distinct, capped), ``*_lines_total``
        (raw line counts), ``*_count_distinct``, and ``info_count``.
    """
    errors = []
    warnings = []
    info_count = 0
    error_lines_total = 0
    warning_lines_total = 0

    # Track seen error/warning lines to avoid duplicates (same error in stdout and stderr)
    seen_errors = set()
    seen_warnings = set()

    # Patterns that indicate actual errors (not just mentions of the word "error")
    error_patterns = [
        "traceback",
        "exception:",
        "error:",
        "failed",
        "failure",
        "critical",
        "fatal",
    ]

    # Patterns that indicate the word "error" is used in a non-error context
    error_exclusions = [
        "error_type: none",
        "error_message: none",
        "errors total: 0",
        "no error",
        "error_count: 0",
        "error: none",
        "error: null",
        "'error_type': none",
        "'error_message': none",
        '"error_type": null',
        '"error_message": null',
        "failed=0",  # Processing job with no failures
        "failed: 0",  # Processing job with no failures
        "failed= 0",  # Processing job with no failures
        "ok=",  # Success indicators (e.g., "ok=3, failed=0")
        "ok:",
        "result: episodes=",  # Result summary lines
        "degradation policy",  # Degradation warnings
        # (e.g., "Saving transcript without summary (degradation policy: ...)")
        "degradation:",  # Degradation logger messages
        "failures: 0",  # Metrics lines (e.g. "- Gi Failures: 0")
        "failure: 0",
        "gi failures: 0",
    ]

    # Patterns that indicate actual warnings
    warning_patterns = [
        "warning:",
        "warn:",
        "deprecated",
        "deprecation",
    ]

    # Patterns that indicate the word "warning" is used in a non-warning context
    warning_exclusions = [
        "warning_count: 0",
        "warnings total: 0",
        "no warning",
        "suppress_fp16_warning",  # Parameter name
        "warning=",  # Parameter assignment
        "disable_warning",
        "ignore_warning",
    ]

    # Look for log files (dedupe: root glob is a subset of rglob)
    log_files = sorted({p.resolve() for p in output_dir.rglob("*.log")})

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    line_lower = line_stripped.lower()

                    # PRIORITY 1: Check log level first (most reliable indicator)
                    # This must be done before pattern matching to avoid misclassification
                    level_is_error, level_is_warning = _detect_log_level(line_lower, line_stripped)

                    if level_is_error is not None:
                        # Explicit log level found
                        is_error = level_is_error
                        is_warning = level_is_warning
                    else:
                        # No explicit log level found - use pattern matching as fallback
                        is_error, is_warning = _classify_line_by_patterns(
                            line_lower,
                            error_patterns,
                            error_exclusions,
                            warning_patterns,
                            warning_exclusions,
                        )

                    # Add to appropriate list (errors take priority if somehow both are True)
                    # Deduplicate by normalizing the line (remove timestamps, etc.) to avoid
                    # counting the same error from both stdout.log and stderr.log
                    if is_error:
                        error_lines_total += 1
                        normalized = _normalize_log_line(line_stripped)
                        # Only add if we haven't seen this exact error before
                        if normalized and normalized not in seen_errors:
                            seen_errors.add(normalized)
                            errors.append(line_stripped[:200])  # Limit length
                        continue
                    elif is_warning:
                        warning_lines_total += 1
                        normalized = _normalize_log_line(line_stripped)
                        # Only add if we haven't seen this exact warning before
                        if normalized and normalized not in seen_warnings:
                            seen_warnings.add(normalized)
                            warnings.append(line_stripped[:200])
                        continue

                    # Count info messages
                    if "info" in line_lower or any(
                        level in line_lower for level in ["level=info", "levelname=info"]
                    ):
                        info_count += 1
        except Exception as e:
            logger.warning(f"Failed to read log file {log_file}: {e}")

    # Extract key error messages for known error types (e.g., Ollama not running)
    # This helps identify the root cause more quickly
    key_error_messages = []
    for error in errors:
        error_lower = error.lower()
        # Check for common, actionable error messages
        if "ollama server is not running" in error_lower:
            # Extract the full helpful message
            if "ollama serve" in error_lower or "ollama.ai" in error_lower:
                key_error_messages.append(
                    "Ollama server is not running - start with 'ollama serve'"
                )
        elif "connection refused" in error_lower and "ollama" in error_lower:
            key_error_messages.append("Ollama connection refused - server may not be running")
        elif "model" in error_lower and "not available" in error_lower and "ollama" in error_lower:
            key_error_messages.append(
                "Ollama model not available - may need to run 'ollama pull <model>'"
            )
        elif "api key" in error_lower and ("missing" in error_lower or "invalid" in error_lower):
            key_error_messages.append("API key missing or invalid - check environment variables")
        elif "rate limit" in error_lower:
            key_error_messages.append("Rate limit exceeded - consider reducing request frequency")

    # Remove duplicates from key_error_messages
    key_error_messages = list(dict.fromkeys(key_error_messages))  # Preserves order

    return {
        "errors": errors[:50],  # Limit to 50 errors (distinct normalized messages)
        "warnings": warnings[:50],  # Limit to 50 warnings (distinct)
        "error_count_distinct": len(errors),
        "warning_count_distinct": len(warnings),
        "error_lines_total": error_lines_total,
        "warning_lines_total": warning_lines_total,
        "info_count": info_count,
        "key_error_messages": key_error_messages[:10],  # Top 10 key error messages
    }


def copy_service_outputs(service_output_dir: Path, run_output_dir: Path) -> None:
    """Copy service output files (metadata, transcripts, summaries) to run directory.

    The service creates output in a structure like:
    - service_output_dir/run_<suffix>/transcripts/
    - service_output_dir/run_<suffix>/metadata/
    - service_output_dir/run_<suffix>/*.json (run.json, index.json, etc.)

    This function copies all these files directly to run_output_dir (flattened structure).
    Metadata and transcripts folders are kept, but run.json, index.json, etc. are placed flat.

    Args:
        service_output_dir: The output directory configured in the service
            (where service wrote files)
        run_output_dir: The run directory where we want to copy the files
    """
    if not service_output_dir.exists():
        logger.warning(f"Service output directory does not exist: {service_output_dir}")
        return

    # Find the actual run subdirectory (service creates run_<suffix> subdirs)
    run_subdirs = [
        d for d in service_output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_subdirs:
        # Service might have written directly to service_output_dir
        source_dir = service_output_dir
    else:
        # Use the most recent run subdirectory (in case there are multiple)
        source_dir = max(run_subdirs, key=lambda p: p.stat().st_mtime)

    logger.debug(f"Copying service outputs from: {source_dir} to {run_output_dir}")

    def _copytree_if_distinct(src: Path, dst: Path, label: str) -> None:
        """Copy directory unless src and dst are the same path (avoid rmtree deleting source)."""
        if not src.exists():
            return
        try:
            if src.resolve() == dst.resolve():
                logger.debug("Skip %s copy: already at run root (%s)", label, dst)
                return
        except OSError:
            pass
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.debug("Copied %s to: %s", label, dst)

    # Copy metadata files directly to run_output_dir/metadata/ (keep folder structure)
    metadata_source = source_dir / "metadata"
    metadata_dest = run_output_dir / "metadata"
    _copytree_if_distinct(metadata_source, metadata_dest, "metadata files")

    # Copy transcript files directly to run_output_dir/transcripts/ (keep folder structure)
    transcripts_source = source_dir / "transcripts"
    transcripts_dest = run_output_dir / "transcripts"
    _copytree_if_distinct(transcripts_source, transcripts_dest, "transcript files")

    # Copy search/FAISS index directory (vector_search output)
    search_source = source_dir / "search"
    search_dest = run_output_dir / "search"
    _copytree_if_distinct(search_source, search_dest, "search index")

    # Copy run tracking files (run.json, index.json, run_manifest.json, metrics.json)
    # directly to run_output_dir (flat)
    for tracking_file in ["run.json", "index.json", "run_manifest.json", "metrics.json"]:
        source_file = source_dir / tracking_file
        if source_file.exists():
            dest_file = run_output_dir / tracking_file
            shutil.copy2(source_file, dest_file)
            logger.debug(f"Copied {tracking_file} to: {dest_file}")

    # Copy any markdown summary files (if they exist) directly to run_output_dir (flat)
    for md_file in source_dir.rglob("*.md"):
        # Skip if it's in a subdirectory we already copied
        if "metadata" in md_file.parts or "transcripts" in md_file.parts:
            continue
        # Copy directly to run_output_dir (flat structure)
        dest_file = run_output_dir / md_file.name
        shutil.copy2(md_file, dest_file)
        logger.debug(f"Copied markdown file {md_file.name} to: {dest_file}")

    # Clean up the original nested service output directory after copying
    # This prevents confusion and saves disk space (files are now in flattened structure)
    # Only remove if source_dir is different from run_output_dir (i.e., it's a nested subdirectory)
    if source_dir != run_output_dir and source_dir.exists():
        try:
            # Check if source_dir is actually inside run_output_dir (safety check)
            if run_output_dir in source_dir.parents or source_dir.parent == run_output_dir:
                logger.debug(f"Removing original nested service output directory: {source_dir}")
                shutil.rmtree(source_dir)
                logger.debug(f"Cleaned up nested directory: {source_dir}")
        except Exception as exc:
            # Don't fail if cleanup fails - files are already copied
            logger.warning(f"Failed to clean up nested directory {source_dir}: {exc}")


def collect_outputs(output_dir: Path) -> Dict[str, Any]:
    """Collect output file information.

    For multi-feed runs the files live under ``feeds/rss_…/run_…/``.
    We use **full path** dedup (not filename-only) so identically-named
    files in different feed directories are counted separately.

    Args:
        output_dir: Output directory for the run

    Returns:
        Dict with output counts (transcripts, metadata, summaries)
    """
    search_dirs = [output_dir]

    transcripts = 0
    metadata = 0
    summaries = 0

    # Dedup by *full resolved path* so multi-feed files are not collapsed.
    unique_txt_paths: set[Path] = set()
    unique_srt_paths: set[Path] = set()
    unique_metadata: dict[Path, Path] = {}  # resolved → original

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        txt_files = [p for p in search_dir.rglob("*.txt") if not p.name.endswith(".cleaned.txt")]
        srt_files = list(search_dir.rglob("*.srt"))

        for txt_file in txt_files:
            unique_txt_paths.add(txt_file.resolve())
        for srt_file in srt_files:
            unique_srt_paths.add(srt_file.resolve())

        metadata_files = [
            p
            for p in search_dir.rglob("*.json")
            if "metadata" in p.name
            or (p.parent.name == "metadata" and p.suffix in [".json", ".yaml"])
        ] + [
            p
            for p in search_dir.rglob("*.yaml")
            if "metadata" in p.name
            or (p.parent.name == "metadata" and p.suffix in [".json", ".yaml"])
        ]

        for mf in metadata_files:
            resolved = mf.resolve()
            if resolved not in unique_metadata:
                unique_metadata[resolved] = mf

    transcripts = len(unique_txt_paths) + len(unique_srt_paths)
    metadata = len(unique_metadata)

    for _resolved, metadata_file in unique_metadata.items():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                if metadata_file.suffix == ".yaml":
                    import yaml

                    content = yaml.safe_load(f)
                else:
                    content = json.load(f)
                if content and isinstance(content, dict):
                    if content.get("summary") or (
                        content.get("content")
                        and isinstance(content.get("content"), dict)
                        and content["content"].get("summary")
                    ):
                        summaries += 1
        except Exception:
            pass

    # Also count separate markdown summary files (if any exist)
    # Track these separately to avoid double-counting
    summary_md_files = set()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for md_file in search_dir.rglob("*.md"):
            if "summary" in md_file.name.lower() or "summary" in str(md_file):
                # Use normalized name to avoid double-counting
                summary_md_files.add(md_file.name)
    summaries += len(summary_md_files)

    return {
        "transcripts": transcripts,
        "metadata": metadata,
        "summaries": summaries,
    }


def _detect_dry_run_from_config(config_files: list[Path]) -> bool:
    """Detect if dry-run mode is enabled in any of the config files.

    Args:
        config_files: List of config file paths to check

    Returns:
        True if dry-run mode is detected, False otherwise
    """
    for config_file in config_files:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                import yaml

                config_content = yaml.safe_load(f)
                # Check for dry_run flag (can be True, "true", 1, etc.)
                dry_run_value = config_content.get("dry_run") if config_content else None
                if dry_run_value is True or (
                    isinstance(dry_run_value, str) and dry_run_value.lower() in ["true", "1", "yes"]
                ):
                    logger.debug(f"Detected dry-run mode from config: {config_file}")
                    return True
        except Exception as e:
            logger.debug(f"Failed to read config for dry-run detection: {config_file}: {e}")
    return False


def _detect_dry_run_from_stdout(stdout_path: Path) -> bool:
    """Detect if dry-run mode is indicated in stdout.log.

    Args:
        stdout_path: Path to stdout.log file

    Returns:
        True if dry-run mode is detected, False otherwise
    """
    if not stdout_path.exists():
        return False

    # Try reading multiple times with small delay to handle race conditions
    for attempt in range(3):
        try:
            with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
                stdout_content = f.read()
                # Check for various dry-run indicators
                stdout_lower = stdout_content.lower()
                if any(
                    indicator in stdout_lower
                    for indicator in [
                        "dry run complete",
                        "dry-run complete",
                        "(dry-run)",
                        "(dry run)",
                        "dry run mode",
                        "dry-run mode",
                    ]
                ):
                    logger.debug(f"Detected dry-run mode from stdout.log (attempt {attempt + 1})")
                    return True
                # If we got here, file was readable but no dry-run indicator found
                return False
        except (IOError, OSError) as e:
            # File might still be open/writing, wait a bit and retry
            if attempt < 2:
                time.sleep(0.1)  # 100ms delay
                continue
            logger.debug(f"Failed to read stdout.log after {attempt + 1} attempts: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error reading stdout.log: {e}")
            return False
    return False


def _extract_episodes_from_dry_run(stdout_content: str) -> int:
    """Extract planned episode count from dry-run output.

    Args:
        stdout_content: Content of stdout.log

    Returns:
        Number of planned episodes, or 0 if not found
    """
    # Try multiple patterns to be robust
    patterns = [
        r"transcripts_planned[=:]?\s*(\d+)",  # transcripts_planned=3 or transcripts_planned: 3
        r"transcripts.*planned[=:]?\s*(\d+)",  # transcripts planned=3 (with space)
        r"planned.*transcripts[=:]?\s*(\d+)",  # planned transcripts=3
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout_content, re.IGNORECASE)
        if match:
            try:
                count = int(match.group(1))
                logger.debug(
                    f"Extracted planned transcripts from dry-run output: "
                    f"{count} (pattern: {pattern})"
                )
                return count
            except (ValueError, IndexError):
                continue
    logger.debug("Could not extract transcripts_planned from dry-run output, defaulting to 0")
    return 0


def _extract_episodes_from_corpus_summary(output_dir: Path) -> int:
    """Sum episodes_processed across feeds from corpus_run_summary.json.

    Multi-feed runs write this file at the corpus root with per-feed
    episode counts.  Using it avoids the fallback heuristics that
    under-count when ``run.json`` only exists inside each feed subdir.

    Returns:
        Total episodes across all feeds, or 0 if file missing/invalid.
    """
    summary_path = output_dir / "corpus_run_summary.json"
    if not summary_path.exists():
        return 0
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        feeds = data.get("feeds", [])
        if not isinstance(feeds, list):
            return 0
        total = sum(int(fd.get("episodes_processed", 0)) for fd in feeds if isinstance(fd, dict))
        if total > 0:
            logger.debug(
                "Episode count from corpus_run_summary.json: %d " "(%d feeds)",
                total,
                len(feeds),
            )
        return total
    except Exception as exc:
        logger.debug("Failed to read corpus_run_summary.json: %s", exc)
        return 0


def _extract_episodes_from_run_json(run_json_path: Path) -> int:
    """Extract episode count from run.json.

    Args:
        run_json_path: Path to run.json file

    Returns:
        Number of episodes processed, or 0 if not found
    """
    if not run_json_path.exists():
        return 0

    try:
        with open(run_json_path, "r") as f:
            run_data = json.load(f)
            # Try different possible field names
            return (
                run_data.get("episodes_scraped_total")
                or run_data.get("episodes_processed")
                or run_data.get("episodes")
                or run_data.get("total_episodes")
                or 0
            )
    except Exception as e:
        logger.debug(f"Failed to read episode count from run.json: {e}")
        return 0


def _extract_provider_info(config_path: Path) -> Dict[str, Any]:
    """Extract provider and model information from config file.

    Args:
        config_path: Path to config file

    Returns:
        Dict with provider/model information
    """
    try:
        config_dict = config.load_config_file(str(config_path))
    except Exception as e:
        logger.debug(f"Failed to load config for provider extraction: {e}")
        return {}

    provider_info: Dict[str, Any] = {}

    # Transcription provider
    transcription_provider = config_dict.get("transcription_provider", "whisper")
    provider_info["transcription_provider"] = transcription_provider
    if transcription_provider == "whisper":
        # Match Config default (config.py) so report reflects what the service used
        provider_info["transcription_model"] = config_dict.get("whisper_model", "base.en")
    elif transcription_provider == "openai":
        provider_info["transcription_model"] = config_dict.get(
            "openai_transcription_model", "whisper-1"
        )
    elif transcription_provider == "gemini":
        provider_info["transcription_model"] = config_dict.get(
            "gemini_transcription_model", "gemini-2.5-flash-lite"
        )
    elif transcription_provider == "mistral":
        provider_info["transcription_model"] = config_dict.get(
            "mistral_transcription_model", "mistral-large-latest"
        )

    # Speaker detection provider
    speaker_provider = config_dict.get("speaker_detector_provider", "spacy")
    provider_info["speaker_provider"] = speaker_provider
    if speaker_provider == "spacy":
        provider_info["speaker_model"] = config_dict.get("ner_model", "en_core_web_sm")
    elif speaker_provider == "openai":
        provider_info["speaker_model"] = config_dict.get("openai_speaker_model", "gpt-4o-mini")
    elif speaker_provider == "gemini":
        provider_info["speaker_model"] = config_dict.get(
            "gemini_speaker_model", "gemini-2.5-flash-lite"
        )
    elif speaker_provider == "anthropic":
        provider_info["speaker_model"] = config_dict.get(
            "anthropic_speaker_model", "claude-haiku-4-5"
        )
    elif speaker_provider == "mistral":
        provider_info["speaker_model"] = config_dict.get(
            "mistral_speaker_model", "mistral-large-latest"
        )
    elif speaker_provider == "grok":
        provider_info["speaker_model"] = config_dict.get("grok_speaker_model", "grok-beta")
    elif speaker_provider == "deepseek":
        provider_info["speaker_model"] = config_dict.get("deepseek_speaker_model", "deepseek-chat")
    elif speaker_provider == "ollama":
        provider_info["speaker_model"] = config_dict.get("ollama_speaker_model", "llama3.1:8b")

    # Summarization provider
    summary_provider = config_dict.get("summary_provider", "transformers")
    provider_info["summary_provider"] = summary_provider
    if summary_provider in ("transformers", "local"):
        provider_info["summary_map_model"] = config_dict.get(
            "summary_model", "facebook/bart-large-cnn"
        )
        provider_info["summary_reduce_model"] = config_dict.get("summary_reduce_model")
    elif summary_provider == "openai":
        provider_info["summary_model"] = config_dict.get("openai_summary_model", "gpt-4o-mini")
    elif summary_provider == "gemini":
        provider_info["summary_model"] = config_dict.get(
            "gemini_summary_model", "gemini-2.5-flash-lite"
        )
    elif summary_provider == "anthropic":
        provider_info["summary_model"] = config_dict.get(
            "anthropic_summary_model", "claude-haiku-4-5"
        )
    elif summary_provider == "mistral":
        provider_info["summary_model"] = config_dict.get(
            "mistral_summary_model", "mistral-large-latest"
        )
    elif summary_provider == "grok":
        provider_info["summary_model"] = config_dict.get("grok_summary_model", "grok-beta")
    elif summary_provider == "deepseek":
        provider_info["summary_model"] = config_dict.get("deepseek_summary_model", "deepseek-chat")
    elif summary_provider == "ollama":
        provider_info["summary_model"] = config_dict.get("ollama_summary_model", "llama3.1:8b")
    elif summary_provider == "hybrid_ml":
        provider_info["summary_map_model"] = config_dict.get("hybrid_map_model", "longt5-base")
        provider_info["summary_reduce_model"] = config_dict.get(
            "hybrid_reduce_model", "google/flan-t5-base"
        )
        provider_info["summary_reduce_backend"] = config_dict.get(
            "hybrid_reduce_backend", "transformers"
        )

    return provider_info


def _extract_episodes_from_index_json(index_json_path: Path) -> int:
    """Extract episode count from index.json.

    Args:
        index_json_path: Path to index.json file

    Returns:
        Number of episodes processed, or 0 if not found
    """
    if not index_json_path.exists():
        return 0

    try:
        with open(index_json_path, "r") as f:
            index_data = json.load(f)
            episodes = index_data.get("episodes", [])
            if isinstance(episodes, list):
                return len(episodes)
            elif isinstance(episodes, dict):
                return len(episodes.get("items", []))
    except Exception as e:
        logger.debug(f"Failed to read episode count from index.json: {e}")
    return 0


def _execute_process_with_streaming(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    config_name: str,
    timeout_seconds: Optional[int] = None,
    hide_debug_console: bool = True,
) -> tuple[int, Dict[str, Any]]:
    """Execute process with real-time log streaming.

    Args:
        cmd: Command to execute
        stdout_path: Path to save stdout
        stderr_path: Path to save stderr
        config_name: Config name for log prefix
        timeout_seconds: If set, kill process after this many seconds (per-run timeout).
        hide_debug_console: If True, omit DEBUG lines on the console (files unchanged).

    Returns:
        Tuple of (exit_code, resource_usage_dict). exit_code is EXIT_TIMEOUT if timed out.
    """
    stdout_file = open(stdout_path, "w", buffering=1)  # Line buffered
    stderr_file = open(stderr_path, "w", buffering=1)  # Line buffered

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        text=True,
        bufsize=1,  # Line buffered
    )

    # Stream output in real-time
    def stream_output(pipe, file, prefix="", hide_debug: bool = False):
        """Stream output from pipe to both file and console."""
        for line in iter(pipe.readline, ""):
            if not line:
                break
            # Write to file (always full fidelity)
            file.write(line)
            file.flush()
            stripped = line.rstrip()
            if hide_debug and _line_is_debug_for_console_filter(stripped):
                continue
            # Print to console with prefix
            if prefix:
                print(f"{prefix}{stripped}", flush=True)
            else:
                print(stripped, flush=True)

    # Start threads to stream stdout and stderr
    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, stdout_file, f"[{config_name}] "),
        kwargs={"hide_debug": hide_debug_console},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, stderr_file, f"[{config_name}] "),
        kwargs={"hide_debug": hide_debug_console},
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    # Monitor resources continuously in background thread
    resource_result = {"resource_usage": None}

    def monitor_resources():
        resource_result["resource_usage"] = monitor_process_resources(process)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    # Wait for completion (with optional timeout)
    try:
        exit_code = process.wait(timeout=timeout_seconds if timeout_seconds else None)
    except subprocess.TimeoutExpired:
        logger.warning(f"[{config_name}] Run timed out after {timeout_seconds}s, killing process")
        process.kill()
        process.wait(timeout=5)
        exit_code = EXIT_TIMEOUT
    except Exception:
        process.kill()
        raise

    # Wait for monitoring to finish
    monitor_thread.join(timeout=2.0)
    resource_usage = resource_result["resource_usage"] or {
        "peak_memory_mb": None,
        "cpu_time_seconds": None,
        "cpu_percent": None,
    }

    # Wait for output threads to finish
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)

    # Close files
    stdout_file.close()
    stderr_file.close()

    return exit_code, resource_usage


def _execute_process_without_streaming(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: Optional[int] = None,
) -> tuple[int, Dict[str, Any]]:
    """Execute process without log streaming (only save to files).

    Args:
        cmd: Command to execute
        stdout_path: Path to save stdout
        stderr_path: Path to save stderr
        timeout_seconds: If set, kill process after this many seconds (per-run timeout).

    Returns:
        Tuple of (exit_code, resource_usage_dict). exit_code is EXIT_TIMEOUT if timed out.
    """
    with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=os.environ.copy(),
        )

        # Monitor resources continuously
        resource_usage = monitor_process_resources(process)

        # Wait for completion (with optional timeout)
        try:
            exit_code = process.wait(timeout=timeout_seconds if timeout_seconds else None)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            exit_code = EXIT_TIMEOUT
        except Exception:
            process.kill()
            raise

    return exit_code, resource_usage


def run_config(
    config_path: Path,
    e2e_server: Optional[E2EHTTPServer],
    output_dir: Path,
    run_id: str,
    use_fixtures: bool = True,
    show_logs: bool = True,
    timeout_seconds: Optional[int] = None,
    hide_debug_console: bool = True,
    run_index: Optional[int] = None,
    run_total: Optional[int] = None,
    *,
    strict_vector_index: bool = False,
    assert_artifacts: bool = False,
) -> Dict[str, Any]:
    """Run a single config and collect data.

    Args:
        config_path: Path to config file
        e2e_server: E2E server instance (None if using real feeds/APIs)
        output_dir: Base output directory
        run_id: Unique run identifier
        use_fixtures: If True, use E2E server fixtures; if False, use real RSS/APIs
        show_logs: If True, stream logs to console in real-time; if False, only save to files
        timeout_seconds: If set, kill the run after this many seconds (per-run timeout).
        hide_debug_console: When streaming, omit DEBUG lines from the console (files unchanged).
        run_index: 1-based index when running multiple configs (optional).
        run_total: Total configs in session when running multiple (optional).
        strict_vector_index: When True, set ``exit_code`` to 1 if ``vector_search`` is enabled
            but ``search/metadata.json`` has no vector rows after episodes ran.

    Returns:
        Dict with run data
    """
    config_name = config_path.stem
    # When main runs multiple configs, it prints one run banner per iteration; avoid duplicate.
    if run_index is None or run_total is None or run_total <= 1:
        if run_index is not None and run_total is not None:
            logger.info("Running config %d/%d: %s", run_index, run_total, config_name)
        else:
            logger.info("Running config: %s", config_name)

    # Create timestamped run directory (not based on config name since feeds may differ)
    # output_dir is runs_dir (session_dir / "runs"); use resolved path so run dir is absolute
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # Include milliseconds for uniqueness
    run_dir_name = f"run_{run_timestamp}"
    run_output_dir = (output_dir / run_dir_name).resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original config to run folder for reference
    original_config_copy = run_output_dir / "config.original.yaml"
    shutil.copy2(config_path, original_config_copy)

    # Modify config (with optional fixture support)
    # Pass session_dir and run_output_dir so it can set the output_dir in the config
    # Need to get session_dir from output_dir (which is now runs_dir)
    session_dir = output_dir.parent  # runs_dir.parent is session_dir
    modified_config = modify_config_for_fixtures(
        config_path, e2e_server, session_dir, run_output_dir, use_fixtures=use_fixtures
    )

    # Get config hash (from original config)
    config_hash = get_config_hash(config_path)

    # Run the service
    start_time = time.time()
    start_timestamp = datetime.utcnow().isoformat() + "Z"

    # Build command
    python_cmd = sys.executable
    cmd = [
        python_cmd,
        "-m",
        "podcast_scraper.service",
        "--config",
        str(modified_config),
    ]

    # Capture stdout/stderr to files (and optionally stream to console)
    stdout_path = run_output_dir / "stdout.log"
    stderr_path = run_output_dir / "stderr.log"

    try:
        if show_logs:
            exit_code, resource_usage = _execute_process_with_streaming(
                cmd,
                stdout_path,
                stderr_path,
                config_name,
                timeout_seconds=timeout_seconds,
                hide_debug_console=hide_debug_console,
            )
        else:
            exit_code, resource_usage = _execute_process_without_streaming(
                cmd, stdout_path, stderr_path, timeout_seconds=timeout_seconds
            )
    except Exception as e:
        logger.error(f"Failed to run config {config_name}: {e}", exc_info=True)
        exit_code = 1
        resource_usage = {
            "peak_memory_mb": None,
            "cpu_time_seconds": None,
            "cpu_percent": None,
        }

    end_time = time.time()
    end_timestamp = datetime.utcnow().isoformat() + "Z"
    duration_seconds = end_time - start_time

    # Detect dry-run mode EARLY (before collecting outputs)
    stdout_path = run_output_dir / "stdout.log"
    configs_to_check = [original_config_copy]
    if modified_config.exists():
        configs_to_check.append(modified_config)

    is_dry_run = _detect_dry_run_from_config(configs_to_check)
    if not is_dry_run:
        is_dry_run = _detect_dry_run_from_stdout(stdout_path)

    # Read stdout content for episode extraction (if needed)
    stdout_content = ""
    if stdout_path.exists():
        try:
            with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
                stdout_content = f.read()
        except Exception:
            pass  # Ignore read errors

    # Copy service output files (metadata, transcripts, summaries) to run directory
    # The service writes to run_output_dir (or a subdirectory within it)
    # We copy those files directly to run_output_dir (flattened structure)
    # Metadata and transcripts folders are kept, but run.json, index.json, etc. are placed flat
    # Skip for dry-run mode (no files are created)
    if not is_dry_run:
        copy_service_outputs(run_output_dir, run_output_dir)

    # Collect logs
    logs = collect_logs_from_output(run_output_dir)

    # Collect outputs (skip for dry-run mode, or set to 0)
    if is_dry_run:
        outputs = {"transcripts": 0, "metadata": 0, "summaries": 0}
    else:
        outputs = collect_outputs(run_output_dir)

    # Determine episodes processed from service output files (run.json, index.json, or metrics.json)
    # These files contain the authoritative episode count from the service
    episodes_processed = 0

    if is_dry_run:
        episodes_processed = _extract_episodes_from_dry_run(stdout_content)
    else:
        # Multi-feed: prefer corpus_run_summary.json (authoritative per-feed counts)
        episodes_processed = _extract_episodes_from_corpus_summary(run_output_dir)

        # Single-feed: try run.json at corpus root
        if episodes_processed == 0:
            run_json_path = run_output_dir / "run.json"
            episodes_processed = _extract_episodes_from_run_json(run_json_path)

        # Fallback: try index.json
        if episodes_processed == 0:
            index_json_path = run_output_dir / "index.json"
            episodes_processed = _extract_episodes_from_index_json(index_json_path)

        # Final fallback: count transcript files
        if episodes_processed == 0:
            episodes_processed = outputs.get("transcripts", 0)
            logger.debug(
                "Using transcript file count as fallback: %d",
                episodes_processed,
            )

    # Log dry-run detection result for debugging
    if is_dry_run:
        logger.info(
            f"✓ Dry-run mode detected for {config_name} (episodes_planned={episodes_processed})"
        )

    # Extract provider/model information from config
    provider_info = _extract_provider_info(original_config_copy)

    timed_out = exit_code == EXIT_TIMEOUT

    cfg_for_vectors: Optional[config.Config] = None
    try:
        cfg_dict_v = config.load_config_file(str(original_config_copy))
        cfg_for_vectors = config.Config.model_validate(cfg_dict_v)
    except Exception as exc:
        logger.debug("Could not load config for vector index check: %s", exc)

    vector_index_ok = True
    vector_index_notes = "not_evaluated"
    if cfg_for_vectors is not None:
        vector_index_ok, vector_index_notes = _assess_vector_index_run(
            run_output_dir, cfg_for_vectors, episodes_processed, is_dry_run
        )

    # Artifact assertions (#622) — opt-in via --assert-artifacts
    artifact_assertions_ok = True
    artifact_assertion_failures: list = []
    if assert_artifacts and exit_code == 0 and not is_dry_run:
        artifact_assertions_ok, artifact_assertion_failures = _assert_artifacts_from_config(
            run_output_dir, config_path, episodes_processed
        )

    effective_exit = exit_code
    if strict_vector_index and not vector_index_ok and effective_exit == 0:
        effective_exit = 1
        logger.error(
            "Strict vector index: failing run %s (notes=%s)",
            config_name,
            vector_index_notes,
        )
    if assert_artifacts and not artifact_assertions_ok and effective_exit == 0:
        effective_exit = 1
        logger.error(
            "Artifact assertions: failing run %s (%d failures)",
            config_name,
            len(artifact_assertion_failures),
        )

    # Build run data (store absolute output_dir so run artifacts are findable regardless of cwd)
    run_data = {
        "run_id": run_id,
        "config_file": str(config_path),
        "config_name": config_name,
        "config_hash": config_hash,
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "duration_seconds": round(duration_seconds, 2),
        "wall_clock_seconds": round(duration_seconds, 2),
        "exit_code": effective_exit,
        "service_exit_code": exit_code,
        "timeout": timed_out,  # True if run was killed by --timeout
        "episodes_processed": episodes_processed,
        "is_dry_run": is_dry_run,  # Flag indicating dry-run mode
        "output_dir": str(run_output_dir.resolve()),
        "logs": logs,
        "outputs": outputs,
        "resource_usage": resource_usage,
        "provider_info": provider_info,  # Provider/model information for benchmarking
        "estimated_cost_usd": _estimate_llm_cost_usd_for_run_dir(run_output_dir),
        "vector_index_ok": vector_index_ok,
        "vector_index_notes": vector_index_notes,
        "strict_vector_index": strict_vector_index,
        "artifact_assertions_ok": artifact_assertions_ok,
        "artifact_assertion_failures": artifact_assertion_failures,
    }

    # Save run data
    run_data_path = run_output_dir / "run_data.json"
    with open(run_data_path, "w") as f:
        json.dump(run_data, f, indent=2)

    assertions_note = ""
    if assert_artifacts and not is_dry_run:
        assertions_note = (
            f", assertions={'PASS' if artifact_assertions_ok else 'FAIL'}"
            f"({len(artifact_assertion_failures)} failures)"
            if not artifact_assertions_ok
            else ", assertions=PASS"
        )

    logger.info(
        f"Completed {config_name}: exit_code={effective_exit}, "
        f"duration={duration_seconds:.1f}s, episodes={episodes_processed}"
        + (" (timed out)" if timed_out else "")
        + (f", vector_index={vector_index_notes}" if not is_dry_run else "")
        + assertions_note
    )

    return run_data


def save_baseline(baseline_id: str, runs_data: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save current runs as a baseline.

    Copies all run data into the baseline folder for easy access and comparison.

    Args:
        baseline_id: Baseline identifier
        runs_data: List of run data dicts
        output_dir: Base output directory
    """
    baseline_dir = output_dir / "baselines" / baseline_id
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Copy run data directories into baseline folder
    for run_data in runs_data:
        run_output_dir = Path(run_data.get("output_dir", ""))
        if run_output_dir.exists():
            # Copy the entire run directory to baseline
            run_name = run_output_dir.name
            baseline_run_dir = baseline_dir / run_name
            if baseline_run_dir.exists():
                # Remove existing if it exists (shouldn't happen, but be safe)
                shutil.rmtree(baseline_run_dir)
            shutil.copytree(run_output_dir, baseline_run_dir)
            logger.debug(f"Copied run data: {run_name} -> {baseline_run_dir}")

    # Save baseline metadata
    baseline_metadata = {
        "baseline_id": baseline_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "runs": runs_data,
        "total_runs": len(runs_data),
        "successful_runs": sum(1 for r in runs_data if r.get("exit_code", 1) == 0),
        "failed_runs": sum(1 for r in runs_data if r.get("exit_code", 1) != 0),
    }

    baseline_path = baseline_dir / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_metadata, f, indent=2)

    logger.info(f"Saved baseline: {baseline_id} ({len(runs_data)} runs)")
    logger.info(f"  Baseline directory: {baseline_dir.absolute()}")


def main() -> None:  # noqa: C901 - CLI orchestrates configs, server, analysis, baselines
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run E2E acceptance tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help=(
            "Config file glob pattern, or multiple space-separated globs "
            "(e.g. 'config/acceptance/*.yaml'). Not required with --from-fast-stems."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".test_outputs/acceptance",
        help="Output directory for results (default: .test_outputs/acceptance)",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Baseline ID to compare current runs against (optional)",
    )
    parser.add_argument(
        "--save-as-baseline",
        type=str,
        default=None,
        help="Save current runs as baseline with this ID (optional)",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        default=False,
        help="Use E2E server fixtures (test feeds and mock APIs). "
        "If not set, uses real RSS feeds and real API providers from config/environment.",
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        default=True,
        help="Stream service logs to console in real-time (default: True)",
    )
    parser.add_argument(
        "--no-show-logs",
        dest="show_logs",
        action="store_false",
        help="Disable streaming service logs to console (only save to files)",
    )
    parser.add_argument(
        "--stream-debug",
        action="store_true",
        default=False,
        help=(
            "When streaming logs to the console, include DEBUG lines. "
            "Default: hide DEBUG on the console (INFO and above still shown); "
            "full output is always written to stdout.log/stderr.log."
        ),
    )
    parser.add_argument(
        "--no-auto-analyze",
        dest="auto_analyze",
        action="store_false",
        help="Disable automatic analysis after session completes (default: enabled)",
    )
    parser.set_defaults(auto_analyze=True)
    parser.add_argument(
        "--no-auto-benchmark",
        dest="auto_benchmark",
        action="store_false",
        help=(
            "Disable automatic performance benchmark report generation "
            "after session completes (default: enabled)"
        ),
    )
    parser.set_defaults(auto_benchmark=True)
    parser.add_argument(
        "--analyze-mode",
        type=str,
        default="basic",
        choices=["basic", "comprehensive"],
        help="Analysis mode for auto-analysis (default: basic)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Per-run timeout in seconds. If a config run exceeds this, it is killed and "
        "recorded as failed (exit 124). Useful for CI or long summarization suites.",
    )
    parser.add_argument(
        "--per-run-wall-seconds",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "Wall-clock budget per config run (seconds). After each run, if elapsed wall time "
            f"exceeds this value, the run fails with exit {EXIT_PER_RUN_WALL_BUDGET} when the "
            "service exit was 0. Use 0 to disable. Default: ACCEPTANCE_PER_RUN_WALL_SECONDS "
            "environment variable or 600. This is independent of --timeout (subprocess kill); "
            "it surfaces slow successful runs before the outer job budget is hit."
        ),
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        default=False,
        help=(
            "After resolving --configs globs, keep only files whose stem matches a row "
            f"``id`` in config/acceptance/{MAIN_ACCEPTANCE_CONFIG_BASENAME} (enabled runs only)."
        ),
    )
    parser.add_argument(
        "--from-fast-stems",
        action="store_true",
        default=False,
        help=(
            "Ignore --configs; materialize each enabled row from "
            f"config/acceptance/{MAIN_ACCEPTANCE_CONFIG_BASENAME} into "
            "session_dir/materialized/{id}.yaml "
            "and run those (pair with --use-fixtures for CI fixture smoke)."
        ),
    )
    parser.add_argument(
        "--strict-vector-index",
        action="store_true",
        default=False,
        help=(
            "Treat empty FAISS output (vector_search on, episodes processed, but no rows in "
            "search/metadata.json) as a failed run (exit_code 1). Same as STRICT_VECTOR_INDEX=1."
        ),
    )
    parser.add_argument(
        "--assert-artifacts",
        action="store_true",
        default=False,
        help=(
            "After each run, validate artifact content against assertions spec "
            "(*.assertions.yaml alongside config). Fail the run if assertions fail. "
            "Off by default for exploration mode. Same as ASSERT_ARTIFACTS=1."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    per_run_wall_budget = _resolved_per_run_wall_seconds(args.per_run_wall_seconds)

    # Check E2E server availability (only needed if using fixtures)
    use_fixtures = args.use_fixtures
    e2e_server = None

    if use_fixtures:
        if E2EHTTPServer is None:
            logger.error(
                "E2E server not available. Cannot use fixtures without E2E server. "
                "Either install test dependencies or use --use-fixtures=false to use real feeds."
            )
            sys.exit(1)

        # Start E2E server
        logger.info("Starting E2E server for fixture feeds and mock APIs...")
        e2e_server = E2EHTTPServer()
        e2e_server.start()

    # Output + session dirs first (``--from-fast-stems`` materializes under session_dir).
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / "sessions" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = session_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Find config files
    if args.from_fast_stems:
        if args.configs:
            logger.info("--from-fast-stems set; ignoring --configs %r", args.configs)
        fast_ids = load_fast_matrix_ids()
        if not fast_ids:
            logger.error(
                "No acceptance matrix rows: add config/acceptance/%s with "
                "non-empty runs (see config/acceptance/README.md).",
                MAIN_ACCEPTANCE_CONFIG_BASENAME,
            )
            sys.exit(1)
        try:
            config_files = materialize_fast_matrix_configs(session_dir)
        except (OSError, ValueError, FileNotFoundError) as exc:
            logger.error("Failed to materialize fast matrix: %s", exc)
            sys.exit(1)
        if not config_files:
            logger.error(
                "No materialized configs for acceptance matrix (check %s runs).",
                MAIN_ACCEPTANCE_CONFIG_BASENAME,
            )
            sys.exit(1)
        logger.info(
            "--from-fast-stems: running %d config(s) from %s (%d row id(s))",
            len(config_files),
            MAIN_ACCEPTANCE_CONFIG_BASENAME,
            len(fast_ids),
        )
    else:
        if not args.configs:
            logger.error("No --configs given (or use --from-fast-stems)")
            sys.exit(1)
        config_files = find_config_files(args.configs)
        if not config_files:
            logger.error("No config files found")
            sys.exit(1)

        # Optionally restrict to fast subset (for CI: run fast configs on PR, full suite nightly)
        if args.fast_only:
            fast_ids = load_fast_matrix_ids()
            if not fast_ids:
                logger.warning(
                    "Acceptance matrix empty or missing; --fast-only has no effect. "
                    "See config/acceptance/%s.",
                    MAIN_ACCEPTANCE_CONFIG_BASENAME,
                )
            else:
                before = len(config_files)
                config_files = filter_fast_configs(config_files, fast_ids)
                logger.info(
                    "--fast-only: running %d of %d configs (from %s row ids)",
                    len(config_files),
                    before,
                    MAIN_ACCEPTANCE_CONFIG_BASENAME,
                )
                if not config_files:
                    logger.error("No configs matched %s row ids", MAIN_ACCEPTANCE_CONFIG_BASENAME)
                    sys.exit(1)

    # Reuse RSS feed XML across sequential configs (same URL hits cache after first fetch).
    # See podcast_scraper.rss.feed_cache (PODCAST_SCRAPER_RSS_CACHE_DIR); off by default for CLI.
    rss_cache_dir = apply_session_rss_cache_env(session_dir)

    _log_acceptance_session_header(
        num_configs=len(config_files),
        use_fixtures=use_fixtures,
        e2e_server=e2e_server,
        output_dir=output_dir,
        session_dir=session_dir,
        rss_cache_dir=rss_cache_dir,
        runs_dir=runs_dir,
        baselines_dir=output_dir / "baselines",
    )
    if per_run_wall_budget > 0:
        logger.info(
            "Per-run wall budget: %ds (post-run check; exit %d if exceeded when service exit=0); "
            "subprocess --timeout: %s",
            per_run_wall_budget,
            EXIT_PER_RUN_WALL_BUDGET,
            f"{args.timeout}s" if args.timeout else "none",
        )

    try:
        # Run all configs sequentially
        runs_data = []

        session_summary_path = session_dir / "session.json"

        def _write_session_summary(runs: List[Dict[str, Any]]) -> None:
            """Persist current session summary so analysis works after partial/interrupted runs."""
            per_run_costs = [
                r.get("estimated_cost_usd")
                for r in runs
                if isinstance(r.get("estimated_cost_usd"), (int, float))
            ]
            session_cost: Optional[float] = None
            if per_run_costs:
                session_cost = round(sum(per_run_costs), 6)

            walltime_vs: Optional[Dict[str, Any]] = None
            if args.compare_baseline:
                walltime_vs = compute_walltime_vs_baseline_summary(
                    runs, args.compare_baseline, output_dir
                )

            total_wall = sum(
                float(r.get("wall_clock_seconds", r.get("duration_seconds", 0)) or 0.0)
                for r in runs
            )

            summary = {
                "session_id": session_id,
                "start_time": runs[0]["start_time"] if runs else None,
                "end_time": runs[-1]["end_time"] if runs else None,
                "total_runs": len(runs),
                "successful_runs": sum(1 for r in runs if r["exit_code"] == 0),
                "failed_runs": sum(1 for r in runs if r["exit_code"] != 0),
                "total_duration_seconds": sum(r["duration_seconds"] for r in runs),
                "total_wall_clock_seconds": round(total_wall, 2),
                "per_run_wall_budget_seconds": per_run_wall_budget,
                "acceptance_matrix_file": (
                    MAIN_ACCEPTANCE_CONFIG_BASENAME if args.from_fast_stems else None
                ),
                "walltime_vs_baseline": walltime_vs,
                "estimated_session_cost_usd": session_cost,
                "config_files": [str(cf) for cf in config_files],
                "runs": runs,
            }
            with open(session_summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        total_cfgs = len(config_files)
        hide_debug = not args.stream_debug
        strict_vec = bool(args.strict_vector_index) or _strict_vector_index_requested()
        for i, config_file in enumerate(config_files, 1):
            if total_cfgs > 1:
                logger.info("")
                logger.info("── Run %d/%d: %s ──", i, total_cfgs, config_file.name)
            run_id = f"{config_file.stem}_{session_id}"
            assert_artifacts = getattr(args, "assert_artifacts", False) or os.environ.get(
                "ASSERT_ARTIFACTS", ""
            ).strip().lower() in ("1", "true", "yes")
            run_data = run_config(
                config_file,
                e2e_server,
                runs_dir,  # Pass runs_dir instead of output_dir
                run_id,
                use_fixtures=use_fixtures,
                show_logs=args.show_logs,
                timeout_seconds=args.timeout,
                hide_debug_console=hide_debug,
                run_index=i if total_cfgs > 1 else None,
                run_total=total_cfgs if total_cfgs > 1 else None,
                strict_vector_index=strict_vec,
                assert_artifacts=assert_artifacts,
            )
            _apply_per_run_wall_budget_failure(run_data, per_run_wall_budget)
            runs_data.append(run_data)
            # Persist after each config so analysis works if run is interrupted
            _write_session_summary(runs_data)

        logger.info(f"Session summary saved: {session_summary_path}")

        # Run automatic analysis if enabled (default)
        logger.debug(f"Auto-analyze setting: {args.auto_analyze}")
        if args.auto_analyze:
            logger.info("")
            logger.info("Running automatic analysis...")
            try:
                # Use absolute path to the analysis script
                analyze_script = Path(__file__).parent / "analyze_bulk_runs.py"
                if not analyze_script.exists():
                    raise FileNotFoundError(f"Analysis script not found: {analyze_script}")

                analyze_cmd = [
                    sys.executable,
                    str(analyze_script.absolute()),
                    "--session-id",
                    session_id,
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    args.analyze_mode,
                    "--output-format",
                    "both",
                    "--log-level",
                    "INFO",
                ]
                if args.compare_baseline:
                    analyze_cmd.extend(["--compare-baseline", args.compare_baseline])

                logger.debug(f"Running analysis command: {' '.join(analyze_cmd)}")
                result = subprocess.run(
                    analyze_cmd,
                    capture_output=False,
                )
                if result.returncode == 0:
                    logger.info("✓ Automatic analysis completed")
                else:
                    logger.warning(f"Analysis completed with exit code {result.returncode}")

                # Also generate performance benchmark report (if enabled)
                if args.auto_benchmark:
                    benchmark_script = Path(__file__).parent / "generate_performance_benchmark.py"
                    if benchmark_script.exists():
                        logger.info("")
                        logger.info("Generating performance benchmark report...")
                        benchmark_cmd = [
                            sys.executable,
                            str(benchmark_script.absolute()),
                            "--session-id",
                            session_id,
                            "--output-dir",
                            str(output_dir),
                            "--output-format",
                            "both",
                            "--log-level",
                            "INFO",
                        ]
                        if args.compare_baseline:
                            benchmark_cmd.extend(["--compare-baseline", args.compare_baseline])
                        logger.debug(f"Running benchmark command: {' '.join(benchmark_cmd)}")
                        benchmark_result = subprocess.run(
                            benchmark_cmd,
                            capture_output=False,
                        )
                        if benchmark_result.returncode == 0:
                            logger.info("✓ Performance benchmark report generated")
                        else:
                            logger.warning(
                                f"Benchmark generation completed with exit code "
                                f"{benchmark_result.returncode}"
                            )
                    else:
                        logger.warning(
                            "Benchmark script not found, skipping performance "
                            "benchmark report generation"
                        )
                else:
                    logger.debug(
                        "Auto-benchmark disabled, skipping performance "
                        "benchmark report generation"
                    )
            except Exception as e:
                logger.error(f"Failed to run automatic analysis: {e}", exc_info=True)
                logger.info(
                    "You can run analysis manually with: "
                    f"make analyze-acceptance SESSION_ID={session_id}"
                )

        # Save as baseline if requested
        if args.save_as_baseline:
            save_baseline(args.save_as_baseline, runs_data, output_dir)

        # Compare with baseline if requested
        if args.compare_baseline:
            baseline_path = output_dir / "baselines" / args.compare_baseline / "baseline.json"
            if baseline_path.exists():
                logger.info(f"Baseline comparison requested: {args.compare_baseline}")
                # Comparison will be done by analysis script
                logger.info(
                    "Use scripts/acceptance/analyze_bulk_runs.py to generate comparison report"
                )
            else:
                logger.warning(f"Baseline not found: {baseline_path}. " "Skipping comparison.")

        logger.info("Acceptance tests completed")
        _log_session_estimated_llm_cost(runs_data)

        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_dir.absolute()}")
        logger.info(f"  - Session folder: {session_dir.absolute()}")
        logger.info(f"  - Session summary: {session_summary_path.absolute()}")
        logger.info(f"  - Run data: {runs_dir.absolute()}")
        if args.auto_analyze:
            logger.info(
                f"  - Analysis reports: {session_dir / f'report_{session_id}.md'} and .json"
            )
        logger.info(f"  - Baselines: {(output_dir / 'baselines').absolute()}")
        logger.info("=" * 70)

        failed_count = sum(1 for r in runs_data if r.get("exit_code", 0) != 0)
        if failed_count:
            logger.error(
                "%d acceptance run(s) failed (see session.json exit_code / vector_index fields).",
                failed_count,
            )
            sys.exit(1)

        walltime_vs_final: Optional[Dict[str, Any]] = None
        if args.compare_baseline:
            walltime_vs_final = compute_walltime_vs_baseline_summary(
                runs_data, args.compare_baseline, output_dir
            )
        regressions = (walltime_vs_final or {}).get("regressions_ge_threshold") or []
        if regressions:
            pct = int(round((WALLTIME_REGRESSION_RATIO - 1.0) * 100))
            logger.error(
                "Wall-clock regression vs baseline %r: %d run(s) ≥%d%% slower than baseline "
                "(see session.json → walltime_vs_baseline).",
                args.compare_baseline,
                len(regressions),
                pct,
            )
            for row in regressions:
                logger.error(
                    "  %s: baseline %.2fs → current %.2fs (ratio %.3f)",
                    row.get("config_name"),
                    float(row.get("baseline_wall_seconds", 0)),
                    float(row.get("current_wall_seconds", 0)),
                    float(row.get("slowdown_ratio", 0)),
                )
            sys.exit(1)

    finally:
        # Stop E2E server (if it was started)
        if e2e_server:
            logger.info("Stopping E2E server...")
            e2e_server.stop()


if __name__ == "__main__":
    main()
