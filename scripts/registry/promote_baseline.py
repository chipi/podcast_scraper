"""Promote an eval baseline configuration into the code Model Registry.

Implements RFC-044 promotion mechanism: read a baseline's `config.yaml` (and
optional `metrics.json`) from `data/eval/baselines/`, validate it, then append a
ModeConfiguration entry into `src/podcast_scraper/providers/ml/model_registry.py`.

The runtime application must never import or reference `data/eval/`. Promotion is
the explicit, version-controlled bridge between experimentation and app defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import yaml

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from podcast_scraper.providers.ml.model_registry import ModeConfiguration


def _ensure_src_on_path() -> None:
    """Ensure `src/` is importable when running as a script."""
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _utc_now_iso_z() -> str:
    """Return current UTC timestamp as ISO string ending in 'Z'."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(raw).__name__}")
    return raw


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(raw).__name__}")
    return raw


def _extract_metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact, stable summary of metrics.json for the registry."""
    intrinsic = metrics.get("intrinsic") if isinstance(metrics.get("intrinsic"), dict) else {}
    length = intrinsic.get("length") if isinstance(intrinsic.get("length"), dict) else {}
    gates = intrinsic.get("gates") if isinstance(intrinsic.get("gates"), dict) else {}
    perf = intrinsic.get("performance") if isinstance(intrinsic.get("performance"), dict) else {}

    return {
        "dataset_id": metrics.get("dataset_id"),
        "run_id": metrics.get("run_id"),
        "episode_count": metrics.get("episode_count"),
        "intrinsic": {
            "gates": {
                "boilerplate_leak_rate": gates.get("boilerplate_leak_rate"),
                "speaker_label_leak_rate": gates.get("speaker_label_leak_rate"),
                "truncation_rate": gates.get("truncation_rate"),
            },
            "length": {
                "avg_tokens": length.get("avg_tokens"),
                "min_tokens": length.get("min_tokens"),
                "max_tokens": length.get("max_tokens"),
            },
            "performance": {"avg_latency_ms": perf.get("avg_latency_ms")},
        },
    }


def _render_py_literal(value: Any, indent: str) -> str:
    """Render a Python literal with indentation suitable for multiline dicts."""
    import pprint

    rendered = pprint.pformat(value, width=88, sort_dicts=False)
    if "\n" not in rendered:
        return rendered
    lines = rendered.splitlines()
    return "\n".join([lines[0], *[indent + line for line in lines[1:]]])


def _render_mode_entry(mode: ModeConfiguration, indent: str = " " * 8) -> str:
    """Render a single ModeConfiguration entry for insertion into a dict literal."""
    # Indent for continuation lines under `name=...`
    cont = indent + " " * 4

    map_params = _render_py_literal(mode.map_params, indent=cont)
    reduce_params = _render_py_literal(mode.reduce_params, indent=cont)
    tokenize = _render_py_literal(mode.tokenize, indent=cont)
    chunking = (
        _render_py_literal(mode.chunking, indent=cont) if mode.chunking is not None else "None"
    )
    metrics = (
        _render_py_literal(mode.metrics_summary, indent=cont)
        if mode.metrics_summary is not None
        else "None"
    )

    return (
        f"{indent}{mode.mode_id!r}: ModeConfiguration(\n"
        f"{indent}    mode_id={mode.mode_id!r},\n"
        f"{indent}    map_model={mode.map_model!r},\n"
        f"{indent}    reduce_model={mode.reduce_model!r},\n"
        f"{indent}    preprocessing_profile={mode.preprocessing_profile!r},\n"
        f"{indent}    map_params={map_params},\n"
        f"{indent}    reduce_params={reduce_params},\n"
        f"{indent}    tokenize={tokenize},\n"
        f"{indent}    chunking={chunking},\n"
        f"{indent}    promoted_from={mode.promoted_from!r},\n"
        f"{indent}    promoted_at={mode.promoted_at!r},\n"
        f"{indent}    metrics_summary={metrics},\n"
        f"{indent}),\n"
    )


def _insert_mode_into_registry_file(registry_path: Path, mode: ModeConfiguration) -> None:
    """Append a mode entry between the BEGIN/END MODE REGISTRY markers."""
    begin = "# BEGIN MODE REGISTRY (append-only)"
    end = "# END MODE REGISTRY (append-only)"

    text = registry_path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        raise ValueError(
            "Registry file missing mode markers. Expected lines:\n"
            f"  {begin}\n"
            f"  {end}\n"
            f"File: {registry_path}"
        )

    if f"{mode.mode_id!r}:" in text:
        raise ValueError(f"Mode '{mode.mode_id}' already exists in registry: {registry_path}")

    before, rest = text.split(begin, 1)
    middle, after = rest.split(end, 1)

    insertion = _render_mode_entry(mode)
    # Ensure the insertion lands on its own line and keeps the block readable.
    middle = middle.rstrip() + "\n" + insertion + " " * 8

    updated = before + begin + middle + end + after
    registry_path.write_text(updated, encoding="utf-8")


def promote_baseline(
    *,
    baseline_dir: Path,
    mode_id: str,
    registry_path: Path,
    baseline_id: Optional[str] = None,
) -> ModeConfiguration:
    """Promote a baseline directory to a ModeConfiguration and append to registry.

    Args:
        baseline_dir: Directory containing `config.yaml` and optionally `metrics.json`.
        mode_id: Mode ID to write into the registry.
        registry_path: Path to `model_registry.py` to update.
        baseline_id: Optional baseline ID override (otherwise derived).

    Returns:
        The promoted ModeConfiguration.
    """
    _ensure_src_on_path()

    from podcast_scraper.preprocessing import profiles
    from podcast_scraper.providers.ml.model_registry import ModeConfiguration, ModelRegistry

    config_path = baseline_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Baseline config not found: {config_path}")

    raw = _load_yaml(config_path)
    effective_baseline_id = baseline_id or str(raw.get("id") or baseline_dir.name)

    backend = raw.get("backend") if isinstance(raw.get("backend"), dict) else {}
    map_model = backend.get("map_model") or raw.get("map_model")
    reduce_model = backend.get("reduce_model") or raw.get("reduce_model")
    preprocessing_profile = raw.get("preprocessing_profile")
    map_params = raw.get("map_params")
    reduce_params = raw.get("reduce_params")
    tokenize = raw.get("tokenize")
    chunking = raw.get("chunking")

    if not isinstance(map_model, str) or not map_model.strip():
        raise ValueError("Baseline config missing backend.map_model")
    if not isinstance(reduce_model, str) or not reduce_model.strip():
        raise ValueError("Baseline config missing backend.reduce_model")
    if not isinstance(preprocessing_profile, str) or not preprocessing_profile.strip():
        raise ValueError("Baseline config missing preprocessing_profile")
    if not isinstance(map_params, dict):
        raise ValueError("Baseline config missing map_params mapping")
    if not isinstance(reduce_params, dict):
        raise ValueError("Baseline config missing reduce_params mapping")
    if not isinstance(tokenize, dict):
        raise ValueError("Baseline config missing tokenize mapping")
    if chunking is not None and not isinstance(chunking, dict):
        raise ValueError("Baseline config chunking must be a mapping when provided")

    # Validate preprocessing profile is registered.
    profiles.get_profile(preprocessing_profile)

    # Validate models are explicitly registered (no pattern fallbacks for promotion).
    if map_model not in ModelRegistry._registry:
        raise ValueError(f"MAP model '{map_model}' is not registered in ModelRegistry")
    if reduce_model not in ModelRegistry._registry:
        raise ValueError(f"REDUCE model '{reduce_model}' is not registered in ModelRegistry")

    metrics_summary: Optional[Dict[str, Any]] = None
    metrics_path = baseline_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        metrics_summary = _extract_metrics_summary(metrics)

    mode = ModeConfiguration(
        mode_id=mode_id,
        map_model=map_model,
        reduce_model=reduce_model,
        preprocessing_profile=preprocessing_profile,
        map_params=dict(map_params),
        reduce_params=dict(reduce_params),
        tokenize=dict(tokenize),
        chunking=dict(chunking) if isinstance(chunking, dict) else None,
        promoted_from=effective_baseline_id,
        promoted_at=_utc_now_iso_z(),
        metrics_summary=metrics_summary,
    )

    _insert_mode_into_registry_file(registry_path, mode)
    logger.info(
        "Promoted baseline '%s' → mode '%s' in %s", effective_baseline_id, mode_id, registry_path
    )
    return mode


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline-id",
        required=True,
        help="Baseline ID (directory name under data/eval/baselines/)",
    )
    p.add_argument("--mode-id", required=True, help="Mode ID to write into the registry")
    p.add_argument(
        "--baseline-dir",
        default=None,
        help="Baseline directory path (defaults to data/eval/baselines/{baseline_id})",
    )
    p.add_argument(
        "--registry-path",
        default="src/podcast_scraper/providers/ml/model_registry.py",
        help="Path to model_registry.py to update",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    baseline_dir = (
        Path(args.baseline_dir)
        if args.baseline_dir
        else (project_root / "data" / "eval" / "baselines" / args.baseline_id)
    )
    registry_path = project_root / str(args.registry_path)

    promote_baseline(
        baseline_dir=baseline_dir,
        mode_id=str(args.mode_id),
        registry_path=registry_path,
        baseline_id=str(args.baseline_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
