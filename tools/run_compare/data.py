"""Load eval run artifacts for the run comparison tool (RFC-047)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACTS = ("metrics.json", "predictions.jsonl", "fingerprint.json")
OPTIONAL_ARTIFACTS = ("run_summary.json", "diagnostics.jsonl")


@dataclass(frozen=True)
class RunEntry:
    """One directory under data/eval that contains metrics.json."""

    run_id: str
    rel_label: str
    path: Path
    category: str


def repo_root_from_here() -> Path:
    """Project root (parent of ``tools/run_compare``)."""
    return Path(__file__).resolve().parent.parent.parent


def discover_runs(eval_root: Optional[Path] = None) -> List[RunEntry]:
    """Scan ``data/eval/{runs,baselines,references}/`` for ``metrics.json`` trees.

    Skips ``_archived`` and path segments starting with ``_``.

    Args:
        eval_root: Base ``data/eval`` directory (default: repo ``data/eval``).

    Returns:
        Sorted list of run entries.
    """
    root = eval_root or Path("data/eval")
    if not root.is_absolute():
        root = repo_root_from_here() / root
    entries: List[RunEntry] = []
    for sub, cat in (
        ("runs", "run"),
        ("baselines", "baseline"),
        ("references", "reference"),
    ):
        base = root / sub
        if not base.exists():
            continue
        for metrics_path in sorted(base.rglob("metrics.json")):
            parent = metrics_path.parent
            if "_archived" in parent.parts:
                continue
            if any(part.startswith("_") for part in parent.parts):
                continue
            try:
                rel_label = parent.relative_to(root).as_posix()
            except ValueError:
                rel_label = parent.name
            entries.append(
                RunEntry(
                    run_id=parent.name,
                    rel_label=rel_label,
                    path=parent.resolve(),
                    category=cat,
                )
            )
    entries.sort(key=lambda e: e.rel_label)
    return entries


def artifact_status(run_dir: Path) -> Dict[str, bool]:
    """Return presence flags for known artifact filenames."""
    out: Dict[str, bool] = {}
    for name in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS:
        out[name] = (run_dir / name).is_file()
    return out


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load ``metrics.json`` from a run directory."""
    return load_json(run_dir / "metrics.json")


def load_predictions_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all JSON lines from ``predictions.jsonl``."""
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("Skipping bad JSONL line in %s: %s", path, e)
    return rows


def load_diagnostics_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load ``diagnostics.jsonl`` rows."""
    return load_predictions_jsonl(path)


def extract_rouge_l_f1(metrics: Dict[str, Any]) -> Optional[float]:
    """Mean ROUGE-L F1 from first ``vs_reference`` entry with ``rougeL_f1`` set."""
    vs = metrics.get("vs_reference")
    if not isinstance(vs, dict):
        return None
    for _ref_id, blob in vs.items():
        if not isinstance(blob, dict) or "error" in blob:
            continue
        val = blob.get("rougeL_f1")
        if val is not None:
            return float(val)
    return None


def extract_aggregate_rouge(
    metrics: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Optional[float]]]]:
    """First ``vs_reference`` block that reports any ROUGE F1 (for run-level charts)."""
    vs = metrics.get("vs_reference")
    if not isinstance(vs, dict):
        return None
    for ref_id, blob in vs.items():
        if not isinstance(blob, dict) or "error" in blob:
            continue
        r1, r2, r3 = blob.get("rouge1_f1"), blob.get("rouge2_f1"), blob.get("rougeL_f1")
        if r1 is None and r2 is None and r3 is None:
            continue
        return (
            ref_id,
            {
                "rouge1_f1": float(r1) if r1 is not None else None,
                "rouge2_f1": float(r2) if r2 is not None else None,
                "rougeL_f1": float(r3) if r3 is not None else None,
            },
        )
    return None


def pick_shared_reference_id(
    loaded: Dict[str, Dict[str, Any]],
    ordered_labels: Optional[List[str]] = None,
) -> Tuple[Optional[str], bool]:
    """Reference id for per-episode ROUGE (first selected run; all same → consistent)."""
    keys = ordered_labels if ordered_labels is not None else list(loaded.keys())
    ids: List[str] = []
    for label in keys:
        data = loaded.get(label)
        if not data:
            continue
        hit = extract_aggregate_rouge(data["metrics"])
        if hit:
            ids.append(hit[0])
    if not ids:
        return None, True
    first = ids[0]
    return first, all(r == first for r in ids)


def reference_predictions_path(eval_root: Path, ref_id: str) -> Path:
    """Path to ``data/eval/references/<ref_id>/predictions.jsonl``."""
    return eval_root / "references" / ref_id / "predictions.jsonl"


def compute_per_episode_rouge_rows(
    run_label: str,
    preds: List[Dict[str, Any]],
    failed_ids: Optional[Iterable[str]],
    ref_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-episode ROUGE vs reference text (same scoring as ``compute_rouge_vs_reference``)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return []

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    failed_set = set(failed_ids or [])
    rows: List[Dict[str, Any]] = []
    for p in preds:
        if not is_rouge_comparable_prediction(p, failed_set):
            continue
        eid = str(p.get("episode_id") or "")
        ref_p = ref_by_id.get(eid)
        if not ref_p:
            continue
        pred_text = _output_summary_text_for_rouge(p)
        ref_text = _output_summary_text_for_rouge(ref_p)
        if not pred_text or not ref_text:
            continue
        scores = scorer.score(ref_text, pred_text)
        rows.append(
            {
                "run": run_label,
                "episode_id": eid,
                "rouge1_f1": scores["rouge1"].fmeasure,
                "rouge2_f1": scores["rouge2"].fmeasure,
                "rougeL_f1": scores["rougeL"].fmeasure,
            }
        )
    return rows


def _output_summary_text_for_rouge(pred: Dict[str, Any]) -> str:
    """Summary text used for ROUGE-style checks (aligned with ``compute_rouge_vs_reference``)."""
    out = pred.get("output")
    if isinstance(out, dict):
        text = out.get("summary_final") or out.get("summary_long") or out.get("summary")
        return str(text or "").strip()
    if isinstance(out, str):
        return out.strip()
    return str(out or "").strip()


def is_rouge_comparable_prediction(pred: Dict[str, Any], failed_episode_ids: set) -> bool:
    """True if the scorer would score this prediction for ROUGE (pred side).

    Excludes gate failures, inference errors, and empty summaries — same as
    ``compute_rouge_vs_reference`` skipping empty prediction text.
    """
    eid = str(pred.get("episode_id") or "")
    if not eid:
        return False
    if eid in failed_episode_ids:
        return False
    err = pred.get("error")
    if isinstance(err, str) and err.strip():
        return False
    if not _output_summary_text_for_rouge(pred):
        return False
    return True


def estimate_output_tokens(pred: Dict[str, Any]) -> float:
    """Rough token estimate from prediction row (chars/4 fallback)."""
    meta = pred.get("metadata") or {}
    chars = meta.get("output_length_chars")
    if isinstance(chars, (int, float)) and chars > 0:
        return float(chars) / 4.0
    out = pred.get("output")
    if isinstance(out, dict):
        text = out.get("summary_final") or out.get("summary") or ""
    else:
        text = str(out or "")
    return max(float(len(text)) / 4.0, 1.0)


def extract_kpis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate KPIs from ``metrics.json``."""
    intrinsic = metrics.get("intrinsic") or {}
    gates = intrinsic.get("gates") or {}
    length = intrinsic.get("length") or {}
    perf = intrinsic.get("performance") or {}
    episode_count = int(metrics.get("episode_count") or 0)
    failed_episodes = list(gates.get("failed_episodes") or [])
    n_failed = len(failed_episodes)
    success_rate = ((episode_count - n_failed) / episode_count) if episode_count else 0.0
    avg_tokens = length.get("avg_tokens")
    avg_latency_ms = perf.get("avg_latency_ms")
    avg_latency_s = (float(avg_latency_ms) / 1000.0) if avg_latency_ms is not None else None
    return {
        "episode_count": episode_count,
        "success_rate": success_rate,
        "failed_count": n_failed,
        "avg_output_tokens": avg_tokens,
        "avg_latency_s": avg_latency_s,
        "truncation_rate": gates.get("truncation_rate"),
        "speaker_label_leak_rate": gates.get("speaker_label_leak_rate"),
        "boilerplate_leak_rate": gates.get("boilerplate_leak_rate"),
        "failed_episodes": failed_episodes,
        "rougeL_f1": extract_rouge_l_f1(metrics),
    }


def merge_run_summary(metrics: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Optionally overlay ``run_summary.json`` for faster aggregates."""
    p = run_dir / "run_summary.json"
    if not p.is_file():
        return metrics
    try:
        summary = load_json(p)
    except (OSError, json.JSONDecodeError):
        return metrics
    out = dict(metrics)
    if "avg_output_tokens" in summary:
        intrinsic = dict(out.get("intrinsic") or {})
        length = dict(intrinsic.get("length") or {})
        length["avg_tokens"] = summary["avg_output_tokens"]
        intrinsic["length"] = length
        out["intrinsic"] = intrinsic
    if "avg_latency_s" in summary:
        intrinsic = dict(out.get("intrinsic") or {})
        perf = dict(intrinsic.get("performance") or {})
        perf["avg_latency_ms"] = float(summary["avg_latency_s"]) * 1000.0
        intrinsic["performance"] = perf
        out["intrinsic"] = intrinsic
    return out


def predictions_to_chart_rows(
    run_label: str,
    preds: Iterable[Dict[str, Any]],
    failed_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Flatten predictions for Plotly charts.

    Only includes **ROUGE-comparable** episodes: no gate failure, no inference error,
    non-empty summary text (same class as ``compute_rouge_vs_reference`` uses).
    """
    failed_set = set(failed_ids or [])
    rows: List[Dict[str, Any]] = []
    for p in preds:
        if not is_rouge_comparable_prediction(p, failed_set):
            continue
        eid = str(p.get("episode_id") or "")
        meta = p.get("metadata") or {}
        lat = meta.get("processing_time_seconds")
        rows.append(
            {
                "run": run_label,
                "episode_id": eid,
                "latency_s": float(lat) if lat is not None else None,
                "output_tokens_est": estimate_output_tokens(p),
                "failed": False,
            }
        )
    return rows


def rouge_comparable_episode_ids(
    preds: List[Dict[str, Any]],
    failed_ids: Optional[Iterable[str]] = None,
) -> set:
    """Episode ids that are ROUGE-comparable for this run."""
    failed_set = set(failed_ids or [])
    out: set = set()
    for p in preds:
        if is_rouge_comparable_prediction(p, failed_set):
            eid = str(p.get("episode_id") or "")
            if eid:
                out.add(eid)
    return out


def get_summary_text(pred: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (summary_text, error_message) for drill-down."""
    err = pred.get("error")
    if isinstance(err, str) and err.strip():
        return "", err.strip()
    out = pred.get("output")
    if isinstance(out, dict):
        text = out.get("summary_final") or out.get("summary")
        if text is not None:
            return str(text), None
    return str(out or ""), None


def index_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map episode_id -> prediction row."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in preds:
        eid = str(p.get("episode_id") or "")
        if eid:
            out[eid] = p
    return out


def delta_direction_good(metric_key: str, delta: float) -> Optional[bool]:
    """Return True if delta is good, False if bad, None if neutral."""
    lower_better = {
        "failed_count",
        "avg_latency_s",
        "truncation_rate",
        "speaker_label_leak_rate",
        "boilerplate_leak_rate",
    }
    higher_better = {"success_rate", "avg_output_tokens", "rougeL_f1"}
    if metric_key in lower_better:
        if delta == 0:
            return True
        return delta < 0
    if metric_key in higher_better:
        if delta == 0:
            return True
        return delta > 0
    return None
