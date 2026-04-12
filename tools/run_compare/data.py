"""Load eval run artifacts for the run comparison tool (RFC-047, RFC-066)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

REQUIRED_ARTIFACTS = ("metrics.json", "predictions.jsonl", "fingerprint.json")
OPTIONAL_ARTIFACTS = ("run_summary.json", "diagnostics.jsonl")


@dataclass(frozen=True)
class RunEntry:
    """One directory under data/eval that contains metrics.json."""

    run_id: str
    rel_label: str
    path: Path
    category: str


def longest_common_prefix(strings: Sequence[str]) -> str:
    """Longest prefix shared by all non-empty strings (empty if ``strings`` is empty)."""
    if not strings:
        return ""
    first = strings[0]
    for i in range(len(first)):
        for s in strings[1:]:
            if i >= len(s) or s[i] != first[i]:
                return first[:i]
    return first


def _truncate_run_display(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return f"…{s[-(max_chars - 1) :]}"


def _path_segments(rel_label: str) -> List[str]:
    s = rel_label.replace("\\", "/").strip()
    return [p for p in s.split("/") if p]


def _fit_tail_path_to_width(parts: Sequence[str], max_chars: int) -> str:
    """Keep path tail; drop whole segments from the left, then hard-clip the leaf."""
    if not parts:
        return ""
    for k in range(len(parts), 0, -1):
        chunk = "/".join(parts[-k:])
        if len(chunk) <= max_chars:
            return chunk
    leaf = parts[-1]
    return leaf[-max_chars:] if len(leaf) > max_chars else leaf


def compact_baseline_select_labels(
    rel_labels: Sequence[str],
    *,
    max_chars: int = 36,
) -> Dict[str, str]:
    """Map each option to a **suffix-first** sidebar label.

    Streamlit/Base Web clips overflowing select text with a **left-anchored** ellipsis,
    so strings that *start* with ``…/tail`` still hide the tail. Here the displayed
    string **begins** with the distinguishing path tail (last segments), and any
    further shortening drops segments or clips the leaf from the **left** so the
    **end** of the path stays visible.
    """
    unique = list(dict.fromkeys(rel_labels))
    if not unique:
        return {}

    parts_per = [_path_segments(s) for s in unique]

    def joined_tail(parts: List[str], k: int) -> str:
        if not parts:
            return ""
        kk = min(max(k, 1), len(parts))
        return "/".join(parts[-kk:])

    if len(unique) == 1:
        lab = unique[0]
        p0 = parts_per[0]
        if not p0:
            raw = lab.replace("\\", "/")
            return {lab: raw[-max_chars:] if len(raw) > max_chars else raw}
        return {lab: _fit_tail_path_to_width(p0, max_chars)}

    max_depth = max((len(p) for p in parts_per), default=0)
    best_k = max_depth
    for k in range(1, max_depth + 1):
        labels_k = [joined_tail(p, k) if p else s for p, s in zip(parts_per, unique)]
        if len(labels_k) == len(set(labels_k)):
            best_k = k
            break

    out: Dict[str, str] = {}
    for lab, parts in zip(unique, parts_per):
        if not parts:
            raw = lab.replace("\\", "/")
            out[lab] = raw[-max_chars:] if len(raw) > max_chars else raw
            continue
        joined = joined_tail(parts, best_k)
        segs = [x for x in joined.split("/") if x]
        out[lab] = _fit_tail_path_to_width(segs, max_chars)
    return out


def compact_run_display_names(
    rel_labels: Sequence[str],
    *,
    max_chars: int = 52,
) -> Dict[str, str]:
    """Map each ``rel_label`` to a short UI string.

    Drops the longest common prefix across the set (path-style runs), prefixes the
    remainder with ``…``, then truncates from the left if still over ``max_chars``.
    Colliding displays after truncation are disambiguated with the path leaf and
    optional numeric suffix.
    """
    unique = list(dict.fromkeys(rel_labels))
    if not unique:
        return {}

    def trunc(s: str, mc: int = max_chars) -> str:
        return _truncate_run_display(s, mc)

    if len(unique) == 1:
        return {unique[0]: trunc(unique[0])}

    pfx = longest_common_prefix(unique)
    built: Dict[str, str] = {}
    for lab in unique:
        if pfx and lab.startswith(pfx):
            tail = lab[len(pfx) :]
            disp = f"…{tail}" if tail else trunc(lab)
        else:
            disp = lab
        built[lab] = trunc(disp)

    groups: Dict[str, List[str]] = {}
    for lab in unique:
        groups.setdefault(built[lab], []).append(lab)
    for labs in groups.values():
        if len(labs) <= 1:
            continue
        for i, lab in enumerate(labs):
            leaf = lab.split("/")[-1] if "/" in lab else lab
            suffix = f" ({i + 1})" if len(labs) > 1 else ""
            built[lab] = trunc(f"…{leaf}{suffix}", max_chars + 8)

    return built


def invert_compact_display_map(short_by_full: Dict[str, str]) -> Dict[str, str]:
    """Map short UI strings back to full ``rel_label`` (for native ``title`` tooltips)."""
    return {short: full for full, short in short_by_full.items()}


# Coarse run "types" for run_compare sidebar (path/name convention, not schema validation).
RUN_TYPE_PARAGRAPH = "paragraph"
RUN_TYPE_BULLETS = "bullets"
RUN_TYPE_OTHER = "other"
RUN_TYPE_ORDER: Tuple[str, ...] = (
    RUN_TYPE_PARAGRAPH,
    RUN_TYPE_BULLETS,
    RUN_TYPE_OTHER,
)
RUN_TYPE_LABELS = {
    RUN_TYPE_PARAGRAPH: "Paragraph",
    RUN_TYPE_BULLETS: "Bullets",
    RUN_TYPE_OTHER: "Other",
}


def infer_run_type_bucket(rel_label: str) -> str:
    """Bucket runs by type for apples-to-apples filtering (naming heuristic under ``data/eval``).

    Typical repo layout uses ``*_paragraph_*`` and ``*_bullets_*`` in directory names.
    """
    s = rel_label.lower()
    if "bullets" in s:
        return RUN_TYPE_BULLETS
    if "paragraph" in s:
        return RUN_TYPE_PARAGRAPH
    return RUN_TYPE_OTHER


# Pipeline stage order aligned with ``scripts/eval/freeze_profile.py`` (RFC-064).
PROFILE_STAGE_ORDER: Tuple[str, ...] = (
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
)

_PROFILE_SKIP_FILES = frozenset({"regression_rules.yaml"})


@dataclass(frozen=True)
class ProfileEntry:
    """One frozen performance profile YAML under ``data/profiles/`` (RFC-064)."""

    release: str
    date: str
    dataset_id: str
    hostname: str
    path: Path
    stages: Dict[str, Dict[str, float]]
    totals: Dict[str, Any]
    environment: Dict[str, Any]
    episodes_processed: int
    sort_ts: float
    # RFC-065 companions next to ``<stem>.yaml`` (optional; see PERFORMANCE_PROFILE_GUIDE.md).
    monitor_log_path: Optional[Path] = None
    monitor_trace_lines: Optional[int] = None
    monitor_trace_bytes: Optional[int] = None
    rfc065_monitor: Optional[Dict[str, Any]] = None


def _load_profile_rfc065_companion(
    yaml_path: Path,
) -> Tuple[Optional[Path], Optional[int], Optional[int], Optional[Dict[str, Any]]]:
    """Load sibling ``.stage_truth.json`` / ``.monitor.log`` metadata for a profile YAML."""
    parent = yaml_path.parent
    stem = yaml_path.stem
    log_path = parent / f"{stem}.monitor.log"
    st_path = parent / f"{stem}.stage_truth.json"
    meta: Optional[Dict[str, Any]] = None
    if st_path.is_file():
        try:
            data = json.loads(st_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            data = None
        if isinstance(data, dict):
            raw = data.get("rfc065_monitor")
            if isinstance(raw, dict):
                meta = dict(raw)
    lines: Optional[int] = None
    nbytes: Optional[int] = None
    resolved: Optional[Path] = None
    if log_path.is_file():
        resolved = log_path.resolve()
        try:
            nbytes = int(log_path.stat().st_size)
        except OSError:
            nbytes = None
        if meta is not None:
            if isinstance(meta.get("lines"), int):
                lines = int(meta["lines"])
            if isinstance(meta.get("bytes"), int):
                nbytes = int(meta["bytes"])
    elif meta is not None and meta.get("enabled"):
        if isinstance(meta.get("lines"), int):
            lines = int(meta["lines"])
        if isinstance(meta.get("bytes"), int):
            nbytes = int(meta["bytes"])
    return (resolved, lines, nbytes, meta)


def profile_has_rfc065_trace(entry: ProfileEntry) -> bool:
    """True if a monitor companion log or ``stage_truth`` RFC-065 block is present."""
    if entry.monitor_log_path is not None:
        return True
    m = entry.rfc065_monitor
    return bool(m and m.get("enabled"))


@dataclass(frozen=True)
class JoinedRelease:
    """Quality eval run and/or frozen profile for the same release join key (RFC-066)."""

    release: str
    eval_entry: Optional[RunEntry]
    profile_entry: Optional[ProfileEntry]


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


def _parse_profile_date_ts(date_str: str) -> float:
    """Parse profile ``date`` field for sort order (RFC-064 ISO timestamps)."""
    s = (date_str or "").strip()
    if not s:
        return 0.0
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def discover_profiles(profiles_root: Optional[Path] = None) -> List[ProfileEntry]:
    """Load all ``*.yaml`` profiles under ``data/profiles`` (excluding rules files).

    Each entry may include RFC-065 companion metadata from sibling ``<stem>.monitor.log`` and
    ``<stem>.stage_truth.json`` (``rfc065_monitor``), used by the Run Compare Performance page.

    Args:
        profiles_root: Base directory (default: repo ``data/profiles``).

    Returns:
        List of ``ProfileEntry``, sorted by ``sort_ts`` then ``release``.
    """
    root = profiles_root or Path("data/profiles")
    if not root.is_absolute():
        root = repo_root_from_here() / root
    if not root.is_dir():
        return []
    out: List[ProfileEntry] = []
    for path in sorted(root.glob("*.yaml")):
        name = path.name.lower()
        if name.startswith("."):
            continue
        if path.name in _PROFILE_SKIP_FILES:
            continue
        try:
            out.append(load_profile(path))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("Skipping profile %s: %s", path, e)
        except Exception as e:
            if yaml is not None and isinstance(e, yaml.YAMLError):
                logger.warning("Skipping profile %s: %s", path, e)
                continue
            raise
    out.sort(key=lambda p: (p.sort_ts, p.release))
    return out


def load_profile(path: Path) -> ProfileEntry:
    """Parse a frozen profile YAML into ``ProfileEntry``."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load profiles. Install project dependencies.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    release = raw.get("release")
    if not isinstance(release, str) or not release.strip():
        raise ValueError(f"Missing release in {path}")
    release = release.strip()
    date_s = str(raw.get("date") or "")
    dataset_id = str(raw.get("dataset_id") or "")
    env = raw.get("environment")
    if not isinstance(env, dict):
        env = {}
    hostname = str(env.get("hostname") or "")
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, dict):
        stages_raw = {}
    stages: Dict[str, Dict[str, float]] = {}
    for name, blob in stages_raw.items():
        if not isinstance(name, str) or not isinstance(blob, dict):
            continue
        entry: Dict[str, float] = {}
        for key in ("wall_time_s", "peak_rss_mb", "avg_cpu_pct"):
            v = blob.get(key)
            if isinstance(v, (int, float)):
                entry[key] = float(v)
        if entry:
            stages[name] = entry
    totals = raw.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    ep = raw.get("episodes_processed")
    episodes_processed = int(ep) if isinstance(ep, int) else 0
    sort_ts = _parse_profile_date_ts(date_s)
    mlog, mlines, mbytes, rfc_m = _load_profile_rfc065_companion(path)
    return ProfileEntry(
        release=release,
        date=date_s,
        dataset_id=dataset_id,
        hostname=hostname,
        path=path.resolve(),
        stages=stages,
        totals=totals,
        environment=dict(env),
        episodes_processed=episodes_processed,
        sort_ts=sort_ts,
        monitor_log_path=mlog,
        monitor_trace_lines=mlines,
        monitor_trace_bytes=mbytes,
        rfc065_monitor=rfc_m,
    )


def release_key_from_run_entry(entry: RunEntry) -> str:
    """Join key for an eval run: ``fingerprint`` release field or ``run_id``."""
    fp = entry.path / "fingerprint.json"
    if fp.is_file():
        try:
            data = load_json(fp)
            rc = data.get("run_context")
            if isinstance(rc, dict):
                rel = rc.get("release")
                if isinstance(rel, str) and rel.strip():
                    return rel.strip()
            rel2 = data.get("release")
            if isinstance(rel2, str) and rel2.strip():
                return rel2.strip()
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return entry.run_id


def join_releases(
    runs: Sequence[RunEntry],
    profiles: Sequence[ProfileEntry],
) -> Tuple[List[JoinedRelease], List[str]]:
    """Full outer join eval runs and profiles by release key."""
    warnings: List[str] = []
    profile_by_rel: Dict[str, ProfileEntry] = {}
    for p in sorted(profiles, key=lambda x: x.path.as_posix()):
        if p.release in profile_by_rel:
            warnings.append(
                f"Duplicate frozen profile for release {p.release!r}; using {p.path.name}"
            )
        profile_by_rel[p.release] = p

    eval_by_rel: Dict[str, RunEntry] = {}
    for e in runs:
        k = release_key_from_run_entry(e)
        if k in eval_by_rel:
            warnings.append(
                f"Duplicate eval run for join key {k!r} "
                f"({eval_by_rel[k].rel_label} vs {e.rel_label}); keeping first"
            )
            continue
        eval_by_rel[k] = e

    all_keys = sorted(set(profile_by_rel) | set(eval_by_rel))
    joined = [
        JoinedRelease(
            release=r,
            eval_entry=eval_by_rel.get(r),
            profile_entry=profile_by_rel.get(r),
        )
        for r in all_keys
    ]
    return joined, warnings


def filter_joined_releases(
    joined: Sequence[JoinedRelease],
    *,
    hostnames: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
) -> List[JoinedRelease]:
    """Keep rows whose profile matches hostname/dataset filters (if filters non-empty)."""
    out = list(joined)
    if hostnames:
        hs = set(hostnames)
        out = [j for j in out if j.profile_entry is not None and j.profile_entry.hostname in hs]
    if datasets:
        ds = set(datasets)
        out = [j for j in out if j.profile_entry is not None and j.profile_entry.dataset_id in ds]
    return out


def profile_metric_delta_good(metric: str, delta: float) -> Optional[bool]:
    """Whether a resource delta is good (lower CPU/RSS/wall is better)."""
    lower_better = {
        "wall_time_s",
        "peak_rss_mb",
        "avg_cpu_pct",
        "total_wall_time_s",
        "total_peak_rss_mb",
        "avg_wall_per_episode_s",
    }
    if metric not in lower_better:
        return None
    if delta == 0:
        return True
    return delta < 0


def profile_stage_delta_rows(
    baseline: ProfileEntry,
    candidates: Sequence[ProfileEntry],
) -> List[Dict[str, Any]]:
    """Rows for the performance delta table (baseline vs each candidate)."""
    rows: List[Dict[str, Any]] = []
    stage_names = sorted(
        set(baseline.stages.keys()) | {k for c in candidates for k in c.stages.keys()}
    )

    def _totals_row(prof: ProfileEntry) -> Dict[str, float]:
        t = prof.totals
        return {
            "total_wall_time_s": float(t.get("wall_time_s") or 0),
            "total_peak_rss_mb": float(t.get("peak_rss_mb") or 0),
            "avg_wall_per_episode_s": float(t.get("avg_wall_time_per_episode_s") or 0),
        }

    base_tot = _totals_row(baseline)
    tot_keys = [
        ("total_wall_time_s", "Total wall time (s)"),
        ("total_peak_rss_mb", "Total peak RSS (MB)"),
        ("avg_wall_per_episode_s", "Avg wall / episode (s)"),
    ]
    for tkey, tlabel in tot_keys:
        bv = base_tot.get(tkey) or 0.0
        for cand in candidates:
            cv = _totals_row(cand).get(tkey) or 0.0
            delta = cv - bv
            pct = None if bv == 0 else round((delta / bv) * 100.0, 2)
            rows.append(
                {
                    "stage": "(totals)",
                    "metric": tlabel,
                    "metric_key": tkey,
                    "baseline": bv,
                    "candidate_release": cand.release,
                    "value": cv,
                    "delta": delta,
                    "delta_pct": pct,
                    "good": profile_metric_delta_good(tkey, delta),
                }
            )

    for stage in stage_names:
        for mkey, mlabel in (
            ("wall_time_s", "Wall time (s)"),
            ("peak_rss_mb", "Peak RSS (MB)"),
            ("avg_cpu_pct", "Avg CPU %"),
        ):
            bv = baseline.stages.get(stage, {}).get(mkey)
            for cand in candidates:
                cv = cand.stages.get(stage, {}).get(mkey)
                if bv is None and cv is None:
                    continue
                bvf = float(bv) if bv is not None else None
                cvf = float(cv) if cv is not None else None
                if bvf is None or cvf is None:
                    continue
                delta = cvf - bvf
                pct = None if bvf == 0 else round((delta / bvf) * 100.0, 2)
                rows.append(
                    {
                        "stage": stage,
                        "metric": mlabel,
                        "metric_key": mkey,
                        "baseline": bvf,
                        "candidate_release": cand.release,
                        "value": cvf,
                        "delta": delta,
                        "delta_pct": pct,
                        "good": profile_metric_delta_good(mkey, delta),
                    }
                )
    return rows


def profile_trend_long_rows(profiles: Sequence[ProfileEntry]) -> List[Dict[str, Any]]:
    """Long-form rows for wall-time and RSS trend charts."""
    ordered = sorted(profiles, key=lambda p: (p.sort_ts, p.release))
    rows: List[Dict[str, Any]] = []
    for p in ordered:
        for stage in PROFILE_STAGE_ORDER:
            st = p.stages.get(stage)
            if not st:
                continue
            rows.append(
                {
                    "release": p.release,
                    "sort_ts": p.sort_ts,
                    "date_label": p.date or p.release,
                    "stage": stage,
                    "wall_time_s": st.get("wall_time_s"),
                    "peak_rss_mb": st.get("peak_rss_mb"),
                    "avg_cpu_pct": st.get("avg_cpu_pct"),
                }
            )
    return rows


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
