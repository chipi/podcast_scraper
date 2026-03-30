"""Load external USD pricing assumptions for LLM cost estimates (optional YAML).

Edit ``config/pricing_assumptions.yaml`` or set ``pricing_assumptions_file`` /
``PRICING_ASSUMPTIONS_FILE`` to change rates without code changes. When unset or the file is
missing, estimates use built-in provider constants only.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_cache_resolved: Optional[str] = None
_cache_mtime: Optional[float] = None
_cache_payload: Optional[Dict[str, Any]] = None

_RATE_KEYS = (
    "cost_per_minute",
    "cost_per_second",
    "input_cost_per_1m_tokens",
    "output_cost_per_1m_tokens",
    "cache_hit_input_cost_per_1m_tokens",
    "cost_per_hour",
)


def clear_pricing_assumptions_cache() -> None:
    """Reset loader cache (for tests)."""
    global _cache_resolved, _cache_mtime, _cache_payload
    _cache_resolved = None
    _cache_mtime = None
    _cache_payload = None


def resolve_assumptions_path(configured: str, *, cwd: Optional[Path] = None) -> Optional[Path]:
    """Resolve ``configured`` to an existing file path, or None."""
    raw = (configured or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.is_file() else None
    start = cwd or Path.cwd()
    candidate = start / path
    if candidate.is_file():
        return candidate
    for base in [start, *start.parents]:
        c = base / path
        if c.is_file():
            return c
    return None


def _coerce_rates(entry: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in _RATE_KEYS:
        if key in entry and entry[key] is not None:
            out[key] = float(entry[key])
    return out


def _match_model_section(section: Any, model: str) -> Optional[Dict[str, float]]:
    if not isinstance(section, dict) or not model:
        return None
    ml = model.lower().strip()
    if not ml:
        return None
    # Exact (case-insensitive) key match
    for k, v in section.items():
        if str(k).lower() == "default":
            continue
        if str(k).lower() == ml and isinstance(v, dict):
            rates = _coerce_rates(v)
            return rates or None
    # Longest substring match: YAML key appears in model id (e.g. gpt-4o-mini before gpt-4o)
    best: Optional[Dict[str, float]] = None
    best_len = -1
    for k, v in section.items():
        if str(k).lower() == "default" or not isinstance(v, dict):
            continue
        kl = str(k).lower()
        if kl in ml and len(kl) > best_len:
            cand = _coerce_rates(v)
            if cand:
                best = cand
                best_len = len(kl)
    if best is not None:
        return best
    default = section.get("default")
    if isinstance(default, dict):
        rates = _coerce_rates(default)
        return rates or None
    return None


def lookup_external_pricing(
    payload: Dict[str, Any], provider_type: str, capability: str, model: str
) -> Optional[Dict[str, float]]:
    """Return rate dict from loaded YAML, or None."""
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return None
    prov = providers.get(provider_type)
    if not isinstance(prov, dict):
        return None
    if capability == "transcription":
        section = prov.get("transcription")
        return _match_model_section(section, model)
    if capability in ("speaker_detection", "summarization"):
        section = prov.get("text")
        return _match_model_section(section, model)
    return None


def load_pricing_assumptions_payload(path: Path) -> Dict[str, Any]:
    """Parse YAML file; raises if invalid or PyYAML missing."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load pricing assumptions")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("pricing assumptions root must be a mapping")
    return data


def get_loaded_table(
    configured_path: str, *, cwd: Optional[Path] = None
) -> Tuple[Optional[dict], Optional[Path]]:
    """Load (and cache) pricing table from disk.

    Returns:
        (payload dict with 'providers' etc., resolved path) or (None, None) if disabled/missing.
    """
    global _cache_resolved, _cache_mtime, _cache_payload
    resolved = resolve_assumptions_path(configured_path, cwd=cwd)
    if resolved is None:
        return None, None
    key = str(resolved.resolve())
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        return None, None
    if _cache_resolved == key and _cache_mtime == mtime and _cache_payload is not None:
        return _cache_payload, resolved
    payload = load_pricing_assumptions_payload(resolved)
    _cache_resolved = key
    _cache_mtime = mtime
    _cache_payload = payload
    logger.debug("Loaded pricing assumptions from %s", resolved)
    return payload, resolved


def _parse_iso_date(value: Any) -> Optional[_dt.date]:
    if value is None or value == "":
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def check_staleness(
    payload: Dict[str, Any],
    *,
    today: Optional[_dt.date] = None,
) -> Tuple[bool, List[str]]:
    """Return (is_stale, human-readable messages)."""
    day = today or _dt.date.today()
    messages: List[str] = []
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        return False, messages
    last = _parse_iso_date(meta.get("last_reviewed"))
    threshold_days = meta.get("stale_review_after_days")
    try:
        td = int(threshold_days) if threshold_days is not None else None
    except (TypeError, ValueError):
        td = None
    if last is not None and td is not None and td > 0:
        age = (day - last).days
        if age > td:
            messages.append(
                f"last_reviewed ({last}) is older than stale_review_after_days ({td}); "
                f"verify vendor pricing and update the YAML."
            )
    return bool(messages), messages


def format_status_report(
    configured_path: str,
    *,
    cwd: Optional[Path] = None,
    today: Optional[_dt.date] = None,
) -> str:
    """Human-readable report for CLI."""
    lines: List[str] = []
    lines.append("Pricing assumptions")
    lines.append("=" * 40)
    lines.append(f"Configured path: {configured_path!r}")
    payload, resolved = get_loaded_table(configured_path, cwd=cwd)
    if resolved is None:
        lines.append("Resolved file: (not found — using built-in code rates only)")
        return "\n".join(lines) + "\n"
    lines.append(f"Resolved file: {resolved}")
    if payload:
        ver = payload.get("schema_version", "?")
        lines.append(f"schema_version: {ver}")
        meta = payload.get("metadata")
        if isinstance(meta, dict):
            for key in (
                "last_reviewed",
                "pricing_effective_date",
                "stale_review_after_days",
            ):
                if key in meta:
                    lines.append(f"{key}: {meta[key]}")
            urls = meta.get("source_urls")
            if isinstance(urls, dict):
                lines.append("source_urls:")
                for k, u in sorted(urls.items()):
                    lines.append(f"  {k}: {u}")
        stale, msgs = check_staleness(payload, today=today)
        if msgs:
            lines.append("")
            lines.append("Staleness:" if stale else "Note:")
            for m in msgs:
                lines.append(f"  - {m}")
    return "\n".join(lines) + "\n"
