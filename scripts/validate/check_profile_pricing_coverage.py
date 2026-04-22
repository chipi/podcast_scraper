#!/usr/bin/env python3
"""#651 CI guard: every model referenced by a profile must have a YAML rate row.

Walks ``config/profiles/*.yaml`` (and the ``freeze/*.yaml`` subset), extracts
all ``<provider>_<capability>_model`` field values, and asserts each has a
matching row under ``providers.<provider>.<text|transcription>`` in
``config/pricing_assumptions.yaml`` (or at least a ``default`` row under that
provider+capability section).

Fails PR merge if a profile adds a model without adding its rate.

Exit codes
----------
- 0 — all profile models have YAML coverage.
- 1 — one or more models missing a YAML row. Prints the gap list.
- 2 — script or input error (missing file, malformed YAML).

Usage
-----

    python scripts/validate/check_profile_pricing_coverage.py
    # or
    make check-pricing-assumptions     # includes this guard (once wired)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML required.", file=sys.stderr)
    raise SystemExit(2)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "config" / "profiles"
PRICING_YAML = REPO_ROOT / "config" / "pricing_assumptions.yaml"

# Providers that bill per-call (all others are local / free).
_BILLABLE_PROVIDERS: Tuple[str, ...] = (
    "openai",
    "gemini",
    "anthropic",
    "mistral",
    "deepseek",
    "grok",
)

# Field-name regex: <provider>_<capability>_model.
_FIELD_RE = re.compile(
    r"^(?P<provider>openai|gemini|anthropic|mistral|deepseek|grok|ollama)"
    r"_(?P<capability>transcription|speaker|summary|cleaning|insight|kg_extraction)"
    r"_model$"
)

# Map profile-field capability → YAML section.
#   transcription  → providers.<p>.transcription.<model>
#   speaker        → providers.<p>.text.<model>       (speaker_detection uses text rates)
#   summary        → providers.<p>.text.<model>
#   cleaning       → providers.<p>.text.<model>
#   insight        → providers.<p>.text.<model>
#   kg_extraction  → providers.<p>.text.<model>
_SECTION_FOR_CAPABILITY: Dict[str, str] = {
    "transcription": "transcription",
    "speaker": "text",
    "summary": "text",
    "cleaning": "text",
    "insight": "text",
    "kg_extraction": "text",
}


def _iter_profile_paths() -> List[Path]:
    """Collect all *.yaml files under config/profiles/ (recursively)."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p for p in PROFILES_DIR.rglob("*.yaml"))


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        print(f"ERROR: {path} is not a mapping.", file=sys.stderr)
        raise SystemExit(2)
    return data


def _extract_model_refs(profile_doc: Dict[str, object]) -> List[Tuple[str, str, str]]:
    """Return [(provider, capability_section, model), ...] for a profile doc.

    Only billable providers are returned; local/free stacks (spacy, whisper,
    ollama) are excluded.
    """
    out: List[Tuple[str, str, str]] = []
    for key, value in profile_doc.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        m = _FIELD_RE.match(key)
        if not m:
            continue
        provider = m.group("provider")
        if provider not in _BILLABLE_PROVIDERS:
            continue
        capability = m.group("capability")
        section = _SECTION_FOR_CAPABILITY.get(capability)
        if section is None:
            continue
        model = value.strip()
        if not model:
            continue
        out.append((provider, section, model))
    return out


def _pricing_has_model(pricing: Dict[str, object], provider: str, section: str, model: str) -> bool:
    providers = pricing.get("providers")
    if not isinstance(providers, dict):
        return False
    prov = providers.get(provider)
    if not isinstance(prov, dict):
        return False
    sect = prov.get(section)
    if not isinstance(sect, dict):
        return False
    model_lower = model.lower().strip()
    # Exact or prefix match, then default fallback.
    for key in sect.keys():
        if not isinstance(key, str):
            continue
        k_lower = key.lower()
        if k_lower == "default":
            continue
        if k_lower == model_lower or model_lower.startswith(k_lower):
            return True
    return "default" in sect


def _unique_sorted(items: Sequence[Tuple[str, str, str, Path]]) -> List[Tuple[str, str, str, Path]]:
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Tuple[str, str, str, Path]] = []
    for provider, section, model, path in items:
        key = (provider, section, model)
        if key in seen:
            continue
        seen.add(key)
        out.append((provider, section, model, path))
    return out


def main() -> int:
    if not PRICING_YAML.is_file():
        print(f"ERROR: {PRICING_YAML} not found.", file=sys.stderr)
        return 2
    pricing = _load_yaml(PRICING_YAML)

    profile_paths = _iter_profile_paths()
    if not profile_paths:
        print(f"No profiles found under {PROFILES_DIR}; nothing to check.")
        return 0

    refs: List[Tuple[str, str, str, Path]] = []
    for path in profile_paths:
        doc = _load_yaml(path)
        for provider, section, model in _extract_model_refs(doc):
            refs.append((provider, section, model, path))
    refs = _unique_sorted(refs)

    missing: List[Tuple[str, str, str, Path]] = [
        r for r in refs if not _pricing_has_model(pricing, r[0], r[1], r[2])
    ]
    print(
        f"Checked {len(refs)} unique (provider, section, model) refs across "
        f"{len(profile_paths)} profile files."
    )
    if missing:
        print()
        print(f"FAIL: {len(missing)} model(s) referenced by profiles have no YAML rate row:")
        for provider, section, model, path in missing:
            rel = path.relative_to(REPO_ROOT)
            print(f"  providers.{provider}.{section}.{model}  (referenced in {rel})")
        print()
        print("Add a row in config/pricing_assumptions.yaml, or fall back to a ")
        print("provider-level `default` row (input_cost_per_1m_tokens + output_cost_per_1m_tokens ")
        print("for text, cost_per_minute or cost_per_second for transcription).")
        return 1

    print("PASS: every profile-referenced model has a YAML rate row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
