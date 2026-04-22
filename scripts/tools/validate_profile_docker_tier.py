#!/usr/bin/env python3
"""Validate packaged profiles declare a minimum Docker pipeline tier (RFC-079 matrix).

Exit 0 when all profiles match expectations; non-zero on mismatch.
Usage:
  python scripts/tools/validate_profile_docker_tier.py
  EXPECTED_TIER=llm python scripts/tools/validate_profile_docker_tier.py --only cloud_thin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "config" / "profiles"

# Top-level packaged presets (stem -> minimum INSTALL_EXTRAS tier for Docker pipeline image).
EXPECTED_TIER: dict[str, str] = {
    "airgapped": "ml",
    "cloud_balanced": "ml",
    "cloud_quality": "ml",
    "cloud_thin": "llm",
    "dev": "ml",
    "local": "ml",
}


def _tier_rank(t: str) -> int:
    return {"": 0, "llm": 1, "ml": 2}.get(t, 0)


def _derive_tier_from_yaml(data: dict) -> str:
    """Heuristic minimum tier from profile fields (kept in sync with RFC-079 matrix)."""
    if str(data.get("speaker_detector_provider", "")).lower() == "spacy":
        return "ml"
    if data.get("vector_search") is True:
        return "ml"
    # Local in-process Whisper (not OpenAI whisper-1 API).
    if str(data.get("transcription_provider", "")).lower() == "whisper":
        return "ml"
    return "llm"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Check a single profile stem (e.g. cloud_thin)")
    args = p.parse_args()
    only = (args.only or "").strip()

    failures: list[str] = []
    env_expected = __import__("os").environ.get("EXPECTED_TIER", "").strip().lower()

    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        stem = path.stem
        if stem.endswith(".example") or path.name.endswith(".example.yaml"):
            continue
        if only and stem != only:
            continue
        if stem not in EXPECTED_TIER:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            failures.append(f"{path.name}: not a mapping")
            continue
        derived = _derive_tier_from_yaml(data)
        expected = EXPECTED_TIER[stem]
        if env_expected and _tier_rank(derived) < _tier_rank(env_expected):
            failures.append(f"{path.name}: derived={derived} below EXPECTED_TIER={env_expected}")
            continue
        if _tier_rank(derived) < _tier_rank(expected):
            failures.append(
                f"{path.name}: derived minimum tier={derived} expected>={expected} "
                f"(speaker={data.get('speaker_detector_provider')!r} "
                f"vector_search={data.get('vector_search')!r})"
            )

    if failures:
        print("validate_profile_docker_tier: FAILED", file=sys.stderr)
        for line in failures:
            print(line, file=sys.stderr)
        return 1
    print("validate_profile_docker_tier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
