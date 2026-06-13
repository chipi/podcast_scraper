"""Frozen prod-validation tier harness — #933.

Runs lightweight pipeline-output checks against the 15-episode
``prod_validation_v1`` subset. Emits a summary report flagging any
quality regressions vs. the dataset's frozen expectations.

The harness is intentionally **read-only** — it consumes the prepared
transcripts under ``data/eval/datasets/prod_validation_v1/episodes/``
plus any runtime artifacts the caller supplies (e.g. GI outputs from a
recent pipeline run). It never mutates the dataset.

## Checks performed

Each episode is exercised against a configurable subset of these
quick-signal checks:

- ``cleaning``: apply the current default cleaning profile and report
  removed-char delta + post-clean sponsor-pattern residual hits.
- ``ner``: run spaCy NER on the cleaned text and report PERSON entity
  count + a sample of detected hosts.
- ``commercial``: count sponsor-pattern hits + boundary block-end hits.

More expensive checks (full GI grounding, KG entity canon) are out of
scope for the harness itself — they need to be run via the production
pipeline and the outputs passed in.

## Usage

    PYTHONPATH=. .venv/bin/python scripts/eval/validate_prod_set.py \\
        --dataset data/eval/datasets/prod_validation_v1 \\
        --check cleaning ner commercial \\
        --output data/eval/runs/prod_validation_v1_baseline

The ``--check`` flag is repeatable; defaults to all three.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_VALID_CHECKS = ("cleaning", "ner", "commercial")


def _load_manifest(dataset: Path) -> Dict[str, Any]:
    return yaml.safe_load((dataset / "manifest.yaml").read_text(encoding="utf-8")) or {}


def _episode_path(dataset: Path, ep_id: str) -> Path:
    return dataset / "episodes" / f"{ep_id}.txt"


# ---------------------------------------------------------------- checks


def _check_cleaning(text: str) -> Dict[str, Any]:
    from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS
    from podcast_scraper.preprocessing import profiles as profile_mod

    cleaned = profile_mod.apply_profile(text, profile_mod.DEFAULT_PROFILE)
    removed_chars = len(text) - len(cleaned)
    residual_hits = sum(sum(1 for _ in p.pattern.finditer(cleaned)) for p in SPONSOR_PATTERNS)
    return {
        "profile": profile_mod.DEFAULT_PROFILE,
        "removed_chars": removed_chars,
        "removed_pct": round(100.0 * removed_chars / max(1, len(text)), 2),
        "residual_sponsor_hits": residual_hits,
    }


def _check_commercial(text: str) -> Dict[str, Any]:
    from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS

    content_patterns = [p for p in SPONSOR_PATTERNS if p.boundary_hint != "block_end"]
    content_hits = sum(sum(1 for _ in p.pattern.finditer(text)) for p in content_patterns)
    boundary_hits = sum(
        sum(1 for _ in p.pattern.finditer(text))
        for p in SPONSOR_PATTERNS
        if p.boundary_hint == "block_end"
    )
    return {
        "content_pattern_hits": content_hits,
        "boundary_block_end_hits": boundary_hits,
        "pattern_count": len(SPONSOR_PATTERNS),
    }


def _check_ner(text: str, max_persons: int = 10) -> Dict[str, Any]:
    try:
        import spacy
    except ImportError:
        return {"error": "spacy not installed"}

    from podcast_scraper.config_constants import PROD_DEFAULT_NER_MODEL

    try:
        nlp = spacy.load(PROD_DEFAULT_NER_MODEL)
    except (OSError, ValueError) as exc:
        return {
            "error": f"could not load {PROD_DEFAULT_NER_MODEL}: {exc}",
            "ner_model_attempted": PROD_DEFAULT_NER_MODEL,
        }

    # spaCy has a default ~1M char limit; truncate to be safe for long episodes.
    doc = nlp(text[:1_000_000])
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    distinct = sorted(set(persons))
    return {
        "ner_model": PROD_DEFAULT_NER_MODEL,
        "person_mentions": len(persons),
        "distinct_persons": len(distinct),
        "sample_persons": distinct[:max_persons],
    }


# ---------------------------------------------------------------- main


def _run_episode(
    ep_id: str,
    text_path: Path,
    checks: List[str],
) -> Dict[str, Any]:
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    result: Dict[str, Any] = {
        "episode_id": ep_id,
        "source": str(text_path),
        "char_count": len(text),
        "word_count": len(text.split()),
        "checks": {},
    }
    if "cleaning" in checks:
        t0 = time.time()
        result["checks"]["cleaning"] = _check_cleaning(text)
        result["checks"]["cleaning"]["elapsed_s"] = round(time.time() - t0, 3)
    if "commercial" in checks:
        t0 = time.time()
        result["checks"]["commercial"] = _check_commercial(text)
        result["checks"]["commercial"]["elapsed_s"] = round(time.time() - t0, 3)
    if "ner" in checks:
        t0 = time.time()
        result["checks"]["ner"] = _check_ner(text)
        result["checks"]["ner"]["elapsed_s"] = round(time.time() - t0, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--check",
        choices=_VALID_CHECKS,
        action="append",
        help="Repeatable; defaults to all checks.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="Limit to specific episode IDs (e.g. ep_0001 ep_0050).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"ERROR: dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2

    manifest = _load_manifest(args.dataset)
    episode_specs: List[Dict[str, Any]] = list(manifest.get("episodes", []))
    if args.episodes:
        wanted = set(args.episodes)
        episode_specs = [e for e in episode_specs if e.get("id") in wanted]
    checks = args.check or list(_VALID_CHECKS)

    args.output.mkdir(parents=True, exist_ok=True)
    per_episode: List[Dict[str, Any]] = []
    for spec in episode_specs:
        ep_id = spec["id"]
        text_path = _episode_path(args.dataset, ep_id)
        if not text_path.is_file():
            print(f"  SKIP {ep_id}: text missing at {text_path}", file=sys.stderr)
            continue
        print(f"  {ep_id}: running {','.join(checks)} …", file=sys.stderr)
        row = _run_episode(ep_id, text_path, checks)
        row["tags"] = spec.get("tags", [])
        per_episode.append(row)

    # Aggregate
    def _avg(field_path: List[str]) -> Optional[float]:
        vals = []
        for r in per_episode:
            cur: Any = r
            for k in field_path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return sum(vals) / len(vals) if vals else None

    summary = {
        "dataset": str(args.dataset),
        "freeze_status": manifest.get("freeze_status"),
        "frozen_at": str(manifest.get("frozen_at")),
        "episodes_run": len(per_episode),
        "checks_requested": checks,
    }
    if "cleaning" in checks:
        summary["mean_cleaning_removed_pct"] = _avg(["checks", "cleaning", "removed_pct"])
        summary["mean_cleaning_residual_hits"] = _avg(
            ["checks", "cleaning", "residual_sponsor_hits"]
        )
    if "commercial" in checks:
        summary["mean_content_pattern_hits"] = _avg(
            ["checks", "commercial", "content_pattern_hits"]
        )
        summary["mean_boundary_block_end_hits"] = _avg(
            ["checks", "commercial", "boundary_block_end_hits"]
        )
    if "ner" in checks:
        summary["mean_distinct_persons"] = _avg(["checks", "ner", "distinct_persons"])
        summary["mean_person_mentions"] = _avg(["checks", "ner", "person_mentions"])

    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "prod_validation_v1_metrics",
                "summary": summary,
                "rows": per_episode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== prod_validation_v1 summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>40s}: {v:.2f}")
        else:
            print(f"  {k:>40s}: {v}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
