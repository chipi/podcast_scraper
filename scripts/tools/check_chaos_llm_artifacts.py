#!/usr/bin/env python3
"""Acceptance check for ``make preprod-chaos-llm-down``.

Reads the artifacts a chaos run produced and answers the only question that matters: when the LLM
endpoint was dead, did the pipeline degrade HONESTLY or did it fabricate?

Exit code is not enough, and neither is "the run completed". The production incident this guards
against completed successfully, wrote a full corpus, and recorded ``kg_failures=0`` — while every
Topic node was a summary bullet truncated into a sentence fragment and every episode had zero
insights. A green exit code is precisely what the bug looked like.

Checks, each naming what a failure means:

1. ``llm_kg_calls > 0`` OR ``kg_failures > 0`` — the extraction either worked or was RECORDED as
   failing. Both zero is the signature of the swallowed transport error.
2. No Topic label is a prefix of a summary bullet from the same episode — bullets-as-topics.
3. ``extraction.model_version != "topic_labels"`` — that string is minted only on the bullet
   substitution path.
4. Topic labels look like noun phrases, not truncated propositions.

Usage:  python scripts/tools/check_chaos_llm_artifacts.py <corpus-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_MAX_TOPIC_WORDS = 8


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: check_chaos_llm_artifacts.py <corpus-dir>", file=sys.stderr)
        return 2
    root = Path(args[0])
    if not root.is_dir():
        print(f"FAIL: not a directory: {root}", file=sys.stderr)
        return 2

    failures: list[str] = []

    # 1. metrics — a total failure must be RECORDED, not silent.
    for metrics_path in root.rglob("metrics.json"):
        m = _load(metrics_path)
        kg_calls = int(m.get("llm_kg_calls") or 0)
        kg_failures = int(m.get("kg_failures") or 0)
        if kg_calls == 0 and kg_failures == 0:
            failures.append(
                f"{metrics_path}: llm_kg_calls=0 AND kg_failures=0 — the extraction neither ran "
                "nor recorded a failure. This is the swallowed-transport-error signature."
            )
        gi_total = int(m.get("gi_insights_total") or 0)
        gi_failures = int(m.get("gi_failures") or 0)
        if gi_total == 0 and gi_failures == 0:
            failures.append(
                f"{metrics_path}: gi_insights_total=0 AND gi_failures=0 — same defect, GI stage."
            )

    # 2-4. artifacts — nothing may be fabricated from bullets.
    for kg_path in root.rglob("*.kg.json"):
        kg = _load(kg_path)
        provenance = str((kg.get("extraction") or {}).get("model_version") or "")
        if provenance == "topic_labels":
            failures.append(
                f"{kg_path}: extraction.model_version == 'topic_labels' — summary bullets were "
                "substituted as Topic nodes."
            )

        meta_path = kg_path.with_name(kg_path.name.replace(".kg.json", ".metadata.json"))
        bullets = [
            str(b).strip()
            for b in ((_load(meta_path).get("summary") or {}).get("bullets") or [])
            if str(b).strip()
        ]

        for node in kg.get("nodes") or []:
            if not isinstance(node, dict) or node.get("type") != "Topic":
                continue
            label = str((node.get("properties") or {}).get("label") or "").strip()
            if not label:
                continue
            if any(b.startswith(label) and len(b) > len(label) for b in bullets):
                failures.append(
                    f"{kg_path}: Topic {label!r} is a truncated summary bullet, not a topic."
                )
            if len(label.split()) > _MAX_TOPIC_WORDS:
                failures.append(
                    f"{kg_path}: Topic {label!r} is {len(label.split())} words — a proposition, "
                    "not a noun phrase."
                )

    if failures:
        print("CHAOS ACCEPTANCE FAILED:\n", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("chaos acceptance PASSED: the LLM outage degraded honestly (no fabricated topics).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
