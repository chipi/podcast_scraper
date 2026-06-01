#!/usr/bin/env python3
"""Eval the KG entity-canonicalization prompt change (#851 Slice 2).

Measures, on a real corpus, whether the current KG extraction prompt collapses
same-episode duplicate-spelling entities (e.g. "Burne Hobart" + "Byrne Hobart")
at the source. For each episode that *currently* has near-duplicate person/org
nodes in its committed ``*.kg.json``, this re-extracts entities from the episode
transcript with the live prompt and reports BEFORE vs AFTER duplicate counts.

This is a manual eval (live LLM calls + a local corpus), not a unit test.

Requirements:
  - A provider API key (reads ``.env`` if present, else the process env).
  - A corpus dir with ``*.kg.json`` artifacts + their transcripts.

Usage:
  python scripts/eval_entity_canonicalization.py \
      --corpus-dir .test_outputs/manual/my-manual-run-10 \
      --model gpt-4o-mini --limit 5
"""

from __future__ import annotations

import argparse
import difflib
import glob
import itertools
import json
import os
import sys
from pathlib import Path

# Allow running from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from podcast_scraper.builders.bridge_builder import strip_layer_prefixes  # noqa: E402
from podcast_scraper.kg.llm_extract import (  # noqa: E402
    build_kg_transcript_system_prompt,
    build_kg_user_prompt,
    parse_kg_graph_response,
    truncate_transcript_for_kg,
)

NEAR_DUP_RATIO = 0.85


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from .env into os.environ (values never printed)."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _entities(artifact: dict) -> list[tuple[str, str]]:
    """Return [(name, kind)] for person/org nodes in a kg artifact."""
    out = []
    for node in artifact.get("nodes", []):
        nid = strip_layer_prefixes(str(node.get("id", "")))
        prefix = nid.split(":", 1)[0]
        if prefix in ("person", "org"):
            props = node.get("properties", {}) or {}
            name = props.get("name") or props.get("label") or ""
            if name:
                out.append((name, prefix))
    return out


def _near_dup_pairs(entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs = []
    for (na, ka), (nb, kb) in itertools.combinations(entities, 2):
        if ka != kb or na.lower() == nb.lower():
            continue
        if difflib.SequenceMatcher(None, na.lower(), nb.lower()).ratio() >= NEAR_DUP_RATIO:
            pairs.append((na, nb))
    return pairs


def _transcript_index(corpus_dir: str) -> dict[str, str]:
    idx: dict[str, str] = {}
    for t in glob.glob(f"{corpus_dir}/**/transcripts/*.txt", recursive=True):
        if "reprocess-backup" in t:
            continue
        idx.setdefault(os.path.basename(t), t)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=5, help="max dup episodes to re-extract")
    ap.add_argument("--max-entities", type=int, default=15)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (add to .env or env).", file=sys.stderr)
        return 2

    tidx = _transcript_index(args.corpus_dir)

    # Find episodes whose committed KG artifact already has near-duplicate entities.
    targets: list[tuple[str, list[tuple[str, str]]]] = []
    for f in glob.glob(f"{args.corpus_dir}/**/*.kg.json", recursive=True):
        if "reprocess-backup" in f or "/.cache/" in f:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        pairs = _near_dup_pairs(_entities(data))
        if not pairs:
            continue
        tref = (data.get("extraction") or {}).get("transcript_ref") or ""
        tname = os.path.basename(tref)
        if tname in tidx:
            targets.append((tname, pairs))

    # De-dup episodes; cap to --limit.
    seen, uniq = set(), []
    for tname, pairs in targets:
        if tname in seen:
            continue
        seen.add(tname)
        uniq.append((tname, pairs))
    uniq = uniq[: args.limit]

    if not uniq:
        print("No dup episodes found in corpus — nothing to eval.")
        return 0

    from openai import OpenAI

    client = OpenAI()
    print(f"Re-extracting {len(uniq)} dup episode(s) with model={args.model}\n")

    before_total = after_total = 0
    for tname, before_pairs in uniq:
        txt = truncate_transcript_for_kg(open(tidx[tname], errors="ignore").read())
        system = build_kg_transcript_system_prompt(5, args.max_entities)
        user = build_kg_user_prompt(txt, "", 5, args.max_entities)  # default prompt version
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=2048,
        )
        parsed = parse_kg_graph_response(resp.choices[0].message.content) or {}
        fresh = [
            (e.get("name", ""), (e.get("entity_kind") or "person"))
            for e in parsed.get("entities", [])
            if isinstance(e, dict) and e.get("name")
        ]
        after_pairs = _near_dup_pairs(fresh)
        before_total += len(before_pairs)
        after_total += len(after_pairs)
        status = "PASS" if len(after_pairs) < len(before_pairs) or not after_pairs else "STILL-DUP"
        print(f"[{status}] {tname[:60]}")
        print(f"    before: {before_pairs}")
        print(f"    after : {after_pairs or 'no near-dup pairs'}\n")

    print(f"TOTAL near-dup pairs: before={before_total}  after={after_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
