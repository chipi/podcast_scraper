"""Held-out check for 0033's self-introduction rule: every one-word self-introduction the reader
takes from any voice of any served episode, read with the rule off and on.

Read-only. For each served episode, each voice's own text (its segments, in order) goes through
``extract_self_introduced_host`` as ``roster._self_intros_by_voice`` calls it (5,000 characters).
With ``--population`` it prints only the count of one-word captures with the rule off (the
population, measured before the outcome is predicted); without it, it lists every capture the rule
refuses, with the sentence, for a reader to judge. Records 0033
(``docs/wip/changes/0033_hand_removed_names_refused.md``, acceptance criterion 20).

Usage: ``python scripts/measure/self_intro_label_rule_heldout.py <corpus root> [--population]``
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from podcast_scraper.speaker_detectors import hosts
from podcast_scraper.upgrade.corpus_selection import select_served_artifacts


def _voice_texts(meta: Path, data: dict) -> Dict[str, str]:
    tr = Path(str((data.get("content") or {}).get("transcript_file_path") or "")).name
    if not tr.endswith(".txt"):
        return {}
    seg = meta.parent.parent / "transcripts" / tr.replace(".txt", ".segments.json")
    try:
        payload = json.loads(seg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments") or []
    parts: Dict[str, List[str]] = collections.defaultdict(list)
    for r in rows:
        if isinstance(r, dict) and r.get("speaker"):
            parts[str(r["speaker"])].append(str(r.get("text") or ""))
    return {v: " ".join(t) for v, t in parts.items()}


def _capture(text: str, language: Optional[str], rule: bool) -> Optional[str]:
    real = hosts._names_a_label
    if not rule:
        hosts._names_a_label = lambda head, match: False  # type: ignore[assignment]
    try:
        return hosts.extract_self_introduced_host(text, intro_chars=5000, language=language)
    finally:
        hosts._names_a_label = real  # type: ignore[assignment]


def main(root: Path, population: bool) -> None:
    """Print the population count, or every refusal with its context."""
    metas = select_served_artifacts(root, ".metadata.json")[0]
    one_word = 0
    refused = []
    by_lang: collections.Counter = collections.Counter()
    for meta in metas:
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        lang = (data.get("episode") or {}).get("language") or None
        for voice, text in sorted(_voice_texts(meta, data).items()):
            off = _capture(text, lang, rule=False)
            if not off or len(off.split()) != 1:
                continue
            one_word += 1
            by_lang[str(lang)] += 1
            if population:
                continue
            on = _capture(text, lang, rule=True)
            if on != off:
                at = text[:5000].find(off)
                refused.append((meta.name, voice, str(lang), off, on, text[max(0, at - 40) : at + 60]))
    print(f"served {len(metas)}; one-word captures with the rule off {one_word}; by language "
          f"{dict(by_lang)}")
    if not population:
        print(f"changed by the rule {len(refused)}")
        for row in refused:
            print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--population" in sys.argv[2:])
