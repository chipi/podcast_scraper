#!/usr/bin/env python3
"""Census for corpus migration 0043: every served voice whose stored name today's roster code
removes AND types ``commercial``. Read-only.

Each served episode (the corpus membership rule) is replayed through the current tree's
``resolve_speaker_roster`` on its own stored inputs (``roster_replay.py``), with the corpus's ad
signatures built in memory the way finalize builds them. A voice is listed when its stored segment
label is a name and the replay leaves it unnamed with ``voice_type == "commercial"``.

    python scripts/measure/m0043_census.py --corpus <root> --shard K/N --out census_K.jsonl

``--shard`` is 0-indexed (``0/8`` .. ``7/8``); concatenate the outputs. One JSON line per voice
(metadata relpath, episode id, voice, stored name, feed title) and a summary line on stdout. The
output holds names: keep it out of the public repo. 3,130 episodes in 8 shards: about 4 minutes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.upgrade.migrations.m0016_ad_reader_names_removed import (  # noqa: E402
    _load,
    _rows,
)
from podcast_scraper.utils.run_relative import run_relative  # noqa: E402

COMMERCIAL = "commercial"


def _roster_replay() -> Any:
    spec = importlib.util.spec_from_file_location(
        "roster_replay", REPO / "scripts" / "measure" / "roster_replay.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stored_names(meta: Path, content: Dict[str, Any]) -> Dict[str, str]:
    """``{voice: name}`` the episode's segments serve (a raw ``SPEAKER_NN`` label is no name)."""
    rel = run_relative(content.get("transcript_file_path"), meta.parent.parent)
    if not rel.endswith(".txt"):
        return {}
    out: Dict[str, str] = {}
    for r in _rows(_load(meta.parent.parent / rel.replace(".txt", ".segments.json"))):
        label = r.get("speaker_label")
        if r.get("speaker") and label and not str(label).upper().startswith("SPEAKER"):
            out.setdefault(str(r["speaker"]), str(label))
    return out


def commercial_unnamed(by_voice: Dict[str, Any], stored: Dict[str, str]) -> List[str]:
    """The stored-named voices the replay leaves unnamed and types commercial."""
    out = []
    for voice in stored:
        role = by_voice.get(voice)
        if role is not None and not role.named and getattr(role, "voice_type", None) == COMMERCIAL:
            out.append(voice)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--shard", required=True, help="K/N, 0-indexed")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    k, n = (int(x) for x in args.shard.split("/"))
    rr = _roster_replay()
    rr.index_siblings(args.corpus)
    new = rr.load_variant({})
    t0 = time.time()
    sig = rr._signatures(new.get("ad_signatures"), args.corpus)
    tally: Counter = Counter()
    with args.out.open("w", encoding="utf-8") as out:
        for ep in rr.corpus_episodes(args.corpus, (k, n)):
            meta = Path(ep["meta_path"])
            try:
                res = rr.replay(new["roster"], ep, sig)
            except Exception as exc:  # counted, never silent: a replay error is a census gap
                tally[f"error:{type(exc).__name__}"] += 1
                continue
            tally["episodes"] += 1
            doc = ep["meta"]
            stored = stored_names(meta, doc.get("content") or {})
            for voice in commercial_unnamed(res.by_voice, stored):
                tally["commercial_unnamed"] += 1
                row = {
                    "meta": str(meta.relative_to(args.corpus)),
                    "episode_id": (doc.get("episode") or {}).get("episode_id"),
                    "voice": voice,
                    "name": stored[voice],
                    "feed": (doc.get("feed") or {}).get("title"),
                }
                out.write(json.dumps(row) + "\n")
    print(json.dumps({"shard": k, "seconds": round(time.time() - t0), **tally}))
    return 1 if any(key.startswith("error:") for key in tally) else 0


if __name__ == "__main__":
    sys.exit(main())
