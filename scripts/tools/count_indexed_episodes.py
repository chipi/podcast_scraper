#!/usr/bin/env python3
"""Episodes actually present in a corpus's search index.

`index-two-tier` exits 0 whether it indexed 40 episodes or none — with
`search/episode_fingerprints.json` in place it treats a grown corpus as already covered and
writes nothing. So "the build succeeded" is not evidence the index matches the corpus. This
prints both numbers so they can be compared by eye or by a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _episode_id(doc_key: str) -> str | None:
    """The episode id inside an index doc key, for either corpus's key shape.

    Keys are ``<kind>:<feed-part>__<episode-id>[:<suffix>]``, and the two corpora fill the
    halves differently:

        app     bullet:p01__ep-0d9100843ee555f8:0
        viewer  bullet:sha256_1f9abe36…__254fe2be-b570-592e-84cd-c22cb9e271e9:0

    This was ``re.search(r"(ep-[0-9a-f]+)")``, which matches only the app shape. Against the
    viewer corpus it found nothing, reported "872 docs across 0 episodes", and printed
    MISMATCH — a confidently wrong answer from a tool whose whole job is to be the evidence
    that `index-two-tier`'s exit code is not. Parse the structure instead of one corpus's
    flavour of it, and say so loudly when no key parses at all.
    """
    _, sep, rest = doc_key.partition("__")
    if not sep:
        return None
    episode = rest.split(":", 1)[0].strip()
    return episode or None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: count_indexed_episodes.py <corpus-dir>", file=sys.stderr)
        return 2
    corpus = Path(sys.argv[1])
    meta = corpus / "search" / "metadata.json"
    # `search/metadata.json` is the INDEX sidecar, not an episode. It does not match
    # `*.metadata.json` (that pattern needs a stem before the suffix), so it is not counted
    # here and must not be subtracted — doing so reported 39 of 40 and a false MISMATCH.
    on_disk = len(list(corpus.rglob("*.metadata.json")))
    if not meta.is_file():
        print(f"  no index at {meta} — corpus has {on_disk} episode(s)")
        return 1
    try:
        docs = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  index at {meta} is unreadable: {exc}")
        return 1
    episodes = {e for k in docs if (e := _episode_id(str(k)))}
    if docs and not episodes:
        print(
            f"  {len(docs)} indexed doc(s) but no episode id could be parsed from any key "
            f"(e.g. {next(iter(docs))!r}) — this tool cannot judge coverage here"
        )
        return 1
    print(
        f"  {len(docs)} indexed doc(s) across {len(episodes)} episode(s); "
        f"{on_disk} episode(s) on disk"
    )
    if len(episodes) != on_disk:
        print(
            "  MISMATCH — the index does not cover the corpus. Clear "
            "search/episode_fingerprints.json and rebuild."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
