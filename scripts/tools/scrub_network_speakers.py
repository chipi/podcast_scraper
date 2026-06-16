#!/usr/bin/env python3
"""Scrub network-name speaker leaks from an existing corpus (#1012, offline, no re-diar).

For each latest-run episode, any Person node whose name is a known network (e.g.
"Pushkin") is RE-ATTRIBUTED to the episode's resolved self-intro host (the current
roster code: extract_self_introduced_host). Offset-safe: only touches *fields*
(gi.json Person node id/name, quote.speaker_id, SPOKEN_BY edges, metadata
content.speakers, adfree.segments.json speaker_label) — never the rendered transcript
text, so GI quote char offsets stay valid.

Default = DRY RUN (reports). Pass --apply to write.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podcast_scraper.identity.slugify import person_id
from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
)
from podcast_scraper.speaker_detectors.hosts import extract_self_introduced_host, is_known_network

CORPUS = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else Path(".test_outputs/manual/prod-v2/corpus")
)
APPLY = "--apply" in sys.argv


def _gi_path(meta_path: Path, doc: dict) -> Path:
    root = episode_root_from_metadata_path(meta_path)
    rel = ((doc.get("grounded_insights") or {}).get("artifact_path") or "").strip()
    if rel:
        return (root / rel).resolve()
    return meta_path.with_name(meta_path.name.replace(".metadata.json", ".gi.json"))


def _transcript_text(meta_path: Path, doc: dict) -> str:
    root = episode_root_from_metadata_path(meta_path)
    rel = ((doc.get("content") or {}).get("transcript_file_path") or "").strip()
    if not rel:
        return ""
    p = (root / rel).resolve()
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def scrub_gi(gi: dict, leak_to_new: dict[str, tuple[str, str]]) -> int:
    """Rename leak Person ids→new id/name, repoint quote.speaker_id + edges, dedupe.

    Returns the number of edits made.
    """
    edits = 0
    existing_ids = {n.get("id") for n in gi.get("nodes", [])}
    remap: dict[str, str | None] = {}  # leak_id -> new_id, or None to drop
    kept_nodes = []
    for n in gi.get("nodes", []):
        nid = n.get("id")
        if nid in leak_to_new:
            new_id, new_name = leak_to_new[nid]
            remap[nid] = new_id
            edits += 1
            if new_id is None:
                continue  # drop the bogus speaker node entirely
            if new_id in existing_ids and new_id != nid:
                continue  # merge: target already present
            n["id"] = new_id
            (n.setdefault("properties", {}))["name"] = new_name
            existing_ids.add(new_id)
        kept_nodes.append(n)
    gi["nodes"] = kept_nodes
    # Repoint (or null) quote.speaker_id.
    for n in gi.get("nodes", []):
        props = n.get("properties") or {}
        sid = props.get("speaker_id")
        if sid in remap:
            props["speaker_id"] = remap[sid]  # new id, or None (unattributed)
            edits += 1
    # Repoint + dedupe edges; drop edges whose endpoint was dropped.
    seen = set()
    new_edges = []
    for e in gi.get("edges", []):
        frm, to = e.get("from"), e.get("to")
        if (frm in remap and remap[frm] is None) or (to in remap and remap[to] is None):
            edits += 1
            continue  # endpoint dropped -> drop the edge
        e["from"] = remap.get(frm, frm)
        e["to"] = remap.get(to, to)
        key = (e.get("type"), e.get("from"), e.get("to"))
        if key in seen:
            edits += 1
            continue
        seen.add(key)
        new_edges.append(e)
    gi["edges"] = new_edges
    return edits


def _feed_dir(meta_path: Path) -> str:
    for part in meta_path.parts:
        if part.startswith("rss_"):
            return part
    return ""


def main() -> int:
    print(f"corpus: {CORPUS}  mode: {'APPLY' if APPLY else 'DRY-RUN'}\n")
    metas = list(discover_metadata_files(CORPUS))

    # Pass 1: resolve a host per leaking episode + build a feed-level modal host so an
    # episode whose own self-intro doesn't resolve inherits the same show's host.
    from collections import Counter

    per_meta: dict[Path, str | None] = {}
    feed_votes: dict[str, Counter] = {}
    for meta_path in metas:
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gi_path = _gi_path(meta_path, doc)
        if not gi_path.is_file():
            continue
        try:
            gi = json.loads(gi_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not any(
            n.get("type") == "Person"
            and is_known_network((n.get("properties") or {}).get("name") or "")
            for n in gi.get("nodes", [])
        ):
            continue
        r = extract_self_introduced_host(_transcript_text(meta_path, doc))
        r = r if (r and not is_known_network(r)) else None
        per_meta[meta_path] = r
        if r:
            feed_votes.setdefault(_feed_dir(meta_path), Counter())[r] += 1
    feed_host = {f: c.most_common(1)[0][0] for f, c in feed_votes.items() if c}

    total = 0
    for meta_path in per_meta:
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
        gi_path = _gi_path(meta_path, doc)
        gi = json.loads(gi_path.read_text(encoding="utf-8"))
        leaks = [
            n
            for n in gi.get("nodes", [])
            if n.get("type") == "Person"
            and is_known_network((n.get("properties") or {}).get("name") or "")
        ]
        resolved = per_meta[meta_path] or feed_host.get(_feed_dir(meta_path))
        if resolved:
            new_id, new_name = person_id(resolved), resolved
        else:
            new_id, new_name = None, None  # no host anywhere → DROP the bogus speaker
        leak_to_new = {n["id"]: (new_id, new_name) for n in leaks}
        leak_summary = [(n["id"], (n.get("properties") or {}).get("name")) for n in leaks]
        print(f"{gi_path.name}\n  {leak_summary}  ->  {new_id} / {new_name!r}")
        edits = scrub_gi(gi, leak_to_new)
        # metadata content.speakers rename
        spk = (doc.get("content") or {}).get("speakers") or []
        md_edits = 0
        for s in spk:
            if is_known_network(s.get("name") or ""):
                s["name"] = new_name
                md_edits += 1
        # adfree.segments.json speaker_label rename
        root = episode_root_from_metadata_path(meta_path)
        tr_rel = ((doc.get("content") or {}).get("transcript_file_path") or "").strip()
        seg_edits = 0
        seg_file = None
        if tr_rel:
            base = root / tr_rel
            seg_file = base.with_suffix("")  # strip .txt
            seg_file = seg_file.with_name(seg_file.name + ".adfree.segments.json")
        if seg_file and seg_file.is_file():
            try:
                segs = json.loads(seg_file.read_text(encoding="utf-8"))
                for sg in segs:
                    if is_known_network(sg.get("speaker_label") or ""):
                        sg["speaker_label"] = new_name
                        seg_edits += 1
            except (OSError, ValueError):
                segs = None
        print(f"  edits: gi={edits} content.speakers={md_edits} adfree.segments={seg_edits}")
        total += 1
        if APPLY:
            gi_path.write_text(json.dumps(gi, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            meta_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if seg_file and seg_file.is_file() and seg_edits:
                seg_file.write_text(
                    json.dumps(segs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
    print(f"\n{'APPLIED' if APPLY else 'WOULD CHANGE'} {total} episode(s).")
    print(
        "NOTE: rendered transcript .txt/.adfree.txt keep 'Pushkin:' line labels"
        " (changing them shifts GI quote char offsets — needs a re-render, not done here)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
