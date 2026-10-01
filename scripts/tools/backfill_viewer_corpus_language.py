"""#2185: write the declared language onto the committed viewer corpus. Surgical, not a rebuild.

WHY NOT REGENERATE. `build_synthetic_validation_corpus.py` already writes these fields — the
corpus simply predates that change. But re-running it is not a safe way to apply them:

* `base_date = datetime.utcnow()` (line 812), so every run shifts all 40 publish dates and the
  derived graph nodes with them — 131 of 332 files differ on a no-op rerun, almost none of it
  about language;
* it would DELETE 18 git-TRACKED `.app/users/*` files (profiles, preferences, graph events) that
  the viewer e2e reads and the generator does not produce.

So this writes exactly the three fields, computed the way the generator computes them — the RSS
tag through `normalize_language_tag` — and touches nothing else. Making the generator
deterministic is the other half of #2185 and is not this change.
"""

from __future__ import annotations

import json
import pathlib
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "src")
from podcast_scraper.languages import normalize_language_tag  # noqa: E402

ROOT = pathlib.Path("tests/fixtures/viewer-validation-corpus/v3")
RSS = pathlib.Path("tests/fixtures/rss")
APPLY = "--apply" in sys.argv


def declared_language(feed_id: str) -> tuple[str, str] | None:
    """`(raw, normalized)` from the feed's own RSS fixture — the same input the generator reads."""
    for xml in sorted(RSS.glob(f"{feed_id}_*.xml")):
        try:
            ch = ET.parse(xml).getroot().find("channel")
        except ET.ParseError:
            continue
        if ch is None:
            continue
        raw = (ch.findtext("language") or "").strip()
        if raw:
            norm = normalize_language_tag(raw)
            if norm:
                return raw, norm
    return None


changed, skipped = 0, []
for meta in sorted(ROOT.glob("feeds/**/metadata/*.metadata.json")):
    doc = json.loads(meta.read_text(encoding="utf-8"))
    feed = doc.setdefault("feed", {})
    episode = doc.setdefault("episode", {})
    # The SHOW id comes from the path (`feeds/p01/metadata/...`), not from `feed.feed_id` —
    # the viewer corpus stores a sha256 there, which matches no RSS fixture filename.
    feed_id = meta.parent.parent.name
    found = declared_language(feed_id)
    if not found:
        skipped.append(f"{meta.name} (no <language> in {feed_id}'s RSS)")
        continue
    raw, norm = found
    wanted = {
        ("feed", "language"): norm,
        ("feed", "language_raw"): raw,
        ("feed", "language_source"): "rss",
        ("episode", "language"): norm,
        ("episode", "language_source"): "rss",
    }
    current = {k: (feed if k[0] == "feed" else episode).get(k[1]) for k in wanted}
    if current == wanted:
        continue
    if APPLY:
        for (block, field), value in wanted.items():
            (feed if block == "feed" else episode)[field] = value
        # `ensure_ascii` DEFAULTED (True), matching the generator at line 922 and the bytes
        # already on disk. Passing False rewrote every `\u2014` as a literal em-dash and turned a
        # 200-line change into 490 lines of diff that said nothing about language.
        meta.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    changed += 1

print(f"{'wrote' if APPLY else 'would write'} language onto {changed} episode(s)")
if skipped:
    print(f"  skipped {len(skipped)}: {skipped[:3]}")
if not APPLY:
    print("  DRY RUN — pass --apply")
