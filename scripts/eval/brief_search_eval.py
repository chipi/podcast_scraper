#!/usr/bin/env python3
"""Evaluate the Brief's search for what a listener remembers (operator 2026-10-10).

The use case: someone listened to an episode, remembers a few words or a term, and wants that
part again — to highlight a piece of the transcript, or to find the insight about it. This runs
the Brief's own search (``structured_corpus_search`` scoped to one episode, exactly as
``GET /episodes/{slug}/search`` calls it) over real episodes, with queries built from what a
listener might remember of a random passage:

* ``one_word``     — the passage's most distinctive word (rarest in the episode)
* ``few_words``    — its three most distinctive words, shuffled
* ``phrase``       — four consecutive words, verbatim
* ``word_form``    — the distinctive word in another form (plural/singular, -ing/-ed)
* ``near_synonym`` — the few words with one swapped for a near-synonym (an LLM writes it;
                     optional, ``--synonyms``)

A query FINDS the passage when a transcript result contains it (text match: a run of eight of the
passage's words; the index's transcript chunks carry no timing on this corpus or on prod). For
passages with an insight grounded in them, it also checks whether that insight comes back.

The baseline ``exact`` is a plain word search over the transcript: the passages that contain every
query word as a whole word.

No real transcript text leaves ``--out`` (gitignored); the committed report carries numbers only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

STOP = set(
    """a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't down during each even ever every few for from further get gets getting got
    had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him
    himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just know
    kind let's like lot really right said say says see so some such than that that's the their
    theirs them themselves then there there's these they they'd they'll they're they've thing
    things think this those through to too under until up us very was wasn't way we we'd we'll
    we're we've well were weren't what what's when when's where where's which while who who's whom
    why why's will with won't would wouldn't yeah yes you you'd you'll you're you've your yours
    yourself yourselves actually going gonna want mean okay sort maybe much many more most other
    something someone because been people time year years one two three first""".split()
)
WORD = re.compile(r"[a-z][a-z'\-]*")
TYPES = ("one_word", "few_words", "phrase", "word_form", "near_synonym")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def norm(text: str) -> str:
    return " ".join(words(text))


def content(ws: list[str]) -> list[str]:
    return [w for w in ws if len(w) >= 5 and w not in STOP and "'" not in w]


def other_form(w: str) -> str | None:
    if w.endswith("ing") and len(w) > 6:
        return w[:-3] + "ed"
    if w.endswith("ed") and len(w) > 5:
        return w[:-2] + "ing"
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 5:
        return w[:-1]
    if w.endswith("y") and len(w) > 4:
        return w[:-1] + "ies"
    return w + "s"


def passages(segs: list[dict], target_words: int = 30) -> list[dict]:
    """Consecutive Whisper segments merged into passages of ~``target_words`` words."""
    out: list[dict] = []
    cur: list[dict] = []
    for s in segs:
        cur.append(s)
        n = sum(len(words(x.get("text") or "")) for x in cur)
        if n >= target_words:
            text = " ".join((x.get("text") or "").strip() for x in cur)
            out.append(
                {
                    "text": text,
                    "start": float(cur[0].get("start") or 0),
                    "end": float(cur[-1].get("end") or 0),
                }
            )
            cur = []
    return out


def episodes(corpus: Path) -> list[dict]:
    rows = []
    for m in sorted(glob.glob(str(corpus / "feeds/*/*/metadata/*.metadata.json"))):
        meta = json.loads(Path(m).read_text())
        rel = (meta.get("content") or {}).get("transcript_file_path")
        eid = (meta.get("episode") or {}).get("episode_id")
        if not rel or not eid:
            continue
        run = Path(m).parent.parent
        seg = run / (rel[:-4] + ".segments.json")
        gi = Path(m[: -len(".metadata.json")] + ".gi.json")
        if not seg.exists():
            continue
        segs = json.loads(seg.read_text())
        segs = segs.get("segments", segs) if isinstance(segs, dict) else segs
        rows.append(
            {
                "episode_id": eid,
                "feed_id": (meta.get("feed") or {}).get("feed_id"),
                "segments": segs,
                "gi": json.loads(gi.read_text()) if gi.exists() else None,
            }
        )
    return rows


def grounded_insights(gi: dict | None, start: float, end: float) -> set[str]:
    """Insight node ids whose supporting quote overlaps [start, end] seconds."""
    if not gi:
        return set()
    quotes = {}
    for n in gi.get("nodes", []):
        if n.get("type") == "Quote":
            p = n.get("properties") or {}
            s, e = p.get("timestamp_start_ms"), p.get("timestamp_end_ms")
            if isinstance(s, (int, float)) and isinstance(e, (int, float)):
                quotes[n["id"]] = (s / 1000.0, e / 1000.0)
    out = set()
    for e in gi.get("edges", []):
        if e.get("type") == "SUPPORTED_BY" and e.get("to") in quotes:
            qs, qe = quotes[e["to"]]
            if qs < end and qe > start:
                out.add(e["from"])
    return out


class Synonyms:
    """One LLM call per episode: each passage's few words with one swapped for a near-synonym."""

    def __init__(self, cache: Path, model: str) -> None:
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key and (REPO / ".env").exists():
            for line in (REPO / ".env").read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.cache = cache
        self.done: dict[str, dict] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.done[row["episode_id"]] = row["queries"]
        self.tokens = [0, 0]

    def get(self, eid: str, items: list[tuple[int, str, list[str]]]) -> dict:
        if eid in self.done:
            return self.done[eid]
        listing = "\n".join(
            f'{i}. passage: "{text}" | remembered words: {", ".join(ws)}' for i, text, ws in items
        )
        prompt = (
            "A podcast listener half-remembers a moment and types a few words to find it again. "
            "For each item, rewrite the remembered words as the listener might misremember them: "
            "replace exactly ONE of the words with a near-synonym or closely related word "
            "that does NOT appear in the passage. Keep the others. Reply with only a JSON "
            "array of "
            '{"id": <number>, "query": "<words>"}.\n\n' + listing
        )
        msg = self.client.messages.create(
            model=self.model, max_tokens=1500, messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(getattr(b, "text", "") or "" for b in msg.content)
        self.tokens[0] += int(msg.usage.input_tokens)
        self.tokens[1] += int(msg.usage.output_tokens)
        start = text.find("[")
        rows, _ = json.JSONDecoder().raw_decode(text[start:])
        out = {str(int(r["id"])): str(r["query"]) for r in rows}
        with self.cache.open("a") as f:
            f.write(json.dumps({"episode_id": eid, "queries": out}) + "\n")
        self.done[eid] = out
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=".test_outputs/brief_search_eval/corpus")
    ap.add_argument("--out", default=".test_outputs/brief_search_eval")
    ap.add_argument("--per-episode", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--synonyms", action="store_true", help="add near_synonym queries (paid LLM)")
    ap.add_argument("--synonym-model", default="claude-haiku-4-5-20251001")
    ap.add_argument(
        "--brief",
        action="store_true",
        help="compose results as the Brief's route does: verbatim passages first, then the index "
        "results with their real time (app_episode_search)",
    )
    args = ap.parse_args()

    from types import SimpleNamespace

    from podcast_scraper.search.capability import structured_corpus_search
    from podcast_scraper.server.app_episode_search import exact_passages, time_transcript_hits

    corpus, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    syn = Synonyms(out / "synonyms.jsonl", args.synonym_model) if args.synonyms else None
    eps = episodes(corpus)
    cases: list[dict] = []
    for n_ep, ep in enumerate(eps, 1):
        ps = [p for p in passages(ep["segments"]) if len(content(words(p["text"]))) >= 3]
        if len(ps) < args.per_episode:
            continue
        freq = Counter(words(" ".join((s.get("text") or "") for s in ep["segments"])))
        seg_ms = [
            (
                int(float(x.get("start") or 0) * 1000),
                int(float(x.get("end") or 0) * 1000),
                x.get("text") or "",
            )
            for x in ep["segments"]
        ]
        all_norm = [norm(p["text"]) for p in ps]
        picked = rng.sample(range(len(ps)), args.per_episode)
        built: list[tuple[int, dict, dict]] = []
        for i in picked:
            p = ps[i]
            ws = words(p["text"])
            # Sorted before the random tie-break: a set's order changes per process (string hashing
            # is randomized), which made the seeded run give different queries on every run.
            cw = sorted(sorted(set(content(ws))), key=lambda w: (freq[w], rng.random()))
            rare = cw[0]
            few = cw[:3]
            rng.shuffle(few)
            at = ws.index(rare)
            lo = max(0, min(at - 1, len(ws) - 4))
            queries = {
                "one_word": rare,
                "few_words": " ".join(few),
                "phrase": " ".join(ws[lo : lo + 4]),
            }
            form = other_form(rare)
            if form and form != rare:
                queries["word_form"] = form
            built.append((i, p, queries))
        if syn:
            items = [(k, p["text"], q["few_words"].split()) for k, (_, p, q) in enumerate(built)]
            swapped = syn.get(ep["episode_id"], items)
            for k, (_, _, q) in enumerate(built):
                if str(k) in swapped:
                    q["near_synonym"] = swapped[str(k)]
        for i, p, queries in built:
            pn = norm(p["text"]).split()
            probes = {
                " ".join(pn[j : j + 8]) for j in (0, max(0, len(pn) - 8), max(0, len(pn) // 2 - 4))
            }
            insight_ids = grounded_insights(ep["gi"], p["start"], p["end"])
            p_ms = (p["start"] * 1000, p["end"] * 1000)
            for qtype, query in queries.items():
                res = structured_corpus_search(
                    corpus, query, feed=ep["feed_id"], episode_id=ep["episode_id"], top_k=args.top_k
                )
                results = res.get("results") or []
                if args.brief:
                    hits = [SimpleNamespace(**r) for r in results]
                    time_transcript_hits(hits, seg_ms)
                    results = [
                        *exact_passages(seg_ms, query, episode_id=ep["episode_id"]),
                        *(vars(h) for h in hits),
                    ]
                rank = None
                insight_rank = None
                types: Counter[str] = Counter()
                for r_i, r in enumerate(results, 1):
                    md = r.get("metadata") or {}
                    dt = str(md.get("doc_type"))
                    types[dt] += 1
                    if rank is None and dt == "transcript":
                        hn = norm(r.get("text") or "")
                        s_ms, e_ms = md.get("timestamp_start_ms"), md.get("timestamp_end_ms")
                        # Found: the passage's words, or (Brief mode, timed results) its moment.
                        timed = (
                            args.brief
                            and isinstance(s_ms, (int, float))
                            and isinstance(e_ms, (int, float))
                            and s_ms < p_ms[1]
                            and e_ms > p_ms[0]
                        )
                        if timed or any(pr and pr in hn for pr in probes):
                            rank = r_i
                    if (
                        insight_rank is None
                        and dt == "insight"
                        and md.get("source_id") in insight_ids
                    ):
                        insight_rank = r_i
                qws = [w for w in words(query) if w not in STOP]
                exact = [
                    k
                    for k, a in enumerate(all_norm)
                    if qws and all(re.search(rf"\b{re.escape(w)}\b", a) for w in qws)
                ]
                cases.append(
                    {
                        "episode_id": ep["episode_id"],
                        "passage": i,
                        "type": qtype,
                        "query": query,
                        "rank": rank,
                        "insight_rank": insight_rank,
                        "has_insight": bool(insight_ids),
                        "types": dict(types),
                        "error": res.get("error"),
                        "exact_found": i in exact,
                        "exact_matches": len(exact),
                    }
                )
        print(f"[{n_ep}/{len(eps)}] {ep['episode_id']}: {len(built)} passages", flush=True)

    (out / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    summary = summarise(cases, args.top_k)
    if syn:
        summary["synonym_tokens"] = {"input": syn.tokens[0], "output": syn.tokens[1]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def summarise(cases: list[dict], top_k: int) -> dict[str, Any]:
    by: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by[c["type"]].append(c)
    table = {}
    for t in TYPES:
        cs = by.get(t) or []
        if not cs:
            continue
        n = len(cs)
        ranks = [c["rank"] for c in cs]
        ins = [c for c in cs if c["has_insight"]]
        table[t] = {
            "n": n,
            "found@1": round(sum(1 for r in ranks if r == 1) / n, 3),
            "found@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
            f"found@{top_k}": round(sum(1 for r in ranks if r) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "insight_cases": len(ins),
            f"insight_found@{top_k}": (
                round(sum(1 for c in ins if c["insight_rank"]) / len(ins), 3) if ins else None
            ),
            "exact_found": round(sum(1 for c in cs if c["exact_found"]) / n, 3),
            "exact_median_matches": sorted(c["exact_matches"] for c in cs)[n // 2],
            "transcript_slots_mean": round(sum(c["types"].get("transcript", 0) for c in cs) / n, 2),
            "errors": sum(1 for c in cs if c["error"]),
        }
    slot_types: Counter = Counter()
    for c in cases:
        slot_types.update(c["types"])
    return {
        "episodes": len({c["episode_id"] for c in cases}),
        "cases": len(cases),
        "by_type": table,
        "result_types": dict(slot_types),
    }


if __name__ == "__main__":
    raise SystemExit(main())
