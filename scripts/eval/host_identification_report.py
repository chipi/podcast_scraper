"""Host-identification scorecard — EPIC-HOST-IDENTIFICATION Slice 0.

Builds each show's **identified** host(s) from a corpus (the ``role==host`` Person
nodes in the episode KGs, aggregated per feed), compares to the **gold** set
(``data/eval/host/gold.jsonl``), and reports coverage / recall / precision /
co-host completeness / named-not-SPEAKER, split into ``dev`` (tune on) and the
held-out ``test`` shows (the honest generalization measure). Running it now is the
BASELINE row for the epic.

It measures the corpus **as built** (the pipeline version that produced it),
mirroring the roadmap's replay-on-cached-artifacts method — no audio/LLM/GPU.

Usage:
    python scripts/eval/host_identification_report.py --corpus <corpus_root> [--json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_PATH = os.path.join(REPO, "data", "eval", "host", "gold.jsonl")
DEFAULT_CORPUS = os.path.join(REPO, ".test_outputs", "manual", "prod-v2", "corpus")

# A corpus feed's show TITLE (substring, lowered) -> gold show_id.
TITLE_TO_SHOW = {
    "invest like the best": "invest_like_the_best",
    "no priors": "no_priors",
    "unhedged": "unhedged",
    "planet money": "planet_money",
    "the journal": "the_journal",
    "nvidia ai podcast": "nvidia_ai_podcast",
    "hard fork": "hard_fork",
    "the daily": "the_daily",
    "latent space": "latent_space",
    "odd lots": "odd_lots",
}
# Known host aliases (stage-name -> legal name) so the matcher is not fooled by them.
ALIASES = {"swyx": "shawn wang"}
_SPEAKER_RE = re.compile(r"^(?:person:)?speaker[_\-]", re.IGNORECASE)


def _norm(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def _tokens(name: str) -> set[str]:
    return {t for t in _norm(name).split() if len(t) > 1}


def _matches(gold_name: str, identified: set[str]) -> bool:
    """A gold host is found if its normalized name equals, or its tokens are a subset of,
    any identified name (handles 'Patrick' vs 'Patrick O'Shaughnessy' and aliases)."""
    gnorm, gtok = _norm(gold_name), _tokens(gold_name)
    for cand in identified:
        cnorm, ctok = _norm(cand), _tokens(cand)
        if gnorm and gnorm == cnorm:
            return True
        if gtok and gtok <= ctok:
            return True
    return False


def load_gold(path: str) -> dict[str, dict[str, Any]]:
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                gold[row["show_id"]] = row
    return gold


def _feed_title(feed_dir: str) -> str | None:
    for m in sorted(glob.glob(f"{feed_dir}/**/*metadata*.json", recursive=True)):
        try:
            d = json.load(open(m, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        f = d.get("feed") if isinstance(d.get("feed"), dict) else d
        title = f.get("title") or f.get("feed_title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


def _show_id_for_title(title: str | None) -> str | None:
    if not title:
        return None
    low = title.lower()
    for frag, sid in TITLE_TO_SHOW.items():
        if frag in low:
            return sid
    return None


def _identified_hosts(feed_dir: str) -> tuple[set[str], int, int]:
    """Return (distinct host names across episodes, n_named, n_speaker_placeholder)."""
    names: set[str] = set()
    n_named = n_speaker = 0
    for p in glob.glob(f"{feed_dir}/**/*.kg.json", recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for n in d.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            props = n.get("properties") or {}
            is_person = n.get("type") == "Person" or str(n.get("id", "")).startswith("person:")
            if not is_person or props.get("role") != "host":
                continue
            nid = str(n.get("id") or "")
            nm = props.get("name") or nid
            if _SPEAKER_RE.match(nid) or _SPEAKER_RE.match(str(nm)):
                n_speaker += 1
            else:
                n_named += 1
                names.add(str(nm))
    return names, n_named, n_speaker


def score_show(gold_row: dict[str, Any], identified: set[str]) -> dict[str, Any]:
    gold_hosts = gold_row["hosts"]
    found = [g for g in gold_hosts if _matches(g, identified)]
    correct = [c for c in identified if any(_matches(g, {c}) for g in gold_hosts)]
    coverage = len(identified) >= 1
    recall = len(found) / len(gold_hosts) if gold_hosts else 0.0
    precision = len(correct) / len(identified) if identified else 0.0
    cohost_complete = (len(found) == len(gold_hosts)) if len(gold_hosts) >= 2 else None
    return {
        "coverage": coverage,
        "recall": recall,
        "precision": precision,
        "cohost_complete": cohost_complete,
        "found": found,
        "identified": sorted(identified),
    }


def build_report(corpus: str) -> dict[str, Any]:
    gold = load_gold(GOLD_PATH)
    per_show: dict[str, dict[str, Any]] = {}
    named_tot = speaker_tot = 0
    for feed_dir in sorted(glob.glob(f"{corpus}/feeds/*/")):
        sid = _show_id_for_title(_feed_title(feed_dir))
        if sid is None or sid not in gold:
            continue
        identified, n_named, n_speaker = _identified_hosts(feed_dir)
        named_tot += n_named
        speaker_tot += n_speaker
        row = score_show(gold[sid], identified)
        row.update(split=gold[sid]["split"], show_title=gold[sid]["show_title"])
        per_show[sid] = row
    named_pct = named_tot / (named_tot + speaker_tot) if (named_tot + speaker_tot) else 0.0
    return {"per_show": per_show, "named_not_speaker_pct": named_pct}


def _agg(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"coverage": 0.0, "recall": 0.0, "precision": 0.0}
    n = len(rows)
    return {
        "coverage": sum(1 for r in rows if r["coverage"]) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = build_report(args.corpus)
    per_show = report["per_show"]
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Host-identification scorecard — corpus {args.corpus}")
    print(f"{'show':22} {'split':5} {'cov':>3} {'rec':>5} {'prec':>5} {'cohost':>6}  identified")
    for sid in sorted(per_show, key=lambda s: (per_show[s]["split"], s)):
        r = per_show[sid]
        ch = "-" if r["cohost_complete"] is None else ("yes" if r["cohost_complete"] else "NO")
        cov = "yes" if r["coverage"] else "NO"
        ident = ", ".join(r["identified"]) or "(none)"
        print(
            f"{sid:22} {r['split']:5} {cov:>3} {r['recall']:5.2f} {r['precision']:5.2f} "
            f"{ch:>6}  {ident[:44]}"
        )
    for split in ("dev", "test"):
        rows = [r for r in per_show.values() if r["split"] == split]
        a = _agg(rows)
        print(
            f"\n[{split}] n={len(rows)}  coverage={a['coverage']:.0%}  "
            f"recall={a['recall']:.0%}  precision={a['precision']:.0%}"
        )
    print(f"\nnamed-not-SPEAKER (host voices): {report['named_not_speaker_pct']:.0%}")


if __name__ == "__main__":
    main()
