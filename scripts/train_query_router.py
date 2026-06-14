#!/usr/bin/env python3
"""Train the ML query router (RFC-092 / #860).

Bootstraps the ≥500 labeled queries the learned router needs, then trains a
LogisticRegression over MiniLM query embeddings and persists it for
``search.query_router.MLQueryRouter``.

Where the labels come from — and why they are real, not circular:
  Each query is generated from an intent *template* (e.g. "exact quote from
  {person} about {topic}" → ``raw_evidence``). The template defines the ground
  truth by construction, so the label is genuine — we are NOT silver-labeling with
  the rules router (which would only teach the model to copy the rules). The slots
  ({person}, {topic}) are filled from REAL corpus entities/topics (the LanceDB
  aux tier's ``kg_entity`` / ``kg_topic`` surface text), so the lexical distribution
  matches production queries. A small built-in entity list is used when no corpus
  is given.

Validation:
  - Stratified holdout: accuracy + per-class report on unseen queries.
  - Divergence vs the rules router on the holdout: shows the model learned signal
    beyond the rules (and where it disagrees) rather than memorising them.

Output: a joblib LogisticRegression whose ``predict`` takes a 384-dim MiniLM
embedding — exactly what ``MLQueryRouter`` feeds it. No ONNX runtime needed.

Usage:
  python scripts/train_query_router.py \
      --lance-dir .test_outputs/manual/my-manual-run-10/search/lance_index \
      --out ./data/query_router.joblib --per-template 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from podcast_scraper.providers.ml.embedding_loader import encode  # noqa: E402
from podcast_scraper.search.router import classify_query, QUERY_TYPES  # noqa: E402

# Intent templates. `{p}` = person/entity slot, `{t}` = topic slot. The label is
# the template's intent by construction.
_TEMPLATES: Dict[str, List[str]] = {
    "entity_lookup": [
        "{p}",
        "who is {p}",
        "tell me about {p}",
        "{p} background",
        "profile of {p}",
    ],
    "raw_evidence": [
        "exact quote from {p} about {t}",
        "what did {p} say verbatim about {t}",
        "transcript where {p} mentions {t}",
        "the phrase {p} used for {t}",
        "{p} said exactly what about {t}",
    ],
    "temporal_tracking": [
        "how has {t} evolved",
        "{t} over time",
        "the history of {t}",
        "how {p} changed their view on {t}",
        "trend in {t} discussions",
    ],
    "cross_show_synthesis": [
        "compare {p} and {t}",
        "{p} versus {t}",
        "contrast views on {t} across shows",
        "{t} across shows",
        "compare how shows cover {t}",
    ],
    "semantic": [
        "{t} explained",
        "why does {t} matter",
        "the idea behind {t}",
        "what drives {t}",
        "understanding {t}",
    ],
}

_FALLBACK_PEOPLE = [
    "Sam Altman",
    "Tim Cook",
    "Byrne Hobart",
    "Scott Bessent",
    "Rory Johnston",
    "Elon Musk",
    "Jensen Huang",
    "Janet Yellen",
    "Patrick Collison",
    "Marc Andreessen",
]
_FALLBACK_TOPICS = [
    "artificial intelligence",
    "interest rates",
    "oil markets",
    "inflation",
    "semiconductors",
    "venture capital",
    "energy policy",
    "the dollar",
    "machine learning",
    "supply chains",
]


def _corpus_entities(lance_dir: Path) -> Tuple[List[str], List[str]]:
    """(people, topics) surface strings from the LanceDB aux tier, else built-in fallbacks."""
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

    try:
        tbl = LanceDBBackend(str(lance_dir))._open_if_exists("aux")
    except Exception:  # noqa: BLE001 - no/unreadable index → fallbacks
        tbl = None
    if tbl is None:
        return _FALLBACK_PEOPLE, _FALLBACK_TOPICS
    people, topics = [], []
    cols = [c for c in ("doc_type", "text") if c in tbl.schema.names]
    n = tbl.count_rows()
    for r in tbl.search().limit(max(n, 1)).select(cols).to_list():
        txt = (r.get("text") or "").strip()
        if not txt or len(txt) > 40:
            continue
        if r.get("doc_type") == "kg_entity":
            people.append(txt)
        elif r.get("doc_type") == "kg_topic":
            topics.append(txt)
    # De-dupe, keep deterministic order.
    people = list(dict.fromkeys(people)) or _FALLBACK_PEOPLE
    topics = list(dict.fromkeys(topics)) or _FALLBACK_TOPICS
    return people, topics


def _build_dataset(
    people: List[str], topics: List[str], per_template: int
) -> List[Tuple[str, str]]:
    """Generate (query, intent) pairs — `per_template` per template, slots cycled."""
    data: List[Tuple[str, str]] = []
    for intent, templates in _TEMPLATES.items():
        for tmpl in templates:
            for i in range(per_template):
                q = tmpl.format(p=people[i % len(people)], t=topics[i % len(topics)])
                data.append((q, intent))
    return data


def main() -> int:
    """Bootstrap labeled queries, train the router, validate, and persist it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lance-dir",
        default=None,
        help="corpus LanceDB index dir (…/search/lance_index) for real slots",
    )
    ap.add_argument("--out", default="./data/query_router.joblib")
    ap.add_argument("--per-template", type=int, default=30)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    people, topics = (
        _corpus_entities(Path(args.lance_dir))
        if args.lance_dir
        else (_FALLBACK_PEOPLE, _FALLBACK_TOPICS)
    )
    print(f"Slots: {len(people)} people, {len(topics)} topics")

    dataset = _build_dataset(people, topics, args.per_template)
    print(f"Generated {len(dataset)} labeled queries across {len(QUERY_TYPES)} intents")
    if len(dataset) < 500:
        print(f"WARNING: only {len(dataset)} queries (< 500 target); raise --per-template")

    texts = [q for q, _ in dataset]
    labels = [y for _, y in dataset]
    print("Embedding queries (MiniLM) ...")
    X = np.array([encode(t, "minilm-l6", allow_download=True) for t in texts])
    y = np.array(labels)

    X_tr, X_te, y_tr, y_te, txt_tr, txt_te = train_test_split(
        X, y, texts, test_size=args.test_frac, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, C=4.0)
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"\nHoldout accuracy: {acc:.3f}  (n_test={len(y_te)})")
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))

    # Divergence vs rules — proves the model isn't just the rules in disguise.
    preds = clf.predict(X_te)
    rules = np.array([classify_query(t) for t in txt_te])
    agree_rules = float((preds == rules).mean())
    rules_acc = float((rules == y_te).mean())
    gain = acc - rules_acc
    print(f"ML vs rules agreement on holdout: {agree_rules:.3f}")
    print(f"Rules vs ground-truth accuracy:   {rules_acc:.3f}  (ML adds {gain:+.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)
    print(f"\nSaved router model → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
