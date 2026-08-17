#!/usr/bin/env python3
"""A purpose-built corpus for OBSERVING what each ranking signal does (#71).

Run it and read the table:

    python scripts/eval/score/rank_scenarios_v1.py

For every scenario it prints the feed a persona gets under each signal configuration, so a human
can see what moved rather than trusting a boolean. That is the point: a passing assertion tells you
a property held; this tells you *what the ranker did*.

WHY A SECOND CORPUS EXISTS
--------------------------
``tests/fixtures/app-validation-corpus/v3`` is the product's realism fixture — it answers "does the
app work against a real pipeline's output". It cannot answer "what does this signal do", and that
is not a defect in it; the two jobs pull in opposite directions. Realism means the confounds stay
in. Observation means removing every confound but one.

Measured, not assumed. Three times in one day v3 could not discriminate a ranking question:

* **half-life (#22)** — v3 spans 925 days, so 14d / 30d / 90d produced IDENTICAL orderings. The
  parameter was unobservable, and a completely inert first choice still passed the gate.
* **coverage bias (#23)** — v3 is uniformly enriched, so a bias toward well-enriched episodes is
  invisible *by construction*. The bug had to be reproduced in a throwaway two-feed corpus.
* **ranking order (#28)** — a fixture whose count-order happened to equal alphabetical order let a
  sabotaged projection pass. Tautological, and it looked fine.

Each was worked around with a corpus built inline in one test file. Those are invisible to each
other, so nothing showed how the signals INTERACT — and they do: #23's per-feed normalisation moved
#22's numbers by 20 points, and #19's saturation changed how much power a fixed trend weight has.

WHAT IS DESIGNED IN
-------------------
Each axis exists so that exactly one signal can decide an ordering:

    dates        two clusters — four episodes within a week, four spread over two years, so a
                 half-life between them is DISCRIMINABLE (v3's 925-day span is not)
    enrichment   one dense feed (bullets + GI + KG), one sparse feed (bare) — so coverage bias
                 shows up as a preference for the dense feed regardless of content
    topics       per-EPISODE niche topics, never per-feed constants (v3's are per-feed, which is
                 why its picker collapses to one cluster — #1669)
    trend        one topic repeated across recent episodes, one flat
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from podcast_scraper.server.app_discover_view import (  # noqa: E402
    build_discover_pool,
    rank_discover,
)
from podcast_scraper.server.app_ranking_config import (  # noqa: E402
    DEFAULT_RANKING_CONFIG,
    RankingConfig,
    RankingSignal,
    SIGNAL_INTEREST_AFFINITY,
    SIGNAL_RECENCY,
    SIGNAL_SIGNIFICANCE,
    SIGNAL_TREND_VELOCITY,
)
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative  # noqa: E402

#: Every publish date is relative to this, so the corpus is deterministic and its AGE STRUCTURE —
#: which is the thing half-life reads — stays fixed no matter when the file is generated.
_ANCHOR = date(2026, 1, 1)


@dataclass(frozen=True)
class Episode:
    """One episode, described by the properties a ranking signal can see."""

    eid: str
    title: str
    days_old: int
    topics: tuple[str, ...] = ()
    persons: tuple[str, ...] = ()
    #: Enrichment DEPTH, 0-3. The knob coverage bias reads, and deliberately independent of
    #: topics: every episode carries its KG (so affinity can always match it) while depth varies.
    #:
    #:   3  KG + GI + 5 summary bullets   — a fully-enriched episode
    #:   2  KG + GI, no bullets
    #:   1  KG only                        — the sparse show
    #:   0  KG only, and no summary at all
    #:
    #: The first version made `sparse` mean "no artifacts at all", which ALSO removed the topics —
    #: so the coverage-bias scenario could never fire, because affinity had nothing to match on.
    #: The observer table showed that row as unchanged, which is how the design fault surfaced.
    depth: int = 3


@dataclass(frozen=True)
class Feed:
    fid: str
    title: str
    episodes: tuple[Episode, ...] = field(default_factory=tuple)


# --- the corpus ---------------------------------------------------------------------------------
#
# Deliberately small. Every episode is here to make one question answerable, and the comment on
# each feed says which.

CORPUS: tuple[Feed, ...] = (
    # RECENT CLUSTER: four episodes inside one week. Against the OLD cluster below, this is what
    # makes a half-life discriminable — a short half-life separates these from the old ones
    # sharply, a long one barely at all. v3 cannot do this: everything in it is far apart.
    Feed(
        "p01",
        "Trail Craft",
        (
            # Depth VARIES inside the feed, so significance has something to reorder. With a
            # uniform feed, per-feed normalisation divides every episode by the same mean and
            # the signal is invisible — which is exactly what the first table showed.
            Episode(
                "p01_e01",
                "Tyre pressure and traction",
                1,
                ("topic:tyres",),
                ("person:mara",),
                depth=1,
            ),
            Episode(
                "p01_e02", "Suspension setup", 3, ("topic:suspension",), ("person:mara",), depth=3
            ),
            Episode("p01_e03", "Braking late", 5, ("topic:braking",), ("person:ivo",), depth=0),
            Episode("p01_e04", "Line choice", 7, ("topic:lines",), ("person:ivo",), depth=3),
        ),
    ),
    # OLD CLUSTER: same shape, two years back.
    Feed(
        "p02",
        "Long Reads",
        (
            Episode("p02_e01", "The long game", 500, ("topic:strategy",), ("person:nils",)),
            Episode("p02_e02", "Patience as method", 600, ("topic:method",), ("person:nils",)),
            Episode("p02_e03", "Slow compounding", 700, ("topic:compounding",), ("person:ada",)),
            Episode("p02_e04", "Deep work", 800, ("topic:focus",), ("person:ada",)),
        ),
    ),
    # SPARSE FEED: recent, but carries NO enrichment. Under a raw significance signal these lose to
    # the dense feed on coverage alone; per-feed normalisation is what gives them a fair hearing.
    Feed(
        "p03",
        "Field Notes",
        (
            Episode("p03_e01", "Notes from the ridge", 2, ("topic:ridge",), (), depth=1),
            Episode("p03_e02", "Notes from the pass", 4, ("topic:pass",), (), depth=1),
            Episode("p03_e03", "Notes from the col", 6, ("topic:col",), (), depth=1),
        ),
    ),
    # TREND FEED: `topic:hot` recurs across three recent episodes; `topic:flat` appears once, long
    # ago. Trend velocity is the only signal that can tell these apart.
    Feed(
        "p04",
        "Signal Watch",
        (
            Episode("p04_e01", "Hot topic, part one", 2, ("topic:hot",), ("person:sam",)),
            Episode("p04_e02", "Hot topic, part two", 4, ("topic:hot",), ("person:sam",)),
            Episode("p04_e03", "Hot topic, part three", 6, ("topic:hot",), ("person:sam",)),
            Episode("p04_e04", "A flat subject", 400, ("topic:flat",), ("person:sam",)),
        ),
    ),
    # NICHE FEED: one deeply specific topic nothing else touches, so a single follow can be shown
    # to decide an ordering entirely on its own.
    Feed(
        "p05",
        "Niche Hour",
        (
            Episode("p05_e01", "Rope access rigging", 9, ("topic:rigging",), ("person:tess",)),
            Episode("p05_e02", "Anchor theory", 11, ("topic:rigging",), ("person:tess",)),
        ),
    ),
)


def _publish_date(days_old: int) -> str:
    return (_ANCHOR - timedelta(days=days_old)).isoformat() + "T00:00:00"


def write_corpus(root: Path) -> Path:
    """Write the scenario corpus under ``root`` and return the corpus root."""
    for feed in CORPUS:
        meta_dir = root / "feeds" / feed.fid / "run_20260101-000000" / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        for ep in feed.episodes:
            doc = {
                "feed": {
                    "feed_id": feed.fid,
                    "title": feed.title,
                    "url": f"https://example.test/{feed.fid}.xml",
                },
                "episode": {
                    "episode_id": ep.eid,
                    "title": ep.title,
                    "published_date": _publish_date(ep.days_old),
                    "duration_seconds": 1200 + ep.days_old,
                },
                "content": {"transcript_file_path": f"transcripts/{ep.eid}.txt"},
            }
            if ep.depth >= 1:
                doc["summary"] = {
                    "title": ep.title,
                    "bullets": (
                        [f"{ep.title} point {i}" for i in range(1, 6)] if ep.depth >= 3 else []
                    ),
                    "raw_text": f"A real summary of {ep.title.lower()}, in the show's own words.",
                }
            (meta_dir / f"{ep.eid}.metadata.json").write_text(
                json.dumps(doc, indent=2), encoding="utf-8"
            )
            # The KG is written at EVERY depth: topics are what affinity matches on, and stripping
            # them along with the enrichment is what made the coverage-bias scenario inert.
            if True:
                nodes = [
                    {"id": t, "type": "Topic", "properties": {"label": t.split(":", 1)[-1]}}
                    for t in ep.topics
                ]
                nodes += [
                    {"id": p, "type": "Person", "properties": {"name": p.split(":", 1)[-1]}}
                    for p in ep.persons
                ]
                (meta_dir / f"{ep.eid}.kg.json").write_text(
                    json.dumps({"episode_id": ep.eid, "nodes": nodes}, indent=2), encoding="utf-8"
                )
            if ep.depth >= 2:
                (meta_dir / f"{ep.eid}.gi.json").write_text(
                    json.dumps(
                        {
                            "episode_id": ep.eid,
                            "nodes": [
                                {
                                    "type": "Insight",
                                    "properties": {"text": f"An insight about {ep.title.lower()}."},
                                }
                            ],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    # The trend signal reads this. Without it every "+ trend" row is identical to the row above —
    # a configuration that cannot differ, printed as though it were an observation. The first
    # version of this corpus shipped exactly that.
    enrich = root / "enrichments"
    enrich.mkdir(parents=True, exist_ok=True)
    (enrich / "temporal_velocity.json").write_text(
        json.dumps(
            {
                "data": {
                    "topics": [
                        {"topic_id": "topic:hot", "velocity_last_over_6mo": 3.0},
                        {"topic_id": "topic:flat", "velocity_last_over_6mo": 1.0},
                        {"topic_id": "topic:rigging", "velocity_last_over_6mo": 1.0},
                    ]
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


# --- signal configurations ------------------------------------------------------------------------
#
# Each is the SHIPPED default with one thing changed, so a difference between two rows is
# attributable to that one thing and nothing else.


#: Significance params that make the base score CONSTANT — the only way to actually silence it.
#:
#: Zeroing significance's WEIGHT does nothing: it is the base multiplier, not an additive term
#: (``score = (significance / feed_mean) * (1 + boosts)``), and `rank_discover` reads its PARAMS,
#: never its weight. The first version of this table used a zero weight and printed a
#: "+ significance" row identical to "recency only" in every scenario — a configuration that
#: could not differ, presented as an observation. Same trap as #21, one layer along.
_FLAT_SIGNIFICANCE = {"gi_bonus": 0.0, "kg_bonus": 0.0, "bullet_step": 0.0, "bullet_cap": 0}


def _with(*, flat_significance: bool = False, **weights: float) -> RankingConfig:
    """The default config with the named signals' weights overridden (0.0 disables).

    ``flat_significance`` additionally neutralises the base score, which weights cannot do.
    """
    return RankingConfig(
        signals=tuple(
            RankingSignal(
                s.name,
                enabled=(weights.get(s.name, s.weight if s.enabled else 0.0) > 0),
                weight=weights.get(s.name, s.weight),
                params=(
                    dict(_FLAT_SIGNIFICANCE)
                    if (flat_significance and s.name == SIGNAL_SIGNIFICANCE)
                    else s.params
                ),
            )
            for s in DEFAULT_RANKING_CONFIG.signals
        )
    )


CONFIGS: tuple[tuple[str, RankingConfig], ...] = (
    (
        "recency only",
        _with(
            flat_significance=True,
            **{SIGNAL_INTEREST_AFFINITY: 0.0, SIGNAL_TREND_VELOCITY: 0.0},
        ),
    ),
    ("+ significance", _with(**{SIGNAL_INTEREST_AFFINITY: 0.0, SIGNAL_TREND_VELOCITY: 0.0})),
    ("+ affinity (shipped)", _with(**{SIGNAL_TREND_VELOCITY: 0.0})),
    ("+ trend", _with(**{SIGNAL_TREND_VELOCITY: 4.0})),
    ("no recency", _with(**{SIGNAL_RECENCY: 0.0, SIGNAL_TREND_VELOCITY: 0.0})),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    interests: tuple[str, ...]
    question: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("no interests at all", (), "what does a brand-new user see?"),
    Scenario("one niche follow", ("topic:rigging",), "can a single follow surface its show?"),
    Scenario(
        "two follows",
        ("topic:rigging", "topic:tyres"),
        "does a second follow WEAKEN the first? (the #19 dilution bug)",
    ),
    Scenario(
        "twenty follows",
        tuple(
            f"topic:{t}"
            for t in (
                "rigging tyres suspension braking lines strategy method compounding focus ridge "
                "pass col hot flat a b c d e f"
            ).split()
        ),
        "does breadth still personalise, or does it wash out?",
    ),
    Scenario(
        "follows the sparse show",
        ("topic:ridge", "topic:pass"),
        "can an unenriched show win on relevance? (the #23 coverage-bias bug)",
    ),
    Scenario("follows the hot topic", ("topic:hot",), "what does trend add on top of affinity?"),
)


def observe(corpus_root: Path, *, limit: int = 6) -> list[dict[str, object]]:
    """For every scenario × configuration, the ranked feed. Pure — no printing."""
    rows = build_catalog_rows_cumulative(corpus_root)
    out: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for label, config in CONFIGS:
            pool = build_discover_pool(
                rows, limit=limit, interests=list(scenario.interests), root=corpus_root
            )
            ranked = rank_discover(
                corpus_root, list(scenario.interests), pool, limit=limit, config=config
            )
            out.append(
                {
                    "scenario": scenario.name,
                    "question": scenario.question,
                    "config": label,
                    "order": [s.title for s in ranked],
                    "ids": [s.slug for s in ranked],
                }
            )
    return out


def _print_table(observations: list[dict[str, object]]) -> None:
    """One block per scenario; each row is one signal configuration.

    Rows identical to the row above print as ``= unchanged``. That is deliberate and load-bearing:
    the first version truncated each order to four titles, so two configurations could differ
    beyond the cut and READ as identical — the display hiding the very movement the table exists to
    show. Compared on the full ranked order, reported explicitly.
    """
    seen: set[str] = set()
    previous: list[str] = []
    for row in observations:
        scenario = str(row["scenario"])
        if scenario not in seen:
            seen.add(scenario)
            previous = []
            print(f"\nscenario: {scenario}")
            print(f"          ({row['question']})")
        order = [str(x) for x in list(row["order"])]
        if order == previous:
            print(f"    {str(row['config']):22} = unchanged")
        else:
            print(f"    {str(row['config']):22} {', '.join(order)}")
        previous = order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Write the scenario corpus here and keep it (default: a temporary directory).",
    )
    parser.add_argument("--json", action="store_true", help="Emit observations as JSON.")
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = write_corpus(Path(args.corpus) if args.corpus else Path(td) / "corpus")
        observations = observe(root)
        if args.json:
            print(json.dumps(observations, indent=2))
        else:
            print("=== ranking scenarios — what each signal actually does ===")
            print(
                f"    corpus: {sum(len(f.episodes) for f in CORPUS)} episodes, {len(CORPUS)} feeds"
            )
            _print_table(observations)
            print(
                "\n  Read DOWN a scenario to see what each signal changed. `= unchanged`\n"
                "  means that\n"
                "  signal did nothing HERE — a finding, not a gap in the table.\n"
                "\n  What this corpus currently shows:\n"
                "\n    * A user with NO interests gets pure recency, and no signal can change it.\n"
                "      `rank_discover` returns early on an empty interest set — the whole ranking\n"
                "      apparatus is inert until the first follow. Every row of that scenario is\n"
                "      `= unchanged` for that reason, not because the signals are weak.\n"
                "\n    * ONE follow surfaces its show, and a SECOND does not push it back\n"
                "      down (#19). Compare `one niche follow` with `two follows` — the show\n"
                "      holds the top two places in both.\n"
                "\n    * A sparse, unenriched show WINS on relevance when followed (#23): under\n"
                "      `+ affinity` the Field Notes episodes lead, despite carrying the least\n"
                "      enrichment in the corpus. Before per-feed normalisation, coverage decided.\n"
                "\n    * At TWENTY follows, affinity stops discriminating — `+ affinity` is\n"
                "      `= unchanged` from `+ significance`. Follow everything and the boost lands\n"
                "      on everything; saturation bounds the damage but cannot create a preference\n"
                "      the follows do not express.\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
