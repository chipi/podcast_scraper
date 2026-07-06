# Fixture corpus as a full-product substrate — comprehensive analysis (#1148, expanded)

Requested analysis before authoring v3 content. The scope grew from "close the enricher eval
loop" to: **one fixture corpus that exercises every player + knowledge + recommendation feature
well, with natural conversation and full TTS**, doubling as a real end-to-end demo. This maps
every feature to the corpus material it needs, finds the gaps, and proposes a target shape.

## Two hard facts that reframe everything

1. **Only 3 of 9 generated feeds actually reach the built corpus.**
   `build_app_validation_corpus.py` wires `p05_investing`, `p02_software`, `p03_scuba` —
   `--max-feeds 3 --max-episodes-per-feed 2` = **6 episodes total**. The other six feeds
   (p01, p04, p06–p09) are generated transcripts that **no enricher and no player feature ever
   sees**. Umbrella cross-cutting topics are thin: p05→personal-finance, p02→systems-thinking,
   p03→safety-practices.

2. **Additive-only, preserve all detection targets.** Everything the pipeline is meant to
   detect — sponsor blocks, native ads, ASR garbles (incl. severe), nickname/alias variants,
   position arcs, low-grounding dialogue — **stays**. New enricher/feature material is woven in
   *on top*; we never override the previous notes.

**Net:** the corpus is far too small and too disconnected to exercise the product. 6 episodes
across 3 unrelated domains can't produce meaningful clusters, perspectives, discovery,
recommendations, or personalization — regardless of the six enricher structures.

## Feature → corpus material → current coverage → gap

Grouped by the consumer surfaces (`/api/app/*`) + operator knowledge views.

| Feature (surface) | Needs in the corpus | Today (6 eps / 3 shows) | Gap |
|---|---|---|---|
| **Catalog / Podcast / Library** (`/podcasts`,`/episodes`) | enough shows + episodes to browse | 3 shows ×2 | thin but works |
| **Player + transcript** (`/listen`,`/segments`,`/audio-source`) | transcript + segments + **audio** | text ✓, audio via TTS | fine (TTS covers audio) |
| **Insights / entities per episode** (`/episodes/{}/insights`,`/entities`) | insights with topics + speakers | present | ok |
| **Entity cards — Topic** (`/topics/{id}`) | a topic across **multiple episodes + speakers** | most topics appear once | **weak** — need recurrence |
| **Entity cards — Person** (`/persons/{id}`) | a person across **≥2 episodes**, with stances | few recurring guests | **weak** |
| **Perspectives** (`/topics/{id}/perspectives`) | **≥2 speakers, attributed insights, same topic** | absent → #1146 e2e mocked | **missing** |
| **Clusters** (`/clusters`) | a topic vocabulary with real semantic groups | 3 umbrellas only | **weak** — too few topics |
| **Enrichment signals** (`/corpus/enrichment`) | velocity / similarity / cooccurrence signal | thin, no time spread | **missing signal** |
| **Search — entity + passage** (`/search`,`/entities/search`) | enough entities/passages to rank | 6 eps → sparse | **weak** |
| **Discover / recommend** (`/discover`,`/discover/click`) | topics + episodes to recommend against interests | little to rank | **weak** |
| **Interests / personalized ranking** (`/interests`,`/interests/derived`,`/ranking-config`) | **seeded users** w/ playback+captures → interest tokens | e2e state only, not corpus | **missing** |
| **Highlights / notes / favorites** (`/highlights`,`/notes`,`/favorites`) | quotable insights + seeded captures | quotes exist, no seeded captures | **missing seeded** |
| **Resurfacing / consolidation** (`/resurfacing`) | per-user heard∪captured over real topics | none seeded | **missing** |
| **Digest / trending** (velocity) | episodes **dated across months** on recurring topics | one-snapshot dates | **missing time** |
| **scope=mine lens** (#1149) | seeded users whose heard set covers shared topics | none | **missing** |

## The topic / knowledge / recommendation work — do we need more material?

**Yes, materially.** Everything we shipped in Epic 3 (clusters, person/topic entity cards,
multi-perspective, entity search, personalized discovery, ranking) is **starved** by a 6-episode,
3-unrelated-domain corpus:

- **Clusters + similarity + cooccurrence** need a *denser topic web* — the same topics recurring
  across shows/speakers. Three umbrellas over three domains barely cluster.
- **Perspectives + person cards + positions** need *the same people/topics across episodes* with
  attributed, sometimes-opposing stances. Nearly absent today.
- **Discovery + recommendation + ranking** need *seeded users with interests* and *enough
  episodes on overlapping topics* to rank meaningfully. The ranking path is flag-off partly
  because there's nothing to rank (#1139).
- **Digest / trending / prediction-tracking** need *time spread*. Single-snapshot dates give a
  flat signal.

## Recommended target corpus shape

Grow the **wired** set (not just generated) and connect it:

- **~5–6 shows × ~3–4 episodes ≈ 18–24 episodes** wired into `build_app_validation_corpus`
  (raise `--max-feeds`/`--max-episodes-per-feed`; pick the 5–6 with the richest overlap
  potential — p05 investing, p02 software, p03 scuba, p04 photography, p01 biking, + one panel).
- **Overlap web anchored on `risk-management` ↔ `systems-thinking`** (a *similar pair* so
  topic_similarity has a real neighbour), recurring across ≥3 shows/speakers with **attributed
  claims** → perspectives + cooccurrence + person cards + clusters, all at once.
- **The six enricher structures** woven in *naturally* (extend episodes, don't bolt on): time
  spread (dated across ~6 months), cross-speaker overlap, one **panel episode** (2 guests →
  co-appearance) carrying the **diversification-vs-concentration opposition** (the nli/
  disagreement positive), grounding/density variation.
- **Seeded users + captures** (`build_v3_corpus_meta.seeded_users`): 2–3 users whose
  heard∪captured sets cover the overlap topics → interests, personalized ranking, resurfacing,
  scope=mine — all become demonstrable.
- **Preserve every detection target** (sponsor/native-ad/garble/position-arc) — additive only.
- **TTS**: author dialogue that unfolds naturally so the ElevenLabs voices sound like a real
  show; the existing per-episode voice/accent hints carry over.

**The six enricher structures are necessary but not sufficient** — full-feature coverage also
needs the wired-set growth, seeded users+captures, and the denser topic web above.

## Sequencing + open decisions

1. **This analysis** — feature×material gap + target shape.
2. **Target corpus design** — the exact shows/episodes/topics/overlap/users to author (next doc).
3. **Author** — extend the wired shows' episodes naturally + the panel + gold; grow the wiring.
4. **Regenerate + run loop** — enrichers → scorers → real gate_metrics; **player e2e over the new
   corpus** to prove every feature lights up.
5. **TTS** — voice the new/extended episodes.

## Target corpus blueprint (LOCKED)

Decisions: **6 shows × 4 episodes = 24 wired**; promote **p04 photography, p01 biking, p07
sustainability** alongside p05/p02/p03; **3 seeded users**; **panel = new episodes within
existing shows (no new podcast)**. Anchor pair: `risk-management ↔ systems-thinking`.

New episodes to author = **6** (p04 already has 4); existing episodes get attributed
`topic_claims` + `publish_offset_days` woven in naturally (additive — sponsor/garble/position-arc
all preserved).

| Show (domain) | Existing eps | + New eps | Overlap role |
|---|---|---|---|
| **p05** investing | e01 index, e02 monetary, e03 macro | **e04 = "The Risk Panel"** (2 guests, diversify-vs-concentrate opposition + co-appearance) | risk-management hub |
| **p02** software | e01 on-call, e02 eng-comm, e03 security | **e04 = systems-thinking/reliability** (Priya) | systems-thinking hub |
| **p01** biking | e01 trail, e02 enduro, e03 drivetrain | **e04 = risk in racing** (Sophie → risk-management from a surprising domain) | cross-domain similarity |
| **p03** scuba | e01 wreck, e02/e03 marine | **e04 = dive risk-management** (Marco) | cross-domain risk |
| **p04** photography | e01–e04 (has roundtable) | — (extend e03 lighting → "light as a system" systems-thinking claim) | visual/systems link |
| **p07** sustainability | e01 sustainability, e02 macro | **e05, e06 = systems-thinking + risk** (Elena, systems thinker) | systems-thinking anchor |

**Time spread:** stamp `publish_offset_days` so the risk-management/systems-thinking episodes
span ~0–180 days → velocity/trending signal.

**3 seeded users** (`build_v3_corpus_meta.seeded_users`):
- `u_risk` — heard risk-management episodes across p05/p02/p01 → interests {risk-management,
  reliability, systems-thinking}; drives scope=mine perspectives on risk-management.
- `u_invest` — heard p05 only → interests {index-investing, monetary-policy, macroeconomics}.
- `u_field` — heard p01/p03/p04 → interests {trail-building, dive-planning, frame}.

**Wiring change:** `build_app_validation_corpus` → `--max-feeds 6 --max-episodes-per-feed 4` +
add p04/p01/p07 to `APP_SHOWS`.

**Gold:** per-episode + corpus `expected_enrichment` for each enricher, reconciled by the loop
run (deterministic exact; embedding/NLI measured, eval-first).

## Execution order (creative content is the critical path)

1. Author `build_v3_corpus_meta` (shared_topics, contradiction_pairs, 3 users, corpus gold).
2. Author the 6 new episodes (panel first — the centerpiece + quality bar) + weave attributed
   claims/time into existing episodes.
3. Regenerate v3 → bump `build_app_validation_corpus` wiring → build corpus.
4. Run loop (enrichers → scorers → real gate_metrics) + player e2e over the new corpus.
5. TTS the new/extended episodes.
