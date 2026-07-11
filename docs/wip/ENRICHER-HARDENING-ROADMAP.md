# Enricher hardening roadmap — fixtures → coverage → surfaces (v1)

**Objective:** eliminate every `gap` / `weak` / `partial` from the enricher surface-area + cross-link
coverage (see `PROD-V2-GOLDEN-WALKTHROUGH.md` §5–6). Every artifact, surface, and cross-link must be
**≥ good, ideally excellent**.

**Coverage bar (the rubric we grade against):**

| Grade | Means |
|---|---|
| weak / gap | unit-only, config-only, or vitest-**mock**-only; no fixture-data assertion |
| **good** | integration test asserts real artifact/contract **+** component (vitest) renders it |
| **excellent** | good **+** e2e / tier-3 drives the surface against a **served fixture corpus** |

**Sequencing (your ordering — it's the right dependency order):** fixtures unblock real assertions →
assertions lock behaviour → only then build the thin surfaces on a tested base.

---

## Phase 1 — Fixture corpus completeness  → **PR-A**

The e2e/tier-3 fixture `tests/fixtures/app-validation-corpus/v3` is built by
`scripts/build_app_validation_corpus.py` — a **deterministic, no-ML** builder that *hand-authors* the
enrichment envelopes to mirror the real enrichers (e.g. `_insight_sentiment_envelope`,
`_temporal_velocity_data`). Closing the gaps = authoring the missing envelopes there (byte-stable).

| # | Item | Detail | DoD |
|---|---|---|---|
| 1.1 | Author `topic_similarity.json` | new `_topic_similarity_data()` — top-k neighbours per topic (deterministic cosine from the corpus's own topic set) | v3 has `enrichments/topic_similarity.json` with ≥1 topic having ≥3 neighbours |
| 1.2 | Author per-episode `topic_cooccurrence` sidecars | mirror the episode-scope enricher shape | every v3 episode has a `*.topic_cooccurrence.json` |
| 1.3 | Guarantee cross-link invariants in the fixture | ✅ cross-show **already met** (4 topics in ≥3 shows: lifelong-learning/expert-interviews span 9); ✅ consensus (10 pairs), sentiment spread, theme clusters, grounding + guest all present; ❌ **no multi-run feed** — add a **second `run_*`** of one existing episode to exercise the latest-run dedup | all invariants present |
| 1.4 | **Fixture-invariants test** | `tests/integration/.../test_app_validation_corpus_invariants.py` asserting each artifact + each cross-link exists in v3 | test green; fixture can't silently regress |

**"Full package?" (do we need new transcripts/audio?)** — mostly **no new content**:
- 1.1 / 1.2 are **envelope authoring on existing episodes** — no transcripts/audio.
- Cross-show is already satisfied — no new shows/episodes.
- The **only** new authored content is 1.3's **second run** of one episode. A v3 episode is a *full
  package*: `metadata` + `bridge`/`gi`/`kg` + `transcripts/<ep>.{txt,segments.json}` + enrichment
  sidecars + **inline data-URL audio** (the builder base64-inlines a tiny MP3 — **there are 0 real
  audio files**). So the second run must go through the deterministic builder to get the full package
  — but **no real audio/transcript files need sourcing**; it's all synthesized + byte-stable.

**Why first:** without these, `topic_similarity` + `topic_cooccurrence` surfaces can only ever be
mock-tested, and the dedup flow has no fixture to run against.

---

## Phase 2 — Coverage + assertions  → **PR-B** (emission) + **PR-C** (viewer/flow e2e)

### 2A — Enricher-*emission* integration tests  (PR-B)
Today we test scorers (unit), config schema, or vitest mocks — **not** "enricher runs → artifact is
correct". Add integration tests that assert the authored/real artifact's shape **and
non-degeneracy** on v3:

| Enricher | Current | Target |
|---|---|---|
| topic_similarity | config-only | int: neighbours present, scores in [0,1], symmetric-ish |
| grounding_rate | unit scorer only | int: rate computed, discriminates grounded vs not |
| guest_coappearance | vitest mock | int: pairs discovered across episodes |
| insight_density | envelope smoke | int: density per episode, non-zero |
| topic_theme_clusters | discovery int | int: clusters formed, membership sane |
| topic_cooccurrence(_corpus) | unit KG only | int: co-occurrence pairs + lift |
| topic_consensus / insight_sentiment | unit enricher ✓ | add int alongside (emission on v3) |

**DoD:** each enricher has an integration test asserting its artifact on the fixture corpus.

### 2B — Operator-viewer + cross-link e2e / tier-3  (PR-C)
The four operator surfaces are **vitest-only**. Add Playwright/stack coverage against a **served v3**
(tier-3 style, mirroring the consumer `perspectives.spec.ts` pattern):

- `TopicConversationArc` (weekly bars + week drill)
- `PositionTrackerPanel` (server rows + sentiment tint — the Fix-1 behaviour)
- `EnrichmentEdgesPanel` (similarity **and** consensus, on a topic **and** a person — the Fix-2 behaviour)
- `NodeEnrichmentSection` (Signals: velocity, mentions, consensus orientation)

Plus the **cross-link** flows with no e2e:
- cross-episode timeline **renders** ≥N episodes for a shared topic
- cross-person consensus **renders** on the operator topic node
- cross-show cluster **renders** in digest/library
- consumer `TopicConversationArc` e2e

**DoD:** every row of the §5 coverage matrix is **≥ good**, the ADR-108 surfaces are **excellent**
(e2e on served fixture), zero vitest-only surfaces remain.

---

## Phase 3 — New surfaces / thin links  → **PR-D**

> **Reconciliation (2026-07-09).** A direct read of both frontends before building PR-D found
> that **every "thin" PR-D surface already exists, renders, and is unit-tested.** The walkthrough's
> `partial`/`minimal` grades were **prod-v2 *data* absence** (surfaces rendered empty because the
> corpus hadn't been re-enriched), **not missing code** — the walkthrough explicitly tags them
> *v2 to-capture* (= "not screenshotted", see §5 header), never *v2 to-build*. After the 2026-07-09
> prod-v2 re-enrichment + the PR-A v3 fixtures, the data is present and the surfaces render.
> **18 vitest mount/unit tests pass** across the four surface components (proof below). So PR-D is a
> **capture + verify + reconcile** phase, not a build phase — building new surfaces would duplicate
> shipped, tested components (violates "do exactly what was asked").

| # | Item | Status (2026-07-09) |
|---|---|---|
| 3.1 | *v2 to-capture* surfaces | **Already shipped + tested.** insight_density strip → `EpisodeEnrichmentSection.vue` (op) + `EpisodeDensity.vue` (player); grounding_rate + guest_coappearance rows → `NodeEnrichmentSection.vue` (op, L191–207) + `EntitySignals.vue` (player, L78–92); topic_cooccurrence "co-occurs with" → `NodeEnrichmentSection.vue` (op, L173–185). Remaining = **capture screenshots** of these on the served corpus (evidence for the walkthrough), needs `make serve` + corpus. |
| 3.2 | Consensus-empty UX | **Decided: keep hide.** Empty consensus on most topics is *correct* (sparse-but-precise, §6). House pattern is best-effort hide; `NodeEnrichmentSection` already shows a catch-all "No enrichment signals for this person/topic" when *all* signals are absent. A consensus-specific "no corroboration yet" affordance would break house consistency (grounding/coappearance just hide) → **no code change.** |
| 3.3 | Cleanup | `nli_contradiction.json` stale leftover **removed** (prod-v2, 2026-07-09). Diarization ASR-name artifacts ("Speaker 02") → **#1167** (separate track, not enricher scope). |

**Surface → component → test (the proof PR-D's DoD is met in code):**

| Surface | Operator viewer | Consumer player | Vitest |
|---|---|---|---|
| insight_density strip | `EpisodeEnrichmentSection.vue` | `EpisodeDensity.vue` | `EpisodeEnrichmentSection.mount.test.ts`, `EpisodeDensity.test.ts` ✓ |
| grounding_rate row | `NodeEnrichmentSection.vue` L191–195 | `EntitySignals.vue` L150–155 | `NodeEnrichmentSection.test.ts`, `EntitySignals.test.ts` ✓ |
| guest_coappearance row | `NodeEnrichmentSection.vue` L196–207 | `EntitySignals.vue` L157–168 | `NodeEnrichmentSection.test.ts`, `EntitySignals.test.ts` ✓ |
| Config→Enrichment dialog | `EnrichmentConfigEditor.vue` (rich; schema-driven) | — | `EnrichmentConfigEditor.mount.test.ts` ✓ |

**DoD (met in code):** each surface has a real rendering **and** a test. Only outstanding item = the
walkthrough **screenshots** (evidence, not code) — optional, gated on bringing the backend up.

---

## Exhaustive closure matrix — every graded item → target

This is the contract: **B is not the whole test story.** Emission integration (PR-B) is one slice;
it pairs with fixtures (PR-A) and e2e/tier-3 (PR-C/D). Every `weak`/`gap`/`partial`/`minimal` row from
the walkthrough audit is listed here with the PR that lifts it to **good/excellent**. Nothing is left.

| Surface / flow | Now | → Target | Closed by |
|---|---|---|---|
| temporal_velocity | good | excellent | B (emission) + C (Signals e2e) |
| topic_similarity | **weak** | excellent | **A** (fixture artifact) + B + C |
| topic_consensus | **gap** | excellent | B (emission) + C (topic+person e2e) |
| insight_sentiment | good | excellent | B + C (arc/tint e2e) |
| insight_density | **weak** | good→excellent | B + D (episode strip e2e) |
| grounding_rate | **weak** | good | B (emission) + D (EntitySignals e2e) |
| guest_coappearance | **gap** | good | B + D |
| topic_cooccurrence(+_corpus) | **weak** | good | **A** (episode sidecars) + B |
| topic_theme_clusters | **partial** | excellent | B + C (cluster render e2e) |
| topic timeline (read) | good/**gap-e2e** | excellent | C (cross-episode render) |
| position arc (read) | excellent/**gap-e2e** | excellent | C |
| conversation arc (read) | good/**gap-e2e** | excellent | C (both viewers) |
| perspectives (read) | excellent | — | (already) |
| person profile/brief | good/**no-vitest** | excellent | B/C (+ add vitest) |
| op-viewer TopicConversationArc | **vitest-only** | excellent | C |
| op-viewer PositionTrackerPanel | **vitest-only** | excellent | C |
| op-viewer EnrichmentEdgesPanel | **vitest-only** | excellent | C |
| op-viewer NodeEnrichmentSection | **vitest-only** | excellent | C |
| op-viewer Config→Enrichment dialog | **minimal** | good | D |
| op-viewer Episode enrichment section | **minimal** | good | D |
| consumer TopicConversationArc | good/**gap-e2e** | excellent | C |
| EntitySignals / momentum | excellent | — | (already) |
| cross-episode render | **minimal** | excellent | C |
| cross-show cluster render | **partial** | excellent | C |
| cross-person consensus render | **weak** | excellent | B (emission) + C (render) |
| episode→GI→KG→bridge chain | good | good | (already; int + full-pipeline e2e) |

If a row isn't in PR-A/B/C/D above, it's already good/excellent and needs nothing.

---

## PR / branch plan

| PR | Phase | Branch / issue | Risk / effort |
|---|---|---|---|
| **A** | 1 (fixtures) | continue `fix/adr108-enricher-surfaces` (related) | low risk, med effort — deterministic authoring, no real audio |
| **B** | 2A (emission int) | `test/enricher-emission` | low-med — one integration test per enricher |
| **C** | 2B (viewer + flow e2e) | **[#1168](https://github.com/chipi/podcast_scraper/issues/1168)** (split out) | **med-high** — needs a new operator-viewer tier-3 harness |
| **D** | 3 (new surfaces) | `feat/enricher-surfaces-round2` | med |

**PR-C is #1168** — pulled out because the operator viewer has **no** served-corpus e2e harness today
(only vitest); the consumer app does (`app-validation` tier-3). Standing that up for gi-kg-viewer is
the single biggest lift, and it's blocked on PR-A (needs the `topic_similarity` fixture + multi-run
feed to drive real data).

---

## Definition of "done" for the whole roadmap
1. Fixture parity: v3 carries every enricher artifact prod-v2 has, guarded by an invariants test.
2. Every enricher has an emission integration test on the fixture.
3. Every enricher surface (operator + consumer) has e2e/tier-3 on a served fixture.
4. Every cross-link (episode / show / person) has an e2e that renders it.
5. No row in the coverage matrix reads gap / weak / partial.

---

## Full re-analysis (2026-07-09) — hunting more Phase-D-type mislabels

After PR-D, three parallel recon agents re-audited **every remaining graded-thin row** to find more of
the same pattern (graded thin, actually built+tested; the gap was *data/screenshot* absence, not code).
**The pattern is pervasive.**

**#1168 premise — MISLABELED.** The operator viewer is **not** missing an e2e harness. It has
`web/gi-kg-viewer/playwright.config.ts` (50+ mocked specs, `viewer-e2e` CI job in `python-app.yml`),
`playwright.validation.config.ts` (tier-3 real-corpus), and `tests/stack-test/*.spec.ts`
(Docker-served, run by `.github/workflows/stack-test.yml`: `stack-viewer`, `stack-person-profile`,
`stack-topic-entity`, `stack-enrichment-*`). The real gap is only that operator tier-3 runs
**post-gate** (stack-test) / operator-local (validation), not on the **fast PR path** like the
consumer's `app-e2e`. **#1168 shrinks: "wire operator tier-3 into the fast CI gate," not "build a
harness from scratch."**

**PR-C rows (11) — 10 MISLABELED, 1 genuinely thin.** All already built **and** vitest-tested (most +
e2e): TopicConversationArc, PositionTrackerPanel, EnrichmentEdgesPanel, TopicTimelineDialog (topic
timeline), position arc, conversation arc, perspectives, PersonLandingView (person profile — the
"no vitest" grade is **false**, 30+ tests), Storylines (cross-show cluster), NodeEnrichmentSection
(cross-person consensus). Only genuine thin spot: **cross-episode render has no *dedicated* test**
(covered only indirectly by `graph-expansion-mocks.spec.ts`).

**Genuine gaps found (small, real):**
1. **Dead surface** — consumer `EntitySignals.vue:96` renders a "disagreements" row from
   `nli_contradiction`, a **retired** enricher (replaced by `topic_consensus` per ADR-108); also
   `services/types.ts:421-430` + `EntitySignals.test.ts:27-37` still declare/mock it. Optional-chaining
   makes it graceful-degrade (benign at runtime) but it's stale. **Decision for operator:** delete the
   row, or repurpose it to `topic_consensus`? *(Left for review — won't silently drop a surface.)*
2. **cross-episode render** — add one dedicated test asserting a shared topic renders ≥N episodes.
3. **insight_sentiment** — *not* dark: no **frontend** reads the artifact directly, but it's consumed
   **server-side** by `cil_queries.py:_attach_sentiment` (L583-591), which tags each Insight with
   sentiment for the position/conversation-arc timelines. **Benign** (confirmed).

**Net:** the enricher surface area is in far better shape than the walkthrough grades implied. PR-A/B
closed the fixture + emission gaps; the surfaces and their tests already exist. #1168's real remaining
work is CI-wiring + the two small gaps above — not the large lift the roadmap assumed.
