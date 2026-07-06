# Evolving the local/fixture corpus for the complex enrichers

Living notes (started 2026-07-06). **Scope — read first:** *fixtures are for TESTS
(deterministic CI / e2e), NOT eval.* Eval uses the **prod-v2 corpus (100 real episodes)** for
now; growing *that* with more real shows is a separate concern →
`docs/wip/ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md`. **This doc is about evolving the committed
TEST fixtures to a v3 that can exercise the complex, cross-person enrichers deterministically.**
Tracked as **#1148**.

This session established — with numbers, not hunches — that the current test fixtures are too
thin and interview-shaped to exercise the complex enrichers (contradiction / disagreement /
multi-perspective). We *measured* this against prod-v2 (the eval corpus); the committed test
fixture is thinner still.

The relevant corpora:

- `tests/fixtures/app-validation-corpus/v2` — **committed TEST fixture** (e2e / CI). ← evolve to **v3**.
- `.test_outputs/manual/prod-v2/corpus` — **eval corpus, 100 real episodes** (evals, screenshots). *Not* a test fixture; grown via real-show onboarding.

---

## What we measured this session (the evidence)

| Enricher / feature | Corpus | Finding | Root cause |
| --- | --- | --- | --- |
| `nli_contradiction` (#1106) | prod-v2 | **0 true contradictions in 150 silver-labelled cross-person pairs**; enricher precision ~0% | Cross-person *atomic-insight* contradictions are near-absent |
| stance disagreement (#1144) | prod-v2 | **0 disagree / 9 agree / 31 no-shared-question in 40 stance pairs** | **Interview format** — guests are on *separate* episodes, never in dialogue, never engage the same proposition |
| `topic_perspectives` (#1146) | prod-v2 | **Rich** — e.g. AI development = 10 speakers, AI agents = 9 | Multi-*perspective* (different facets) IS abundant; multi-*opposition* is not |
| `topic_perspectives` (#1146) | validation corpus | **0 topics with even 1 speaker-attributed perspective** | Committed fixture GI lacks the Insight→SUPPORTED_BY→Quote→SPOKEN_BY→Person + ABOUT chain at any density → e2e had to mock |
| `topic_similarity` (#1105) | prod-v2 | 833 topics with neighbours (usable); stub gold has fictional neighbours | Vocabulary is fine; the *gold* was never populated |

**The through-line:** the fixtures give us *breadth of perspective* but no *opposition* and no
*time*. Interview podcasts are monologue-shaped per episode. Real disagreement + prediction-
tracking need speakers **engaging the same proposition**, which only shows up with (a) debate/
dialogue content, or (b) **scale + a time span** (the same question answered differently over
years — see #1144's "scale-gated" reframe).

---

## What each complex enricher needs from the corpus

- **`nli_contradiction` / stance-disagreement detector (#1144, scale-gated):** cross-person
  claims on the *same proposition* with opposing stances. Needs debate/panel/dialogue content
  OR many episodes over time on recurring contested questions ("is inflation transitory?",
  "will AGI arrive by 2030?").

- **`topic_perspectives` (#1146):** multiple speakers per topic, each speaker-attributed. The
  fixtures HAVE this at prod-v2 scale; the **committed validation corpus does not** (0
  perspectives) → e2e mocks. Needs ≥1 topic with ≥2 speaker-attributed insights baked in.

- **`guest_coappearance`:** multi-guest episodes (two people on the same episode). Interview
  format is mostly single-guest → thin.

- **`temporal_velocity`:** episodes distributed across a **time span** on recurring topics.
- **`topic_similarity` (#1105):** enough topics for genuine semantic clusters (fine at prod-v2).

---

## Evolution directions (two tracks)

### Track A — committed fixture for deterministic e2e/CI

The enrichers' *rendering + accuracy* should be testable without a huge corpus. Options:

1. **Bake a small, hand-authored "rich pocket"** into the validation corpus GI: one topic with
   2–3 speaker-attributed insights (unlocks a real, un-mocked `topic_perspectives` e2e); one
   genuine cross-person contradiction pair (a real `nli_contradiction` positive); a multi-guest
   episode (for `guest_coappearance`).

2. **Synthetic-but-labelled injection**: extend the fixture builder (`tests/fixtures/*/build_fixture.py`)
   to emit controlled GI structures with *known* gold labels, so enricher accuracy is measured
   deterministically in CI (not dependent on the vagaries of real episodes).

### Track B — realistic richness for the EVAL corpus → separate note

Real-content richness (dialogue/debate shows, time span, multi-guest, scale) is about growing
the **eval** corpus (prod-v2, real episodes), *not* the test fixtures. Moved to its own note:
**`docs/wip/ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md`**. Keep this doc scoped to the v3 *test*
fixtures.

---

## Open questions / next steps

- Which track first? Track A (committed rich pocket) unblocks honest e2e/CI *now*; Track B is the
  bigger, longer play behind #1144.

- Do we author the "rich pocket" by hand, or teach the fixture builder to synthesize it?
- **Builder ownership (answered):** the committed corpus is built by `scripts/build_app_validation_corpus.py`; there is already a `scripts/build_synthetic_validation_corpus.py` — the natural home for Track A's synthetic-but-labelled injection. Per-fixture `tests/fixtures/*/build_fixture.py` exist for smaller connectivity fixtures. Track A likely extends the synthetic builder.
- Relationship to #1105/#1144: this corpus work is arguably a *prerequisite* for meaningfully
  finishing either — flag when we return to them.

*(Notes doc — extend freely.)*
