# RFC-088 Enrichment Layer — epic #1101 spec-vs-main audit (2026-07-05)

Chunk-by-chunk audit of the Enrichment Layer epic **#1101** (RFC-088) against what is
actually on `main`, triggered by the observation that #1101 + 7 chunk issues read OPEN
after PR #1141. Method: verify each chunk's deliverables exist on disk / in CI rather
than trust issue state.

## Verdict (read this first)

**RFC-088 was promoted Completed (2026-06-27) and the code is on `main`** — foundation, all
8 enrichers, query protocol, routes+viewer, profile matrix, ADR-104 Accepted, guides. But
two things the raw "Completed" status hides:

1. **Most chunks are done → their open issues are tracking debt.** Chunks 1, 2, 5, 6, 7, 8
   (#1103/#1104/#1107/#1108/#1109/#1110) are verified on main and were **closed
   2026-07-05**.
2. **The two *smart* enrichers' accuracy eval + sweep were explicitly deferred.** Chunks 3
   & 4 (#1105 topic_similarity, #1106 nli_contradiction) shipped the *enricher* + scorer
   *plumbing* (CI-smoke-tested on toy fixtures), but the real labelled gold sets + the
   autoresearch sweeps were deferred per the replan log — so these are **genuinely
   incomplete** and were **kept open**.

Net: this went through two wrong framings before landing here. Not "shipped out-of-band"
(first draft) and not "fully validated, pure tracking debt" (first correction). The
accurate middle: **shipped on-plan + promoted, with the smart-enricher *accuracy*
validation honestly deferred.**

Source of truth, both on `main`:
- `docs/rfc/RFC-088-*.md` header: *"Status: Completed (2026-06-27) — Epic #1101 shipped
  all 9 chunks"* (enumerates every chunk's deliverables).
- `docs/adr/ADR-104-enrichment-layer-boundary-vs-kg-direct.md`: *"Status: Accepted
  (promoted 2026-06-27 with RFC-088 chunks 0-8)."*

> **Update (2026-07-05): #1106 resolved.** The `nli_contradiction` accuracy eval was
> built and run — **0% precision** on prod-v2 (150-pair Opus silver; 0 true
> contradictions even under the broad definition). Two fixes shipped: a softmax
> calibration fix (`scorers/nli.py`; corpus flags ~660→~154) and **disabling the
> enricher** in all shipping profiles (0% precision → every surfaced pair is fabricated).
> Product goal moved to a stance-level detector (**#1144**). #1106 is CLOSED; **#1105**
> (topic_similarity) remains open.

## Chunk-by-chunk evidence

| Chunk | Issue | Issue state | Deliverable | Evidence on `main` | Verdict |
| --- | --- | --- | --- | --- | --- |
| 0 — ADR-104 boundary | #1102 | **CLOSED** | RFC-097↔088 boundary decision | `docs/adr/ADR-104-*.md` = **Accepted** | ✅ done + tracked |
| 1 — Foundation | #1103 | **CLOSED** (07-05) | protocol/registry/executor/paths/envelope/CLI + health/status/resilience/correlation + JSON Schema + jobs API + MCP source + API doc | all modules present in `src/podcast_scraper/enrichment/`; `docs/api/ENRICHMENT_LAYER_API.md`; lock audit `docs/wip/RFC-088-CHUNK1-LOCK-AUDIT.md` | ✅ done — closed |
| 2 — 6 deterministic enrichers | #1104 | **CLOSED** (07-05) | grounding_rate, insight_density, temporal_velocity, guest_coappearance, topic_cooccurrence_corpus, topic_theme_clusters + gold eval | 6 enrichers in `enrichers/`; `data/eval/enrichment/deterministic/gold/` (exact-match); scorer `enrichment_deterministic.py` | ✅ done — closed (deterministic → exact-match gold is real validation) |
| 3 — topic_similarity (embedding) | #1105 | **OPEN (held)** | enricher + eval + autoresearch sweep | enricher `topic_similarity.py` ✓; gold = **3-row `sample_gold.jsonl` stub**; **no `autoresearch/enrichment_topic_similarity/` sweep** | ⚠️ enricher shipped; **accuracy eval + sweep deferred** — kept open |
| 4 — nli_contradiction (ML) | #1106 | **CLOSED** (07-05) | enricher + NLI eval set + sweep; DeBERTa-v3-small CPU | eval ran on prod-v2 (150-pair Opus silver) → **0% precision**; softmax calibration fix shipped + enricher **disabled** in all profiles | ✅ resolved — bug fixed, enricher disabled; product goal → **#1144** |
| 5 — QueryEnricher protocol | #1107 | **CLOSED** (07-05) | query_protocol + query_topic_relatedness + /api/search enrich_results | `query_protocol.py`, `query_registry.py`, `query_enrichers/query_topic_relatedness.py` | ✅ done — closed |
| 6 — Routes + viewer consumption | #1108 | **CLOSED** (07-05) | corpus_enrichments routes; viewer Config Enrichment tab; Topic/Person rail signals | routes `app_enrichment.py`/`corpus_enrichments.py`/`enrichment.py`/`enrichment_config.py`; viewer `EnrichmentPanel.vue`, `EpisodeEnrichmentSection.vue`, `EnrichmentConfigEditor.vue`, `useEnrichmentEnvelopeCache.ts` | ✅ done — closed |
| 7 — Profile-preset wiring | #1109 | **CLOSED** (07-05) | EnricherSet on ProfilePreset + CLI overrides + drift gate | enricher wiring across `config/profiles/*.yaml`; matrix in RFC-088 | ✅ done — closed |
| 8 — Promotion + guide | #1110 | **CLOSED** (07-05) | RFC→Completed, ADR-104→Accepted, ENRICHMENT_LAYER_GUIDE | RFC = Completed; ADR = Accepted; `docs/guides/ENRICHMENT_LAYER_GUIDE.md` | ✅ done — closed |
| — Umbrella | #1101 | **CLOSED** (07-05) | the epic | code shipped + promoted; deferred accuracy work carried by #1105/#1106 | closed — #1105/#1106 stand alone |

## Validation: plumbing yes, smart-enricher accuracy deferred (the load-bearing check)

The eval *plumbing* runs in CI, but for the two smart enrichers it is a **smoke test on
stub fixtures**, not an accuracy gate:
- `tests/unit/enrichment/test_eval_scripts_smoke.py` invokes all three scorers against
  `data/eval/enrichment/*/gold/` and asserts scorer behaviour + exit-code contracts; it
  runs in `make ci-fast`.
- **Deterministic enrichers (chunk 2) are genuinely validated** — exact-match gold, no
  accuracy question.
- **The two *smart* enrichers are not.** Their gold sets are stubs — `sample_gold.jsonl`
  at 3 rows (topic_similarity) / 8 rows (nli_contradiction), vs the planned ~100 labelled
  NLI rows — and no autoresearch sweep tuned their params. This was **explicitly deferred**
  per `docs/wip/RFC-088-CHUNKS-2-8-REPLAN.md` → "Deferrals recorded post-chunk-5
  (2026-06-27)": *"stub scaffolding … gold fixtures empty … Real gold sets get populated as
  the corpus grows + the operator labels."*
- CI never calls the real NLI model (stub scorer), per `[[feedback_no_llm_in_ci]]`.

So the connections layer P3 consumes *produces* signals and its plumbing is tested, but
the ML/embedding signals' **accuracy is unmeasured** until #1105/#1106's deferred gold +
sweep land.

## Findings

1. **6 chunks were tracking debt → closed 2026-07-05.** #1103, #1104, #1107, #1108, #1109,
   #1110 shipped + verified on main; closed citing RFC-088 = Completed. (#1102 was already
   closed.)
2. **2 chunks are genuinely incomplete → kept open.** #1105 (topic_similarity) and #1106
   (nli_contradiction): enricher shipped, but the real labelled gold set + autoresearch
   sweep were deferred (see the validation section). Each now carries a status comment
   recording the exact remaining work (populate gold, build + run the sweep).
3. **Umbrella #1101 closed 2026-07-05.** Operator chose to close it and let #1105/#1106
   carry the deferred smart-enricher accuracy work solo.
4. **`make eval-enrichment-*` absence is intentional, not a gap.** Per the replan
   (REPLAN-O6) scoring scripts are called directly (matching the 39 existing
   `scripts/eval/score/*.py`); Make wrappers are reserved for the *sweep programs*
   (`make autoresearch-enrichment-*`), which don't exist yet precisely because the sweeps
   were deferred.

## Corrections to sibling docs

- `docs/wip/player/LEARNING-PLATFORM-GAP-ANALYSIS-2026-07.md` **Pivot 1** went through two
  wrong framings (out-of-band; then fully-validated tracking-debt). This audit is
  authoritative: RFC-088 shipped on-plan + promoted, most chunks are done (closed), but the
  two smart enrichers' **accuracy eval + sweep are deferred** (#1105/#1106 kept open). The
  gap-analysis doc's Pivot 1 is updated to match.
