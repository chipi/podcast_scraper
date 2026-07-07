# Doc-freshness audit + reconciliation programme (2026-07-07)

Operator-requested thorough review of **all** docs — architecture, tests, RFC, PRD, ADR,
guides — to make them match shipped code. 537 markdown files under `docs/`. This is the
tracking ledger; it is itself WIP (point-in-time) and gets removed when the sweep closes.

## Treatment by category (decision-record integrity preserved)

| Category | Count | Treatment |
| --- | --- | --- |
| `guides/` (core) | ~20 | **Rewrite current** — they describe the live system. |
| `architecture/`, `api/` | ~26 | **Rewrite current** — same. |
| `README.md`, `CONTRIBUTING.md` | 2 | **Rewrite current.** |
| `rfc/` | 103 | **Status-correct + body amendments** (dated design records — no history erasure). |
| `prd/` | 45 | **Status-correct + body amendments** where product behavior changed. |
| `adr/` | 107 | **Status-correct + add "Superseded by" pointers only.** Never rewrite; code-divergent decisions get a NEW superseding ADR. |
| `guides/eval-reports/`, `guides/performance-reports/` | 50 | **Leave** — point-in-time measurement records. |
| `wip/`, `releases/`, `incidents/` | ~78 | **Leave** — point-in-time; at most mark superseded. |

## The recent code changes docs may not reflect (the divergence checklist)

- Consumer app moved `app/` → `web/learning-player/` (slice 14 on main).
- transformers upgraded to **v5** + ML-architecture unification (`providers/ml/*` rewritten).
- Enrichment **accuracy gate** (`eval/admission` → `profile_sets._admit`) gates enrichers by
  measured precision. `nli_contradiction` (0%, #1106) + `stance_disagreement` (0%, #1144) are
  **gated dark** (never run). The live disagreement/multi-speaker surface is `perspectives`
  (#1146), which is a CIL query, not an enricher.
- Real enricher set = 9: 6 deterministic (grounding_rate, guest_coappearance, insight_density,
  temporal_velocity, topic_cooccurrence_corpus, **topic_theme_clusters**), `topic_similarity`
  (embedding), `nli_contradiction` + `stance_disagreement` (ML, gated). There is **no**
  episode-scope `topic_cooccurrence` enricher (older docs claim one).
- Enrichment invoked as `python -m podcast_scraper.cli enrich` (peer of the pipeline).
- The `ingest` primitive was built then **dropped** — the single-feed pipeline *is* ingestion.
- Jobs registry spawns each job's **own** stored command (`argv_from_record`), not always the pipeline.

## Method

1. Parallel audit fan-out (5 read-only agents, one per category) → per-doc staleness findings.
2. Synthesize findings into the progress table below.
3. Fix in waves; reference docs first (highest value), then RFC/PRD bodies, then ADR status/supersession.
4. `make docs` (mkdocs strict) after each wave; commit per wave.

## Audit result (5 parallel auditors, 2026-07-07)

**Headline: the corpus is well-maintained** — most reference docs were reconciled in prior
sessions, and the RFC audit found **zero silent body divergences**. Findings are mostly status
fields, a couple of stale paths, one filename collision, and a few decisions that lack an ADR.

### Fixed this wave

| Doc | Fix |
| --- | --- |
| `guides/ENRICHMENT_LAYER_GUIDE.md` | Real 9-enricher table (dropped non-existent episode `topic_cooccurrence`, added `topic_theme_clusters` + `stance_disagreement`); new accuracy-gate section; per-enricher reference for the two new ones; profile matrix reframed as gate-filtered candidates; `topic_similarity` validation + top_k=7. |
| `guides/testing-strategy-ml.md` | transformers `>=4.40` → `>=5.0.0`. |
| `api/index.md` | Server-Guide row said `web/learning-player/` route arch → `web/gi-kg-viewer/`. |
| `web/learning-player/README.md` | `cd app` → `cd web/learning-player`. |
| `adr/ADR-060`, `adr/ADR-061` | Status `Accepted` → `Superseded by ADR-099` (pointer already present; field lagged). |
| `adr/ADR-077` | Malformed `**Status:**` → list-form `- **Status**:`. |
| `rfc/RFC-096` | Status `Draft` → `Completed` (audio pipeline separation shipped). |
| `rfc/RFC-098`, `rfc/RFC-099`, `rfc/RFC-100` | Status `Draft` → `In Progress` (consumer platform shipped in phases). |
| `prd/PRD-041` | Goals listed "contradictions" as a delivered signal — amended (gated dark; perspectives is the live surface). |
| `prd/PRD-037` | FR1.3a related-topics-in-search amended (shipped on entity cards, not Discovery search). |

### Verified-and-left-alone (agent suggested, I checked, no change needed)

- **ADR-067** — agent thought ADR-068 supersedes it. It doesn't: ADR-068 references it ("see ADR-067")
  but supersedes the prior *authority*, not ADR-067's Pegasus-retirement decision. Left as Accepted.
- **PRD-026** status "Implemented" is accurate (operator Topic Entity View). Minor consumer/operator
  wording nuance only.
- **RFC-082 / RFC-089** use a `> **Status:**` blockquote vs the list form — cosmetic, left.

### Open — need an operator decision (not auto-applied)

1. **ADR-099 number collision.** Two files are both `ADR-099`
   (`...-lancedb-first-single-index-search.md` + `...-response-shape-guardrails-for-self-deployed-services.md`).
   The guardrails one should be renumbered (next free = **ADR-105**), but it has **19 references across 5
   docs** — a rename + ref-update is a discrete, careful task. Awaiting go.
2. **Three shipped decisions lack an ADR** (agent flagged; per the RFC/ADR discipline these warrant a
   *new* ADR, never editing an old one): transformers-v5 ML unification, ingest-primitive drop
   (pipeline-is-ingestion), and the NLI/stance enrichers gated dark. Draft these as ADR-105/106/107?

## Progress

| Wave | Category | Status |
| --- | --- | --- |
| 0 | Enrichment guide | ✅ done |
| A | Guides + architecture + api + README | ✅ audited + fixed (2 stale paths + 1 version) |
| B | RFC status + bodies | ✅ audited (0 body divergence) + 4 status fixes |
| C | PRD status + bodies | ✅ audited (0 status errors) + 2 body amendments |
| D | ADR status + supersession | ✅ audited + 3 status fixes; ADR-099 rename + 3 new ADRs pending decision |
