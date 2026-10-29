# PRD-037: Corpus Scout (Corpus Curation & Feed Prospecting)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: standalone; serves the Knowledge Retention corpus precondition
- **Layer**: internal / Pro — builds + maintains the curated source registry (a moat asset).
- **Architecture**: `RFC-088-corpus-scout.md`
- **Serves precondition in**: `PRD-034-knowledge-retention-layer.md` §Corpus composition
- **Shares measurement with**: `RFC-087-simulation-validation-harness.md` (value-potential
  diagnostic)

> **Proposed numbering** — `PRD-037` / `RFC-088` are placeholders; verify and renumber.

---

## Summary

The Knowledge Retention model's value is gated on **corpus shape** (PRD-034 §Corpus
composition): high **overlap** *and* high **divergence**, in clusters, with bridge entities —
not raw size. That requirement is inert without a way to *act* on it. **Corpus Scout** is the
tool that does: it analyses the current corpus against ideal-shape targets, finds where it
falls short, and **advises which external RSS feeds to bring in to close the gap** — scoring
candidates by their *marginal* contribution **before** committing to full ingestion.

It is how the **curated source registry** is built and maintained, and it is **standalone** —
runnable independently of the live app.

---

## Background

We can now *measure* corpus value-potential (RFC-087: recurrence + dispersion). But measuring
is not curating. Three jobs remain unserved:

1. **Diagnose** — where is the corpus an echo chamber (high overlap, low divergence)? Which
   tracked entities under-recur? Where are clusters thin, stale, or unbridged?
2. **Prospect** — what's *out there* that would fix those specific gaps? And crucially, how to
   judge a candidate feed's contribution **without enriching the whole internet**?
3. **Advise** — rank candidates by marginal value-to-target and recommend, with rationale.

Corpus Scout is these three, plus a calibration loop that makes its prospecting sharper over
time.

---

## The central constraints (why this is a real tool, not a query)

- **Cheap proxy.** Full GIL/KG on every candidate is infeasible. Candidates are scored from
  **metadata → light NER**, then **sampled light enrichment**, with full enrichment spent only
  *after* an ingest decision (where it also feeds calibration). See RFC-088 §cheap-proxy.
- **Marginal-to-target.** A feed's worth = its contribution to the *current gap*. Redundant
  excellence scores low; a modest feed that fills a divergence gap scores high.
- **Divergence measured, never imposed.** Divergence is read **from content** (do sources make
  *differing claims about the same entities?*), **not** from external political-lean labels —
  importing such labels into an objectivization platform would bake in the very bias it exists
  to expose. *(Integrity-critical; RFC-088 §Divergence integrity.)*

---

## Goals

1. **Diagnose** the corpus against per-cluster ideal-shape **targets** (recurrence floor,
   dispersion threshold, bridge coverage, freshness SLO).
2. Turn gaps into a machine-readable **acquisition brief**.
3. **Discover** candidate feeds (directories, similarity, guest graph, frontier-entity search).
4. **Cheaply score** each candidate's marginal contribution to the brief, with confidence.
5. **Advise**: ranked recommendations + rationale; flag echo-only feeds.
6. **Calibrate** from realized post-ingest contribution → better prospecting over time.
7. Feed the **curated source registry** + hand accepted feeds to existing RSS ingest.

---

## Non-Goals

- **Not** an ingestion pipeline — it advises; ingestion is the existing RSS path.
- **Not** full enrichment of candidates — proxy estimation before ingest by design.
- **Not** auto-ingest without a human gate in v1.
- **Not** a political-lean classifier — divergence is content-measured, not label-imposed.
- **Not** coupled to the live app — standalone.

---

## Capabilities (requirements)

### Diagnose
**FR1.1** — Run the RFC-087 value-potential diagnostic over a corpus snapshot: per-cluster
recurrence histogram + dispersion read + bridge-entity map.
**FR1.2** — Compare against **targets** and produce a per-cluster **gap report** flagging echo
chambers (overlap ✓ / divergence ✗), under-recurring entities, thin/unbridged/stale clusters.

### Brief
**FR2.1** — Emit a structured **acquisition brief**: per cluster, `{ recurrence_on:[entities],
divergence_on:[entities], bridge_to:[clusters], freshness:Δ }`.

### Discover
**FR3.1** — Find candidate feeds via directories (Podcast Index/Apple), similarity expansion,
guest/co-occurrence graph, and frontier-entity search driven by the brief.
**FR3.2** — Canonicalise/dedup feeds (same show across feeds) before scoring.

### Score
**FR4.1** — Score each candidate's **marginal value vs the brief**: overlap, divergence,
bridge, extraction-amenability, freshness, minus a **redundancy (echo) penalty** — via the
cheap-proxy tiers, carrying confidence (which tier produced the estimate).

### Advise
**FR5.1** — Rank candidates by marginal value with a **rationale** ("fills divergence gap in
cluster X on A,B; bridges X↔Z") and confidence; explicitly flag **echo-only** feeds.
**FR5.2** — Output both machine-readable and human-readable acquisition reports.

### Register & calibrate
**FR6.1** — Write accepted recommendations to the **curated source registry**; hand accepted
feeds to existing RSS ingest; record rejections to avoid re-proposal.
**FR6.2** — After ingest, compare **realized** (full-diagnostic) contribution to **predicted**
(proxy) score and re-fit the scorer (calibration loop).

---

## Success Criteria

1. Given the real beta corpus + targets, the tool produces a **gap report** that correctly
   identifies its echo chambers and thin/unbridged clusters.
2. Given a candidate list, it produces a **ranked acquisition recommendation** with rationale
   and confidence, and flags echo-only feeds — **without** full enrichment.
3. Every divergence judgement is backed by **content evidence** (differing claims), logged and
   auditable; no political-lean labels drive scores.
4. After ingesting a recommended feed, the **calibration loop** reports predicted-vs-realized
   error and adjusts.
5. The tool runs **standalone** against a corpus snapshot.

---

## Dependencies

- **RFC-087 diagnostic** — measurement core (reused for both corpus and proxy estimation).
- **Light NER + light enrichment** profile — P-meta / P-sample scoring.
- **Podcast directory APIs** — discovery.
- **Existing RSS ingest + Podcast 2.0** — execution path for accepted feeds.
- **Curated source registry** — destination for recommendations.

---

## References

- `RFC-088-corpus-scout.md`
- `PRD-034-knowledge-retention-layer.md` (§Corpus composition)
- `RFC-087-simulation-validation-harness.md`
- `RFC-056-autoresearch-loop.md`
