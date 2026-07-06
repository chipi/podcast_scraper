# Learning Platform — gap analysis & plan refresh (2026-07-05)

Refresh of umbrella epic **#1062** against shipped reality after PR #1141 (Epic 3 +
node-view + graph analytics) merged. Answers three questions: where are we, what's
left to reach the PRD-035 vision, and where did execution diverge from the plan
(pivots to call out). Read against `PRD-035-learning-platform.md` (the North Star)
and the umbrella #1062.

> **Correction (2026-07-05, post-audit):** Pivot 1 below (RFC-088 enrichment) went through
> two wrong framings before landing. The spec-vs-main audit —
> `docs/wip/RFC-088-ENRICHMENT-EPIC-1101-AUDIT-2026-07.md` — is authoritative: RFC-088
> shipped on-plan + promoted (2026-06-27); 6 chunks were tracking-debt (**closed
> 2026-07-05**), but the **two smart enrichers' accuracy eval + sweep were deferred**
> (#1105/#1106 kept open). Pivot 1, the ledger row, and the gap list are updated to match.

## 1. State of the platform — workstream ledger

The platform is far more complete than #1062 reads. Nearly the whole P0–P3 + Epic 3
arc has shipped; the umbrella body was never refreshed to show it.

| Workstream | Epic | Status (issue) | Status (reality) |
| --- | --- | --- | --- |
| P0 Foundation | #911 | OPEN | 9/10 shipped — only **#1069 scrape-on-demand** open |
| P1 Consumer App | #1077 | CLOSED | Shipped ✓ |
| P2 Capture | #1112 | CLOSED | Shipped ✓ (PRD-040) |
| P3 Consolidation | #1113 | CLOSED | Shipped ✓ (PRD-041) |
| Epic 3 Knowledge + Personalized Discovery | #1093 | CLOSED | Shipped ✓ (PRD-043) — **not in #1062's phase map** |
| — 3.5 Personalized discovery | #1098 | CLOSED | Shipped **but dark** — see Pivot 2 |
| Consumer Home + corpus search | #1090 | CLOSED | Shipped ✓ — also outside #1062 |
| Enrichment Layer (RFC-088) | #1101 | CLOSED | Promoted 2026-06-27; umbrella + 6 chunks closed. **#1106 resolved 07-05** (eval → 0% precision; softmax fix + enricher disabled; → #1144). **#1105** (topic_similarity) resolved 07-06 — validated (recall@10 99%), default top_k retuned 10->7. |

Against the PRD-035 goals: Spotify-grade player ✓, intelligence-during-listening ✓,
capture ✓, consolidation ✓, minimal multi-user foundation ✓, a11y+i18n-from-line-1 ✓
(cross-cutting DoD on #1077). The *functional* vision is essentially delivered.

## 2. Gap analysis — what's left to reach the vision

Two items are genuine build work; the rest are decisions or known future cliffs.
Numbered for cross-reference.

1. **Build — `rank_discover` offline eval harness.** The single gate that unblocks *both*
   flipping `APP_PERSONALIZED_RANKING` on (Pivot 2) and #1139 step 1. Without it,
   personalized discovery stays code-complete but off.
2. **Build — `topic_similarity` accuracy eval (#1105).** (**#1106 nli_contradiction:
   DONE 07-05** — eval built + run → 0% precision; softmax fix shipped; enricher
   disabled in all profiles; product goal → stance-level detector **#1144**.) #1105
   remains: populate a real gold set + measure recall@K for `topic_similarity`; sweep
   `top_k` only if the baseline is weak.
3. **Decision — #1069 scrape-on-demand + guardrails.** The only open P0 task, and its
   scope is now questionable: #1077 parked "Discovery" onto #1069, but discovery shipped
   over the *local* corpus without it. Decide keep-as-P0 / re-scope / close.
4. **Decision — #1139 derived-interests → ranking.** Deliberately parked pending the
   eval in (1).
5. **Future cliff — persistence.** PRD-035 D2 locks per-user state = plain files, "a real
   persistence layer is a separate future effort." The moat is "a corpus that grows over
   weeks/months"; plain-files-per-user will strain as captures + history + resurfacing
   state accumulate. Not a v2.7 gap; a scaling decision to schedule before real user load.
6. **Future cliff — voice control.** North-star, deferred by PRD-035. Not a gap.

## 3. Pivots — where execution diverged from the plan

### Pivot 1 — RFC-088 Enrichment: mostly done, smart-enricher accuracy deferred *(corrected twice)*

Two wrong framings before this landed. First draft: "shipped out-of-band, unvalidated
moat" — wrong. First correction: "fully validated, pure tracking debt" — also wrong. The
audit (`docs/wip/RFC-088-ENRICHMENT-EPIC-1101-AUDIT-2026-07.md`) is authoritative.

**What's true:** RFC-088 shipped on-plan + promoted Completed (2026-06-27) — foundation,
all 8 enrichers, query protocol, routes+viewer, profile matrix, ADR-104 Accepted, guides.
Chunks 1/2/5/6/7/8 were done → **closed 2026-07-05** (they were tracking debt). Deterministic
enrichers are genuinely validated (exact-match gold).

**The real residue:** the two *smart* enrichers — `topic_similarity` (#1105) and
`nli_contradiction` (#1106) — shipped the enricher + scorer plumbing (CI-smoke-tested on
3-row / 8-row stub fixtures), but their **real labelled gold sets (~100 NLI rows) +
autoresearch sweeps were explicitly deferred** (`RFC-088-CHUNKS-2-8-REPLAN.md`, "Deferrals
recorded post-chunk-5"). So the connections moat *produces* ML/embedding signals whose
**accuracy is unmeasured**. #1105/#1106 kept open with status comments. This is exactly the
"un-tunable stack" risk #1139 flags, sitting in the connections layer — smaller than first
feared, but real (→ gap item 2).

### Pivot 2 — Personalization shipped but dark

Epic 3.5 (#1098) delivered explicit-interest cluster-affinity ranking, but it is flag-gated **off**
(`APP_PERSONALIZED_RANKING`, "gated until the score is tuned"), and #1139 parks the
derived-interest half pending a `rank_discover` eval that doesn't exist. A headline
vision item is code-complete yet reaching no users, with no eval to turn it on
responsibly. The fix is gap (2) — build the eval; it is the same gate for both halves.

### Pivot 3 — the umbrella never absorbed Epic 3 or Home *(bookkeeping)*

Epic 3 (#1093, Knowledge + Personalization, merging the originally-separate Epic 3 + Epic 4)
and Home (#1090) shipped entirely outside #1062's P0–P3 phase map; the umbrella still marks
P2/P3 "later — re-plan after #1077 ships." The plan-of-record no longer describes the
product. Refresh #1062: check P1/P2/P3 done, add Epic 3 + Home as shipped workstreams,
mark P0 = only #1069 remaining.

### Pivot 4 — Discovery (PRD-037) fragmented, no owning epic *(minor)*

PRD-037 was split across #1098 (ranking), #1090 (home/search), and #1069 (scrape-on-
demand, unbuilt), with no single tracker. Reconcile PRD-037 against what actually
shipped and fold the remainder into the #1069 decision.

## 4. Recommended sequence

1. **Build the `rank_discover` offline eval** — unblocks personalization (Pivot 2) and
   #1139 step 1 in a single move.
2. **Finish #1105 topic_similarity accuracy** — populate a real gold set + measure
   recall@K (#1106 nli_contradiction resolved 07-05: 0% precision → softmax fix +
   disabled; product goal → stance-level detector #1144).
3. **Decide #1069** — keep-scoped / re-scope / close, with PRD-037 reconciliation.
4. **Refresh #1062** — make the plan-of-record match the shipped product.
5. **Schedule the persistence decision** before real user load (future cliff, not v2.7).

Items 1–2 are the work that converts "functionally shipped" into "validated + live"; the
rest is bookkeeping plus one product decision.
