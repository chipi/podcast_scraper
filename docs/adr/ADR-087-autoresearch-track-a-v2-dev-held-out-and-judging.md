# ADR-087: Autoresearch Track A v2 — Dev or Held-Out Split and Judging

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-073 (Autoresearch v2)](../rfc/RFC-073-autoresearch-v2-framework.md),
  [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md)
- **Related ADRs**: [ADR-073](ADR-073-rfc057-autoresearch-closure.md) (v1 closure and champions)

## Context & Problem Statement

RFC-057 Track A (prompt tuning) shipped a working ratchet, but v1 had structural issues: **smoke ⊂
benchmark** contaminated validation; **binary-OR judge contestation** flipped entire runs on a
single noisy episode; **`temperature=0`** on OpenAI was not determinism; bundled JSON formatting
biased judges. Champions could look strong on dev and fail on slightly different held data.

## Decision

Track A adopts a **v2 framework** (thin harness changes + new datasets or silvers), with these
**immutable rules**:

1. **Disjoint dev vs held-out episodes** — Iteration scores on **`curated_5feeds_dev_v1`** (e01+e02
   per feed); champion validation on **`curated_5feeds_benchmark_v2`** (e03 per feed only). **Zero
   overlap** between tuning and final validation sets. v1 datasets remain frozen for history
   (**ADR-073**).

2. **Fraction-based contestation** — A run is contested only when **more than 40%** of scored
   episodes exceed the judge delta threshold (replacing v1 **any-episode** OR).

3. **Efficiency rubric dimension** — Replaces **Conciseness** so longer, information-dense summaries
   are not penalised by default.

4. **JSON prose extraction before judging** — Bundled outputs are flattened to plain prose (title,
   summary, bullets) before judge calls so formatting does not dominate scores.

5. **Seed plumbing (partial mitigation)** — OpenAI **`seed`** is wired through config and factory
   for summarization experiments; documented as **variance reduction**, not byte-identical
   reproducibility.

6. **Non-goals unchanged** — Track B (ML parameter sweep) stays on RFC-057 machinery; v2 does not
   add multi-run averaging or statistical intervals in the first shipping slice.

## Rationale

- **Honest generalisation signal** — Held-out catches overfitting that shared-episode validation
   hides.
- **Stable promotion** — Fraction contestation reduces judge-noise false rejects.
- **Preserves v1 artifacts** — ADR-073 remains the audit trail for shipped v1 champions; v2 runs
   alongside without rewriting history.

## Alternatives Considered

1. **Keep v1 split and only add seeds** — Insufficient; subset contamination remained.
2. **Larger held-out only, no rubric change** — Improves signal but leaves length bias and JSON
   judge artifacts.
3. **Replace ROUGE entirely with LLM judges** — Rejected for v2 scope; rubric + ROUGE hybrid stays
   per RFC-073.

## Consequences

- **Positive**: New champion work must cite dev + held-out scores with the v2 config family.
- **Negative**: More datasets and silver references to maintain; operators must use the v2 YAML
   matrix from the RFC.
- **Neutral**: Future statistical tightening is explicitly deferred in the RFC.

## Implementation Notes

- **Code**: `autoresearch/` (orchestrator, scoring), `data/eval/configs/**/autoresearch_*_v2.yaml`,
  provider **`seed`** wiring as documented in RFC-073
- **Ops**: `autoresearch/README.md`, `autoresearch/JUDGING.md`

## References

- [RFC-073: Autoresearch v2 framework](../rfc/RFC-073-autoresearch-v2-framework.md)
- [ADR-073: RFC-057 autoresearch closure](ADR-073-rfc057-autoresearch-closure.md)
