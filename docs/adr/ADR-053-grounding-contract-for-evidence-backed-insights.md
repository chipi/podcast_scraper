# ADR-053: Grounding Contract for Evidence-Backed Insights

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md)
- **Related PRDs**: [PRD-017](../prd/PRD-017-grounded-insight-layer.md)

## Context & Problem Statement

GIL produces insights (key takeaways) from podcast episodes. Unlike summaries, insights
claim to represent specific points made in the episode. Without a formal definition of
what "evidence-backed" means, insights are no more trustworthy than generated text. The
system needs a verifiable contract that defines when an insight is grounded, what
evidence looks like, and how consumers can audit the chain from claim to transcript.

## Decision

We adopt a **grounding contract** with the following rules:

1. **Every Insight has an explicit `grounded` boolean**: set after extraction, not
   assumed. `grounded = true` means ≥1 `SUPPORTED_BY` edge to a Quote node after
   filtering low-confidence candidates.
2. **Quotes are verbatim transcript spans**: a Quote node contains the exact text from
   the transcript with `char_start`, `char_end`, and optional `timestamp_start_ms` /
   `timestamp_end_ms`. Quotes are auditable against the original transcript file.
3. **Evidence chain**: Insight → `SUPPORTED_BY` → Quote → `EXTRACTED_FROM` →
   transcript. Every link is navigable and verifiable.
4. **Confidence and provenance**: ML-derived nodes carry `confidence`,
   `model_version`, and `prompt_version`. These separate "model certainty" from truth.
5. **Quality target**: ≥80% of insights should be grounded in a well-processed episode.
   Episodes below threshold are flagged with `gil_quality: "low"`.

## Rationale

- **Trust**: Without grounding, insights are indistinguishable from hallucination.
  The contract makes trust explicit and measurable.
- **Verifiability**: Consumers can click through from insight to quote to transcript
  position. This supports the viewer, CLI, and any future UI.
- **Quality measurement**: The `grounded` boolean enables quality metrics, filtering
  (`--grounded-only`), and search ranking by evidence strength.
- **Honest handling**: Ungrounded insights are not hidden — they are labeled honestly,
  letting consumers decide what to trust.

## Alternatives Considered

1. **Grounding as metadata, not a contract**: Rejected; without a formal definition,
   "grounded" becomes meaningless and inconsistent across episodes.
2. **All insights must be grounded (reject ungrounded)**: Rejected; extraction quality
   varies. Honest labeling is better than silent dropping.
3. **Probabilistic grounding score instead of boolean**: Rejected for v1; a binary
   `grounded` is simpler to consume and filter. Confidence score exists separately
   for fine-grained use.

## Consequences

- **Positive**: Clear trust model. Enables `--grounded-only` filtering across CLI,
  search, and viewer. Quality metrics are measurable and trackable.
- **Negative**: Extraction must include QA/NLI step to determine grounding, adding
  pipeline latency. Some insights will be honestly labeled as ungrounded.
- **Neutral**: Quote extraction requires extractive QA across all tiers (ml, hybrid,
  cloud) per RFC-049.

## Implementation Notes

- **Schema**: `gi.json` — Insight nodes have `properties.grounded` (bool),
  `properties.confidence` (float), `properties.model_version` (string)
- **Quote nodes**: `properties.text` (verbatim), `properties.transcript_ref`,
  `properties.char_start`, `properties.char_end`, `properties.timestamp_start_ms`
- **Edge**: `SUPPORTED_BY` from Insight to Quote; `EXTRACTED_FROM` from Quote to
  transcript
- **Quality gate**: `run_grounding_pass()` in GIL extraction; threshold configurable

## References

- [RFC-049: GIL Core — Grounding Contract](../rfc/RFC-049-grounded-insight-layer-core.md)
- [RFC-050: GIL Use Cases — Evidence-Backed Discovery](../rfc/RFC-050-grounded-insight-layer-use-cases.md)
- [ADR-051: Per-Episode JSON Artifacts](ADR-051-per-episode-json-artifacts-with-logical-union.md)
